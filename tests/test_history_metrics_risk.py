import math

import numpy as np
import pytest

from seintel.eval.judge import parse_scores
from seintel.eval.metrics import brier, ece, ndcg_at_k, precision_at_k, roc_auc
from seintel.history.mining import (extract_snapshot, fix_labels, history_features, is_bugfix, is_source_file,
                                    load_history, make_splits)
from seintel.quality import hotspot
from seintel.risk.model import RiskModel


@pytest.mark.parametrize("msg,exp", [
    ("Fix off-by-one bug in helper", True), ("Fix typo in docs", False), ("Add feature", False),
    ("Merge pull request #3 from x/fix-bug", False), ("Handle empty header (closes #123)", True),
    ("Refactor\n\nFixes #77", True), ("Update (#456)", False), ("fixed regression in json", True)])
def test_is_bugfix(msg, exp):
    assert is_bugfix(msg) is exp


def test_is_source_file():
    assert is_source_file("src/flask/app.py")
    assert not is_source_file("tests/test_app.py") and not is_source_file("src/x/conftest.py")
    assert not is_source_file("docs/conf.py") and not is_source_file("README.md")
    assert not is_source_file("requests/packages/urllib3/connection.py")     # vendored code


def test_history_and_labels(git_repo):
    commits = load_history(str(git_repo))
    assert [c.bugfix for c in commits] == [False, True, False]
    paths = {"pkg/util.py", "pkg/core.py"}
    hf = history_features(commits, 0, paths)          # only the first commit is visible
    assert hf["pkg/util.py"]["n_commits"] == 1 and hf["pkg/util.py"]["prior_fixes"] == 0
    sp = make_splits(3, {"x": (1 / 3, 1.0)})["x"]
    assert sp.snapshot == 0 and sp.end == 2
    lab = fix_labels(commits, sp, paths)
    assert lab == {"pkg/util.py": 1, "pkg/core.py": 0}   # the typo fix is not a bug fix


def test_extract_snapshot(git_repo, tmp_path):
    commits = load_history(str(git_repo))
    d = extract_snapshot(git_repo, commits[0].sha, tmp_path / "snap")
    assert "n * 2\n" in (d / "pkg/util.py").read_text()      # content at the first commit, not HEAD


def test_ranking_metrics():
    rel = {"a": 2, "b": 0, "c": 1}
    assert precision_at_k(["a", "b", "c"], rel, 2) == 0.5
    assert ndcg_at_k(["a", "c", "b"], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "c", "a"], rel, 3) < 1.0


def test_prob_metrics():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, p) == 1.0
    assert brier(y, p) == pytest.approx(0.025)
    assert ece(y, np.full(4, 0.5)) == pytest.approx(0.0)
    assert math.isnan(roc_auc([0, 0], [0.1, 0.2]))


def _rows(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cc = int(rng.integers(1, 200))
        rows.append({"path": f"f{i}.py", "loc": cc * 5, "sloc": cc * 4, "cc_sum": cc, "cc_max": cc // 3 + 1,
                     "cc_mean": 2.0, "mi": 50.0, "n_blocks": 5, "n_commits": int(rng.integers(0, 50)), "churn": cc * 3,
                     "n_authors": 2, "prior_fixes": int(rng.integers(0, 5)), "recent_commits": 1, "age_days": 100,
                     "days_since_change": 10, "fan_in": int(rng.integers(0, 10)), "fan_out": 2, "pagerank": 0.1,
                     "dependents_2hop": 3, "calls_in": 1, "y": int(rng.random() < cc / 250), "n_fixes": 0})
        rows[-1]["n_fixes"] = rows[-1]["y"]
    return rows


def test_risk_model_roundtrip_and_calibration():
    train, tune, test = _rows(300, 0), _rows(200, 1), _rows(200, 2)
    m = RiskModel(["cc_sum", "sloc", "fan_in"]).fit(train, tune)
    p = m.predict(test)
    assert ((p >= 0) & (p <= 1)).all() and m.platt is not None
    m2 = RiskModel.from_json(m.to_json())
    assert np.allclose(m2.predict(test), p)
    assert len(m.contributions(test[0], 2)) == 2


def test_hotspot_rank_and_tuning():
    rows = _rows(50, 3)
    assert hotspot.rank(rows, "cc")[0] == max(rows, key=lambda r: r["cc_sum"])["path"]
    assert hotspot.rank(rows, "hotspot", (0.0, 0.0)) == hotspot.rank(rows, "cc")   # no graph weight = CC
    w, trials = hotspot.tune_weights({"r": rows})
    assert len(trials) == len(hotspot.WEIGHT_GRID) ** 2 and w in [(t["w_in"], t["w_out"]) for t in trials]


def test_parse_judge_scores():
    s = parse_scores('```json\n{"correctness": 15, "completeness": 12, "relevance": 18, "clarity": 25, '
                     '"reasoning": 0}\n```')
    assert s["clarity"] == 20 and s["reasoning"] == 1 and s["overall"] == 66
    assert parse_scores("no json") is None
