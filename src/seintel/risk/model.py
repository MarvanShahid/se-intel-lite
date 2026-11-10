"""Gap 3 component: calibrated, change-aware, graph-aware defect risk model.

- Logistic regression on log-transformed, standardised features.
- Trained on the TRAIN split of all four repositories (pooled), calibrated with Platt scaling on the TUNE split,
  evaluated per repository on the TEST split (later time period, realistic imbalance).
- Variants (for E4/E5): static-only, metrics-only (static + change, no graph), full (static + change + graph),
  full without calibration.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import CHANGE, GRAPH, STATIC

FEATURE_SETS = {
    "static": STATIC,
    "metrics": STATIC + CHANGE,          # "metrics-only" comparison model (Phase 1 Gap 3)
    "full": STATIC + CHANGE + GRAPH,     # proposed
}
SIGNED = {"mi"}  # not log-transformed


def _x(rows, feats) -> np.ndarray:
    return np.array([[r[f] if f in SIGNED else math.log1p(max(0.0, float(r[f]))) for f in feats] for r in rows])


@dataclass
class RiskModel:
    features: list[str]
    calibrate: bool = True
    C: float = 1.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    clf: LogisticRegression | None = None
    platt: LogisticRegression | None = None

    def fit(self, train_rows, tune_rows=None) -> "RiskModel":
        X = self.scaler.fit_transform(_x(train_rows, self.features))
        y = np.array([r["y"] for r in train_rows])
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        if self.calibrate and tune_rows:
            s = self._logit(tune_rows).reshape(-1, 1)
            yt = np.array([r["y"] for r in tune_rows])
            self.platt = LogisticRegression(C=1e6, max_iter=2000).fit(s, yt) if 0 < yt.sum() < len(yt) else None
        return self

    def _logit(self, rows) -> np.ndarray:
        return self.clf.decision_function(self.scaler.transform(_x(rows, self.features)))

    def predict(self, rows) -> np.ndarray:
        s = self._logit(rows)
        if self.platt is not None:
            return self.platt.predict_proba(s.reshape(-1, 1))[:, 1]
        return 1 / (1 + np.exp(-s))

    def contributions(self, row: dict, top: int = 4) -> list[tuple[str, float, float]]:
        """(feature, raw value, contribution to the logit) for the largest absolute contributions."""
        z = self.scaler.transform(_x([row], self.features))[0]
        c = self.clf.coef_[0] * z
        idx = np.argsort(-np.abs(c))[:top]
        return [(self.features[i], float(row[self.features[i]]), float(c[i])) for i in idx]

    def to_json(self) -> dict:
        return {"features": self.features, "calibrated": self.platt is not None,
                "coef": dict(zip(self.features, map(float, self.clf.coef_[0]))),
                "intercept": float(self.clf.intercept_[0]),
                "scaler_mean": list(map(float, self.scaler.mean_)), "scaler_scale": list(map(float, self.scaler.scale_)),
                "platt": None if self.platt is None else {"a": float(self.platt.coef_[0][0]),
                                                          "b": float(self.platt.intercept_[0])}}

    @classmethod
    def from_json(cls, d: dict) -> "RiskModel":
        m = cls(d["features"], calibrate=d["platt"] is not None)
        m.scaler.mean_ = np.array(d["scaler_mean"]); m.scaler.scale_ = np.array(d["scaler_scale"])
        m.scaler.n_features_in_ = len(d["features"]); m.scaler.var_ = m.scaler.scale_ ** 2
        m.clf = LogisticRegression()
        m.clf.coef_ = np.array([[d["coef"][f] for f in d["features"]]]); m.clf.intercept_ = np.array([d["intercept"]])
        m.clf.classes_ = np.array([0, 1])
        if d["platt"]:
            m.platt = LogisticRegression()
            m.platt.coef_ = np.array([[d["platt"]["a"]]]); m.platt.intercept_ = np.array([d["platt"]["b"]])
            m.platt.classes_ = np.array([0, 1])
        return m

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2))


def make_variant(name: str) -> RiskModel:
    if name == "full_uncalibrated":
        return RiskModel(FEATURE_SETS["full"], calibrate=False)
    return RiskModel(FEATURE_SETS[name], calibrate=True)
