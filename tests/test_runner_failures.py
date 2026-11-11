import pytest

from seintel.eval import runner
from seintel.llm.base import LLM, LLMResponse


class AlwaysFails(LLM):
    name = "broken"

    def _complete(self, system, messages, max_tokens):
        return LLMResponse("LLM_ERROR: bad request: Error code: 400", stop_reason="error")


def test_api_failures_stop_the_run_and_write_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_llm", lambda *a, **k: AlwaysFails())
    with pytest.raises(runner.LLMUnavailable, match="400"):
        runner.run_system("B2", split="dev", runs=1, repos=["requests"], out_root=tmp_path, judge=False,
                          verbose=False)
    out = tmp_path / "results/qa/dev/B2/run0.jsonl"
    assert not out.exists() or out.read_text() == ""          # failed questions stay retryable


def test_failed_rows_are_retried_on_resume(tmp_path):
    f = tmp_path / "run0.jsonl"
    f.write_text('{"qid": "a", "llm_errors": 3}\n{"qid": "b", "llm_errors": 0}\n')
    assert runner._done(f) == {"b"}
