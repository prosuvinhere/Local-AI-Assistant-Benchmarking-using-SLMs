"""Offline tests for metrics math and quality scorers."""
import math

from app.config import normalize_host
from app.metrics import GenerationMetrics, percentile, summarize
from bench.quality import extract_final_number, score_response


def test_percentile_matches_numpy_linear():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(data, 50) == 5.5
    assert math.isclose(percentile(data, 95), 9.55)


def test_summary_ci_uses_student_t():
    s = summarize([10, 12, 14])
    assert s.n == 3 and s.mean == 12 and s.stdev == 2
    assert math.isclose(s.ci95, 4.303 * 2 / math.sqrt(3), rel_tol=1e-6)
    assert summarize([None, None]) is None


def test_generation_metrics_from_ollama():
    final = {"prompt_eval_count": 100, "prompt_eval_duration": 200_000_000,
             "eval_count": 50, "eval_duration": 1_000_000_000, "load_duration": 5_000_000,
             "total_duration": 1_300_000_000, "done_reason": "stop"}
    ms = 1_000_000
    m = GenerationMetrics.from_ollama("m", final, t_start_ns=0, t_first_ns=250 * ms,
                                      t_end_ns=1_400 * ms, arrivals_ns=[250 * ms, 270 * ms, 290 * ms])
    assert m.ttft_ms == 250 and m.total_latency_ms == 1400
    assert m.decode_tps == 50 and m.prompt_tps == 500
    assert m.mean_itl_ms == 20 and m.server_ttft_ms == 205
    assert math.isclose(m.client_overhead_ms, 100)


def test_normalize_host():
    assert normalize_host("0.0.0.0") == "http://127.0.0.1:11434"
    assert normalize_host("localhost:1234") == "http://localhost:1234"
    assert normalize_host("https://box:443/") == "https://box:443"


def test_number_extraction():
    assert extract_final_number("so 3h35m = 215 minutes.\nAnswer: 215") == 215
    assert extract_final_number("The answer is **1**.") == 1
    assert extract_final_number("nothing") is None


def test_text_checks():
    q = score_response("- a housing plan\n- parking removed\n- grant", [
        {"type": "line_count", "value": 3, "bullets_only": True},
        {"type": "contains_any", "values": ["housing"]},
        {"type": "max_words", "value": 3},
    ])
    assert math.isclose(q.score, 2 / 3)
    assert q.failed[0].type == "max_words"


def test_python_function_static_and_exec():
    answer = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
    check = [{"type": "python_function", "name": "is_even",
              "tests": [{"args": [2], "expected": True}, {"args": [3], "expected": False}]}]
    static = score_response(answer, check)
    assert static.score == 1.0 and static.checks[1].passed is None  # tests skipped
    executed = score_response(answer, check, allow_code_exec=True)
    assert executed.score == 1.0 and "2/2" in executed.checks[1].detail
    broken = score_response("```python\ndef is_even(n) return\n```", check, allow_code_exec=True)
    assert broken.score == 0.0


def test_json_checks():
    q = score_response("", [
        {"type": "json_valid"},
        {"type": "json_field", "field": "priority", "equals": "urgent"},
        {"type": "json_field", "field": "customer_name", "contains": "priya"},
    ], parsed={"priority": "Urgent", "customer_name": "Priya Raman"}, structured_ok=True)
    assert q.score == 1.0
