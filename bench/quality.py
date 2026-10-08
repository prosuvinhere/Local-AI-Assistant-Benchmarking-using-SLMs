"""Rule-based, automatically verifiable quality checks.

Each test prompt declares a list of checks; the prompt's quality score is the
fraction of (non-skipped) checks that pass. Checks are deliberately objective
so results are reproducible and model-agnostic - no LLM-as-judge bias.

Supported check types
---------------------
contains_all   {"values": [...], "case_sensitive": false}
contains_any   {"values": [...]}
not_contains   {"values": [...]}
regex          {"pattern": "...", "flags": "i"}        (must match)
not_regex      {"pattern": "..."}                      (must NOT match)
number         {"value": 215, "tolerance": 0}          ("Answer: N" or last number)
max_words      {"value": 50}
min_words      {"value": 10}
line_count     {"value": 3, "bullets_only": false}     (exact, non-empty lines)
json_valid     {}                                      (passed schema validation)
json_field     {"field": "priority", "equals": "urgent"} | {"field": ..., "contains": "..."}
python_function{"name": "f", "tests": [{"args": [...], "expected": ...}]}
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


class CheckResult(BaseModel):
    type: str
    passed: Optional[bool]  # None = skipped (excluded from the score)
    detail: str = ""


class QualityResult(BaseModel):
    score: float
    checks: List[CheckResult]

    @property
    def failed(self) -> List[CheckResult]:
        return [c for c in self.checks if c.passed is False]


# ----------------------------------------------------------------- helpers
def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def extract_final_number(text: str) -> Optional[float]:
    explicit = re.findall(rf"answer\s*(?:is)?\s*[:=]?\s*\**\s*({_NUM})", text, re.I)
    candidates = explicit or re.findall(_NUM, text)
    if not candidates:
        return None
    try:
        return float(candidates[-1].replace(",", ""))
    except ValueError:
        return None


def extract_code(text: str, func_name: Optional[str] = None) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks:
        return text
    if func_name:
        for b in blocks:
            if re.search(rf"def\s+{re.escape(func_name)}\s*\(", b):
                return b
    return max(blocks, key=len)


_HARNESS = r"""
import json, sys
src = open(sys.argv[1]).read()
tests = json.loads(open(sys.argv[2]).read())
ns = {"__name__": "candidate"}
try:
    exec(compile(src, "<model>", "exec"), ns)
    fn = ns[sys.argv[3]]
except BaseException as e:
    print(json.dumps({"error": repr(e)[:200]})); sys.exit(0)
results = []
for t in tests:
    try:
        results.append(fn(*t["args"]) == t["expected"])
    except BaseException:
        results.append(False)
print(json.dumps({"results": results}))
"""


def run_python_tests(code: str, func_name: str, tests: List[Dict[str, Any]], timeout_s: float = 10.0) -> Dict[str, Any]:
    """Execute model-written code in an isolated subprocess with a timeout.

    This runs untrusted code: it is only invoked when the user opts in with
    --allow-code-exec. ``python -I`` ignores env vars and user site-packages.
    """
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, content in (("src.py", code), ("tests.json", json.dumps(tests)), ("h.py", _HARNESS)):
            paths[name] = f"{tmp}/{name}"
            with open(paths[name], "w") as fh:
                fh.write(content)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", paths["h.py"], paths["src.py"], paths["tests.json"], func_name],
                capture_output=True, text=True, timeout=timeout_s, cwd=tmp, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
        if not lines:
            return {"error": (proc.stderr or "no output")[-200:]}
        return json.loads(lines[-1])


# ------------------------------------------------------------------ checks
Ctx = Dict[str, Any]


def _cs(check: Dict[str, Any], s: str) -> str:
    return s if check.get("case_sensitive") else s.lower()


def _contains_all(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    missing = [v for v in c["values"] if _cs(c, v) not in _cs(c, text)]
    return CheckResult(type="contains_all", passed=not missing, detail=f"missing {missing}" if missing else "")


def _contains_any(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    ok = any(_cs(c, v) in _cs(c, text) for v in c["values"])
    return CheckResult(type="contains_any", passed=ok, detail="" if ok else f"none of {c['values']}")


def _not_contains(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    found = [v for v in c["values"] if _cs(c, v) in _cs(c, text)]
    return CheckResult(type="not_contains", passed=not found, detail=f"found {found}" if found else "")


def _flags(c: Dict[str, Any]) -> int:
    f = 0
    for ch in c.get("flags", ""):
        f |= {"i": re.I, "m": re.M, "s": re.S}[ch]
    return f


def _regex(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    ok = re.search(c["pattern"], text, _flags(c)) is not None
    return CheckResult(type="regex", passed=ok, detail="" if ok else f"no match for /{c['pattern']}/")


def _not_regex(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    m = re.search(c["pattern"], text, _flags(c))
    return CheckResult(type="not_regex", passed=m is None, detail=f"matched {m.group(0)!r}" if m else "")


def _number(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    got = extract_final_number(text)
    ok = got is not None and abs(got - float(c["value"])) <= float(c.get("tolerance", 0))
    return CheckResult(type="number", passed=ok, detail=f"expected {c['value']}, got {got}")


def _max_words(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    n = len(words(text))
    return CheckResult(type="max_words", passed=n <= c["value"], detail=f"{n} words (max {c['value']})")


def _min_words(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    n = len(words(text))
    return CheckResult(type="min_words", passed=n >= c["value"], detail=f"{n} words (min {c['value']})")


def _line_count(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if c.get("bullets_only"):
        lines = [l for l in lines if re.match(r"\s*([-*•]|\d+[.)])\s+", l)]
    return CheckResult(type="line_count", passed=len(lines) == c["value"], detail=f"{len(lines)} lines (want {c['value']})")


def _json_valid(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    ok = bool(ctx.get("structured_ok"))
    return CheckResult(type="json_valid", passed=ok, detail="" if ok else str(ctx.get("structured_error", ""))[:200])


def _json_field(text: str, c: Dict[str, Any], ctx: Ctx) -> CheckResult:
    data = ctx.get("parsed") or {}
    field = c["field"]
    if field not in data:
        return CheckResult(type="json_field", passed=False, detail=f"{field} missing")
    val = data[field]
    if "equals" in c:
        ok = (val.lower() == str(c["equals"]).lower()) if isinstance(val, str) else val == c["equals"]
        return CheckResult(type="json_field", passed=ok, detail=f"{field}={val!r} (want {c['equals']!r})")
    ok = str(c["contains"]).lower() in str(val).lower()
    return CheckResult(type="json_field", passed=ok, detail=f"{field}={val!r} (want ~{c['contains']!r})")


def _python_function(text: str, c: Dict[str, Any], ctx: Ctx) -> List[CheckResult]:
    name = c["name"]
    code = extract_code(text, name)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [CheckResult(type="python_defines", passed=False, detail=f"SyntaxError: {exc.msg}"),
                CheckResult(type="python_tests", passed=False, detail="code does not parse")]
    defined = any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))
    out = [CheckResult(type="python_defines", passed=defined, detail="" if defined else f"no def {name}()")]
    if not ctx.get("allow_code_exec"):
        out.append(CheckResult(type="python_tests", passed=None, detail="skipped (enable --allow-code-exec)"))
    elif not defined:
        out.append(CheckResult(type="python_tests", passed=False, detail="function not defined"))
    else:
        res = run_python_tests(code, name, c["tests"])
        if "error" in res:
            out.append(CheckResult(type="python_tests", passed=False, detail=res["error"]))
        else:
            passed = sum(res["results"])
            out.append(CheckResult(type="python_tests", passed=passed == len(res["results"]),
                                   detail=f"{passed}/{len(res['results'])} tests passed"))
    return out


CHECKS: Dict[str, Callable[..., Any]] = {
    "contains_all": _contains_all,
    "contains_any": _contains_any,
    "not_contains": _not_contains,
    "regex": _regex,
    "not_regex": _not_regex,
    "number": _number,
    "max_words": _max_words,
    "min_words": _min_words,
    "line_count": _line_count,
    "json_valid": _json_valid,
    "json_field": _json_field,
    "python_function": _python_function,
}


def score_response(
    text: str,
    checks: List[Dict[str, Any]],
    *,
    parsed: Optional[Dict[str, Any]] = None,
    structured_ok: Optional[bool] = None,
    structured_error: Optional[str] = None,
    allow_code_exec: bool = False,
) -> QualityResult:
    ctx: Ctx = {"parsed": parsed, "structured_ok": structured_ok,
                "structured_error": structured_error, "allow_code_exec": allow_code_exec}
    results: List[CheckResult] = []
    for check in checks:
        fn = CHECKS.get(check["type"])
        if fn is None:
            raise ValueError(f"Unknown check type: {check['type']}")
        r = fn(text, check, ctx)
        results.extend(r if isinstance(r, list) else [r])
    scored = [r for r in results if r.passed is not None]
    score = sum(1 for r in scored if r.passed) / len(scored) if scored else 0.0
    return QualityResult(score=score, checks=results)
