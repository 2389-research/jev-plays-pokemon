# /// script
# requires-python = ">=3.12"
# ///
"""Layer 2.5: criteria-quality eval. Does the REAL strategist (glm-5.3 via LunaRoute) emit
valid, semantically-correct acceptance criteria (`done_when`) when `Planner.l1_decide` turns a
brainstormed assessment into concrete quest steps?

This is a SCORECARD, not a unit test — grading is fully deterministic (no LLM judge): every
emitted step must pass the same `validate_step` hard-gate the real L1 pipeline uses, and at
least one emitted step must match the fixture case's expected criterion (exact string or
predicate family, see `tests/fixtures/criteria_cases.py`).

IMPORTANT: importing this module makes NO network call. All LLM/network access is inside
`run_eval()` / `main()`, which only run when this script is executed directly:
  uv run python scripts/eval_criteria.py
CI exercises only `grade_case` (pure) via tests/unit/test_criteria_eval.py, on canned steps —
never the live path here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from pokemon_agent.agent.l1_pipeline import validate_step
from pokemon_agent.agent.planner_llm import Planner


def _step_family(step: dict) -> str | None:
    """The predicate FAMILY of a step's `done_when` — the single key `_parse_done_when`
    returns (e.g. "no_item", "hp_frac", "level", "on_map") — or None if it doesn't parse."""
    try:
        map_id = int(step.get("map", 0) or 0)
    except (TypeError, ValueError):
        map_id = 0
    parsed = Planner._parse_done_when(step.get("done_when"), map_id)
    if parsed is None:
        return None
    return next(iter(parsed.keys()))


def grade_case(case: dict, add_steps: list[dict]) -> dict:
    """Deterministically grade the `add` steps a decide call produced for one fixture CASE.

    - hard: EVERY emitted step passes `validate_step` (the same hard-gate `run_l1_pipeline`
      applies before a step ever reaches the plan). Vacuously True for an empty `add_steps`.
    - semantic: at least one step matches what the case expects:
        * `expect_exact` — a step whose `done_when` (stripped, case-insensitive) equals it.
        * `expect_family` — a step whose `_step_family` equals it; a `kind == "travel"` step
          always counts toward family "on_map" (travel's done_when is on_map by construction,
          even if the step under test omits/garbles it).

    Pure function: no network, no LLM. Safe to call from CI.
    """
    hard = all(validate_step(s)[0] for s in add_steps)

    want_exact = case.get("expect_exact")
    want_family = case.get("expect_family")
    if want_exact:
        norm = want_exact.strip().lower()
        semantic = any(str(s.get("done_when") or "").strip().lower() == norm for s in add_steps)
    else:
        def _matches(s: dict) -> bool:
            if want_family == "on_map" and str(s.get("kind")) == "travel":
                return True
            return _step_family(s) == want_family
        semantic = any(_matches(s) for s in add_steps)

    shape = [(s.get("kind"), s.get("map"), s.get("done_when")) for s in add_steps]
    return {
        "hard": hard,
        "semantic": semantic,
        "detail": f"expect={want_exact or want_family!r} steps={shape}",
    }


def _build_planner():
    """LIVE: construct a real Planner (LunaRoute glm-5.3 strategist, KnowledgeBase if
    reachable). Only called from run_eval()/main() — never at import time."""
    from pokemon_agent.providers.lunaroute import LunaRouteProvider

    strategist = LunaRouteProvider(model="glm-5.3", max_tokens=700)
    knowledge = None
    try:
        from pokemon_agent.agent.knowledge import KnowledgeBase
        knowledge = KnowledgeBase(base_url="http://localhost:8100", workspace_id="6d677a16")
    except Exception:
        knowledge = None
    return Planner(strategist=strategist, knowledge=knowledge)


def run_eval(planner=None) -> list[dict]:
    """LIVE runner: for each fixture CASE, call `l1_decide` (seeding a brief brainstorm
    assessment from the case's own milestone — a KB-grounded brainstorm is a nice-to-have but
    out of scope here) and grade the result with `grade_case`.

    NETWORK CALLS. Only invoked from `main()` / when this script runs as __main__."""
    from tests.fixtures.criteria_cases import CASES  # local import: keep module import network-free

    if planner is None:
        planner = _build_planner()

    results = []
    for case in CASES:
        brainstorm = {"assessment": case["context"].get("milestone", "")}
        try:
            decision = planner.l1_decide(case["context"], brainstorm)
        except Exception as e:  # pragma: no cover - live-only path
            results.append({"name": case["name"], "hard": False, "semantic": False,
                             "detail": f"l1_decide raised {type(e).__name__}: {e}", "steps": []})
            continue
        add_steps = [s for s in (decision.get("add") or []) if isinstance(s, dict)]
        grade = grade_case(case, add_steps)
        results.append({"name": case["name"], "steps": add_steps, **grade})
    return results


def main() -> None:
    results = run_eval()
    passes = 0
    for r in results:
        ok = r["hard"] and r["semantic"]
        passes += int(ok)
        print(f"[{r['name']}] {'PASS' if ok else 'FAIL'}  hard={r['hard']} semantic={r['semantic']}  {r['detail']}")
        for s in r["steps"]:
            print(f"        kind={s.get('kind')!r:8} map={s.get('map')!r:4} done_when={s.get('done_when')!r}")
    print(f"\nSCORE: {passes}/{len(results)}")


if __name__ == "__main__":
    main()
