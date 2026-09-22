"""Flow-gate probe (Deliverable 0 of the Jev flow-router spec).

Runs Jev's calibrated `choose_flow` (N parallel yes/no questions in one call) against LABELLED
frames and scores it — BEFORE wiring the router into the loop. Tune the prompt/features here until
the all-lowercase dialogue classifies as `dialogue` and the cutscene blob as `navigate`.

Live (spends Jev credits): fish -c 'uv run python scripts/probe_flow_gate.py'
The `route`/grade logic is pure and offline-unit-tested in tests/unit/test_flow_gate_grader.py.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")   # so tests.fixtures (the real mined frames) imports when run from repo root

FLOW_MIN_CONF = 0.55

# Labelled frames: REAL game-produced strings mined from recorded runs (tests/fixtures/flow_frames.py),
# not invented. The decisive input is `screen_text` (decoded text-box region).
from tests.fixtures.flow_frames import FRAMES as CASES


def route(answers: dict, ram_menu_open: bool, min_conf: float = FLOW_MIN_CONF) -> str:
    """Pure router: Jev's parallel {dialogue:(a,c), menu:(a,c)} answers + the RAM menu signal -> flow.
    Menu wins if RAM says open OR Jev is confident (cross-check; A on a menu SELECTS). Then a confident
    dialogue -> dialogue. Otherwise navigate. (Battle is handled deterministically upstream.)"""
    d_ans, d_conf = answers.get("dialogue", ("no", 0.0))
    m_ans, m_conf = answers.get("menu", ("no", 0.0))
    if ram_menu_open or (m_ans == "yes" and m_conf >= min_conf):
        return "menu"
    if d_ans == "yes" and d_conf >= min_conf:
        return "dialogue"
    return "navigate"


def run_eval():
    from pokemon_agent.agent.typesafe_reasoner import TypeSafeReasoner
    r = TypeSafeReasoner()   # account-default model
    passed = 0
    print(f"{'case':24} {'expect':9} {'got':9} {'dialogue':16} {'menu':16}")
    for c in CASES:
        try:
            ans = r.choose_flow(screen_text=c["screen_text"], has_text=c["has_text"],
                                text_box_id=1 if c["has_text"] else 0, last_action="move")
        except Exception as e:
            print(f"{c['name']:24} CALL FAILED: {type(e).__name__}: {e}")
            continue
        got = route(ans, c["ram_menu_open"])
        ok = got == c["expect"]
        passed += ok
        d = f"{ans['dialogue'][0]}({ans['dialogue'][1]:.2f})"
        m = f"{ans['menu'][0]}({ans['menu'][1]:.2f})"
        print(f"{c['name']:24} {c['expect']:9} {('OK ' if ok else 'XX ')+got:9} {d:16} {m:16}")
    print(f"\nSCORE: {passed}/{len(CASES)}")


if __name__ == "__main__":
    run_eval()
