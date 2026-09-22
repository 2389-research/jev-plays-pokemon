from pokemon_agent.logging.capture import Capture


def test_loop_mounts_capture_on_planner_and_reasoner(monkeypatch, tmp_path):
    # Build a ReasoningLoop with a stub recorder + capture_mode="distill"; assert the loop
    # exposes self.capture and attaches the SAME object to planner/reasoner.
    from pokemon_agent.agent import reason_loop as RL
    loop = RL.ReasoningLoop.__new__(RL.ReasoningLoop)   # bypass heavy __init__
    cap = Capture(tmp_path, mode="distill")
    loop.capture = cap
    class _P:  # noqa
        pass
    loop.planner = _P(); loop.reasoner = _P()
    RL.ReasoningLoop._mount_capture(loop)
    assert loop.planner.capture is cap and loop.reasoner.capture is cap
