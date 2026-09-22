from pokemon_agent.logging.run_recorder import RunRecorder, unique_run_dir


def test_on_event_type_marker_not_clobbered_by_payload_kind(tmp_path):
    """Regression: on_event stamps the event TYPE under "kind" ("step_done"). A step_done payload
    must NOT carry its own "kind" key or it would clobber that marker in the log (it carries the
    step's kind under "step_kind" instead)."""
    rec = RunRecorder(emu=None, run_dir=tmp_path / "r")
    rec.on_event("step_done", {"step": 3, "id": "q1", "done_when": "no_item:oaks_parcel",
                               "step_kind": "action"})
    ev = rec._events[-1]
    assert ev["kind"] == "step_done"          # the event-type marker survives
    assert ev["step_kind"] == "action"        # the step's own kind is preserved, distinctly
    assert ev["done_when"] == "no_item:oaks_parcel"


def test_unique_run_dir_uses_name_when_free(tmp_path):
    base = tmp_path / "myrun"
    assert unique_run_dir(base) == base            # free -> keep the nice name


def test_unique_run_dir_never_clobbers_existing(tmp_path):
    base = tmp_path / "myrun"
    base.mkdir()                                    # simulate a prior run at this name
    u = unique_run_dir(base)
    assert u != base and u.parent == tmp_path and u.name.startswith("myrun-")
    assert not u.exists()                           # a fresh, unused path
