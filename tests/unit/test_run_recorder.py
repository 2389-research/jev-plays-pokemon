from pokemon_agent.logging.run_recorder import unique_run_dir


def test_unique_run_dir_uses_name_when_free(tmp_path):
    base = tmp_path / "myrun"
    assert unique_run_dir(base) == base            # free -> keep the nice name


def test_unique_run_dir_never_clobbers_existing(tmp_path):
    base = tmp_path / "myrun"
    base.mkdir()                                    # simulate a prior run at this name
    u = unique_run_dir(base)
    assert u != base and u.parent == tmp_path and u.name.startswith("myrun-")
    assert not u.exists()                           # a fresh, unused path
