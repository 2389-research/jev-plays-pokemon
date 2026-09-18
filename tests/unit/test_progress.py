"""Progress / milestone detector: pure unit tests + live fixture checks."""
from pathlib import Path

import pytest

from pokemon_agent.games.pokemon_red import progress
from pokemon_agent.games.pokemon_red.progress import (
    WCURMAP,
    progress_vector,
    read_progress,
)

ROM = Path("roms/pokemon_red.gb")

# RAM addresses used to hand-build fake states.
WPARTYCOUNT = 0xD163
WPARTYMON0 = 0xD16B
PARTY_STRUCT = 0x2C
WBADGES = 0xD356
WMONEY = 0xD347


class MemFake:
    """Memory-backed fake emulator (see tests/unit/test_battle.py MemFake)."""

    def __init__(self, mem=None):
        self.mem = dict(mem or {})

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def _mon_level_mem(index: int, species: int, level: int) -> dict:
    base = WPARTYMON0 + index * PARTY_STRUCT
    return {base + 0x00: species, base + 0x21: level}


# --- pure unit tests ------------------------------------------------------

def test_badge_popcount():
    # 0b0000_0101 = two badges set.
    assert read_progress(MemFake({WBADGES: 0b0000_0101})).badges == 2
    assert read_progress(MemFake({WBADGES: 0xFF})).badges == 8
    assert read_progress(MemFake({})).badges == 0


def test_money_bcd_three_bytes():
    # Big-endian 3-byte BCD: 00 12 34 -> 1234.
    mem = {WMONEY: 0x00, WMONEY + 1: 0x12, WMONEY + 2: 0x34}
    assert read_progress(MemFake(mem)).money == 1234
    # 12 34 56 -> 123456
    mem = {WMONEY: 0x12, WMONEY + 1: 0x34, WMONEY + 2: 0x56}
    assert read_progress(MemFake(mem)).money == 123456


def test_milestone_got_starter():
    assert read_progress(MemFake({WPARTYCOUNT: 0})).milestones["got_starter"] is False
    assert read_progress(MemFake({WPARTYCOUNT: 1})).milestones["got_starter"] is True
    assert read_progress(MemFake({WPARTYCOUNT: 3})).milestones["got_starter"] is True


def test_milestone_beat_brock():
    assert read_progress(MemFake({WBADGES: 0})).milestones["beat_brock"] is False
    assert read_progress(MemFake({WBADGES: 0x01})).milestones["beat_brock"] is True


def test_milestone_map_based_booleans():
    # Viridian Forest == 51, Pewter City == 2 (resolved from MAP_NAMES_RAW).
    vf = read_progress(MemFake({WCURMAP: 51})).milestones
    assert vf["entered_viridian_forest"] is True
    assert vf["reached_pewter"] is False

    pw = read_progress(MemFake({WCURMAP: 2})).milestones
    assert pw["reached_pewter"] is True
    assert pw["entered_viridian_forest"] is False

    pallet = read_progress(MemFake({WCURMAP: 0})).milestones
    assert pallet["entered_viridian_forest"] is False
    assert pallet["reached_pewter"] is False


def test_all_milestone_map_names_resolved():
    # Guardrail: both map-based milestones resolve to a real id.
    assert progress.UNRESOLVED_MILESTONES == {}


def test_milestone_keys_exact():
    ms = read_progress(MemFake({})).milestones
    assert set(ms) == {
        "got_starter",
        "entered_viridian_forest",
        "reached_pewter",
        "beat_brock",
    }


def test_progress_vector_flat_scalars():
    mem = {
        WBADGES: 0x01,
        WCURMAP: 2,
        WPARTYCOUNT: 2,
        WMONEY: 0x00, WMONEY + 1: 0x00, WMONEY + 2: 0x99,
    }
    mem.update(_mon_level_mem(0, species=1, level=7))
    mem.update(_mon_level_mem(1, species=2, level=12))
    vec = progress_vector(MemFake(mem))
    assert vec == {
        "badges": 1,
        "map_id": 2,
        "party_size": 2,
        "max_party_level": 12,
        "money": 99,
    }


def test_progress_vector_empty_party_max_level_zero():
    assert progress_vector(MemFake({WPARTYCOUNT: 0}))["max_party_level"] == 0


def test_party_levels_from_read_party():
    mem = {WPARTYCOUNT: 2}
    mem.update(_mon_level_mem(0, species=1, level=5))
    mem.update(_mon_level_mem(1, species=4, level=9))
    assert read_progress(MemFake(mem)).party_levels == [5, 9]


# --- live fixture tests ---------------------------------------------------

def _load(fixture: str):
    from pokemon_agent.emulator.pyboy_adapter import PyBoyEmulator

    emu = PyBoyEmulator(str(ROM), window="null", speed=0)
    emu.load_state(Path(fixture))
    emu.tick(3)
    return emu


_live = pytest.mark.skipif(
    not (ROM.exists() and Path("states/after_starter.state").exists()),
    reason="needs the ROM and real save-state fixtures",
)


@_live
def test_live_after_starter():
    emu = _load("states/after_starter.state")
    try:
        p = read_progress(emu)
        assert p.party_size == 0
        # ground truth once the loaded state SETTLES (~60 frames): this fixture is inside
        # Oak's Lab, not Pallet Town. The old assertion (map 0) read stale pre-settle RAM.
        assert p.map_id == 40
        assert p.map_name == "Oaks Lab"
        assert p.badges == 0
        assert p.milestones["got_starter"] is False
    finally:
        emu.close()


@_live
def test_live_lab():
    emu = _load("states/lab.state")
    try:
        p = read_progress(emu)
        assert p.map_id == 40  # Oaks Lab
        assert p.map_name == "Oaks Lab"
    finally:
        emu.close()


@_live
def test_live_battle():
    emu = _load("states/battle.state")
    try:
        p = read_progress(emu)
        assert p.party_size == 1  # Squirtle L5
        assert p.milestones["got_starter"] is True
    finally:
        emu.close()
