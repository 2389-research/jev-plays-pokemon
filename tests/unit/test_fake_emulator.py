from pokemon_agent.emulator.fake_emulator import FakeEmulator, ADDR_PLAYER_X, ADDR_PLAYER_Y
from pokemon_agent.emulator.interface import GameButton


def test_walk_into_open_floor_moves():
    emu = FakeEmulator(start=(2, 1))  # row 1 is "#......#", open to the right
    emu.hold(GameButton.RIGHT)
    emu.tick(1)
    emu.release(GameButton.RIGHT)
    assert emu.read_memory(ADDR_PLAYER_X) == 3
    assert emu.read_memory(ADDR_PLAYER_Y) == 1


def test_walk_into_wall_blocked():
    emu = FakeEmulator(start=(1, 1))  # left of (1,1) is wall column 0
    emu.hold(GameButton.LEFT)
    emu.tick(3)
    emu.release(GameButton.LEFT)
    assert emu.read_memory(ADDR_PLAYER_X) == 1  # unchanged


def test_screenshot_dimensions_and_bytes():
    emu = FakeEmulator()
    shot = emu.screenshot()
    assert shot.mime_type == "image/png"
    assert shot.width > 0 and shot.height > 0
    assert len(shot.data) > 0


def test_write_memory_round_trips():
    emu = FakeEmulator()
    emu.write_memory(0xD16C, 42)
    assert emu.read_memory(0xD16C) == 42


def test_save_and_load_state_roundtrip(tmp_path):
    emu = FakeEmulator(start=(2, 1))
    p = tmp_path / "s.state"
    emu.save_state(p)
    emu.hold(GameButton.RIGHT)
    emu.tick(1)
    emu.release(GameButton.RIGHT)
    assert emu.read_memory(ADDR_PLAYER_X) == 3
    emu.load_state(p)
    assert emu.read_memory(ADDR_PLAYER_X) == 2
