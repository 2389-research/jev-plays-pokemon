# /// script
# requires-python = ">=3.12"
# ///
"""Manually play the ROM in a window to create a save state for the agent.

  uv run python scripts/play.py --rom roms/pokemon_red.gb

Keyboard (PyBoy defaults):
  Arrow keys .... D-pad
  a ............. A button
  s ............. B button
  Return ........ Start
  Backspace ..... Select
  space (hold) .. fast-forward
  z ............. SAVE state  ->  writes <rom>.state  (e.g. roms/pokemon_red.gb.state)
  x ............. LOAD that state

Play until you're standing in the overworld (past the intro + naming), press `z`
to save, then close the window. Point the agent at it with:

  uv run python scripts/run_agent.py --rom roms/pokemon_red.gb \
      --load-state roms/pokemon_red.gb.state --provider lunaroute \
      --model deepseek-4.1-flash --goal "Explore and leave this area"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_red.gb")
    ap.add_argument("--out", default=None, help="where to save the state (default <rom>.state)")
    ap.add_argument("--state", default=None, help="save as a named fixture states/<NAME>.state")
    ap.add_argument("--from", dest="from_state", default=None,
                    help="resume from states/<NAME>.state instead of booting the ROM")
    ap.add_argument("--speed", type=int, default=1)
    args = ap.parse_args()
    if args.state:
        states_dir = Path(__file__).resolve().parents[1] / "states"
        states_dir.mkdir(exist_ok=True)
        args.out = str(states_dir / f"{args.state}.state")

    if not Path(args.rom).exists():
        raise SystemExit(f"ROM not found: {args.rom}")

    from pyboy import PyBoy

    out = Path(args.out or f"{args.rom}.state")
    pb = PyBoy(args.rom, window="SDL2", sound=True)
    pb.set_emulation_speed(args.speed)
    if args.from_state:
        src = Path(__file__).resolve().parents[1] / "states" / f"{args.from_state}.state"
        with open(src, "rb") as f:
            pb.load_state(f)
        print(f">>> resumed from {src}")
    print(__doc__)
    print(f">>> window open. Play to the overworld, then CLOSE THE WINDOW (or press Ctrl+C)")
    print(f">>> the state is saved automatically to: {out}\n")
    try:
        while pb.tick(1):  # returns False when the window is closed
            pass
    except KeyboardInterrupt:
        pass

    # Auto-save whatever state we're in before shutting down (no hotkey needed).
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pb.save_state(f)
        print(f"\n✓ saved state -> {out}")
        print(f"  start the agent:  uv run python scripts/run_agent.py --rom {args.rom} \\")
        print(f"      --load-state {out} --provider lunaroute --model deepseek-4.1-flash \\")
        print(f"      --goal 'Explore and leave this area'")
    except Exception as e:
        print(f"\n! failed to save state: {e}")
    finally:
        pb.stop(save=False)


if __name__ == "__main__":
    main()
