# pokemon-agent

An agent that plays Pokémon Red through the [PyBoy](https://github.com/Baekalfen/PyBoy) Game Boy emulator.

> **📖 Full setup, run commands, parameters, the Orrery knowledge base, the viewer, tuning constants,
> and current status: [`docs/RUNNING.md`](docs/RUNNING.md).** Read that first — it's the operational
> source of truth (the quick-start below is just the basics).

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a `.venv/` with all dependencies (including PyBoy, whose pip wheels
bundle SDL2 — no separate SDL2 install is needed).

## Running

Run any Python entrypoint inside the project environment with `uv run`:

```bash
uv run python -m pokemon_agent            # once application code exists
# or, e.g.
uv run python path/to/script.py --rom roms/pokemon-red.gb
```

## ROM (you must supply your own)

This project does **not** include a game ROM, and you must **not** download one
from us or anywhere else that distributes it — that is illegal.

You must provide your **own legally-owned** copy of the Pokémon Red ROM:

- Place it at `roms/pokemon-red.gb`, **or**
- Pass its path explicitly with `--rom <path>`.

## Before running the agent: play the intro yourself

The agent boots from a **save state**, not from a cold cartridge — it does not
handle the opening cutscene, the name-entry keyboards, or Oak's introduction. So
before you hand control to the bot, play the beginning yourself and make the
choices you want to live with for the whole run:

- **Name your character** (and your rival) — the agent inherits whatever you pick.
- Sit through Oak's intro and get to a point you want the bot to take over from
  (e.g. standing in the lab, or already holding your starter).
- Save a state there. The default natural save is `roms/pokemon_red.gb.state`
  (auto-written by `scripts/play.py` when you close its window); the agent starts
  from it via `run_agent.py --load-state`, or from any fixture in `states/`.

If you skip this, the agent will be dropped into the naming keyboard / cutscene
with no idea what to do. Name your character first.

## Secrets

Copy your API keys into a `.env` file at the project root (e.g. `OPENAI_API_KEY=...`).

## Ignored files

`ROMs (*.gb, *.gbc, *.gba)`, save states (`*.state`), and `.env` are listed in
`.gitignore` and will never be committed. The `roms/` directory is tracked only
via `roms/.gitkeep`.
