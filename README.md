# pokemon-agent

An agent that plays Pokémon Red through the [PyBoy](https://github.com/Baekalfen/PyBoy) Game Boy emulator.

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

## Secrets

Copy your API keys into a `.env` file at the project root (e.g. `OPENAI_API_KEY=...`).

## Ignored files

`ROMs (*.gb, *.gbc, *.gba)`, save states (`*.state`), and `.env` are listed in
`.gitignore` and will never be committed. The `roms/` directory is tracked only
via `roms/.gitkeep`.
