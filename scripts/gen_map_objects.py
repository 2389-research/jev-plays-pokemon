#!/usr/bin/env python3
"""Rip the interactable background objects of every map from the pokered disassembly.

Some things you press A on aren't sprites: signs (``bg_event``) and "hidden events" (Pokémon Center
PCs, Bill's Cell Separator PC, Lt. Surge's trash cans, gym statues, slot machines ...). RAM has no
sprite for them, so the agent can't see them — runs/vermilion-team: L1 correctly said "use the PC in
Bill's House" but the harness could only aim at people, so it re-talked to Bill forever.

Output ``src/pokemon_agent/games/pokemon_red/map_objects.json``:
    {map_id: [{"name": str, "x": int, "y": int, "kind": "sign"|"object", "face": "north"|... |null}]}
``face`` = the direction the player must face (the routine checks it), null = any adjacent side.
Hidden ITEMS / coins are deliberately left out (finding them is the player's job).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rip_portals import POKERED, REPO, build_name_index, parse_map_constants  # noqa: E402

OUT = REPO / "src" / "pokemon_agent" / "games" / "pokemon_red" / "map_objects.json"

SKIP = {"HiddenItems", "HiddenCoins", "CableClubLeftGameboy", "CableClubRightGameboy"}
NAMES = {
    "OpenPokemonCenterPC": "PC", "OpenRedsPC": "PC",
    "BillsHousePC": "Bill's PC (runs the Cell Separator teleporter machine)",
    "GymTrashScript": "trash can", "PrintTrashText": "trash can",
    "GymStatues": "gym statue", "StartSlotMachine": "slot machine",
    "PrintBenchGuyText": "bench", "PrintBookcaseText": "bookcase",
    "PrintMagazinesText": "magazines", "PrintNewBikeText": "bicycle",
    "PrintCinnabarQuiz": "quiz machine", "Mansion": "statue switch",
    "PrintFightingDojoText": "dojo scroll", "DisplayOakLabEmailText": "PC (email)",
    "DisplayOakLabLeftPoster": "poster", "DisplayOakLabRightPoster": "poster",
    "KabutopsFossil": "fossil display", "AerodactylFossil": "fossil display",
    "PrintRedSNESText": "SNES", "PrintIndigoPlateauHQText": "sign",
}
FACING = {"SPRITE_FACING_UP": "north", "SPRITE_FACING_DOWN": "south",
          "SPRITE_FACING_LEFT": "west", "SPRITE_FACING_RIGHT": "east"}


def _humanize(ident: str) -> str:
    """PrintNotebookText / ViridianSchoolNotebook -> 'notebook'-ish readable words."""
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", ident.replace("Print", "").replace("Text", "")).lower()
    return words.strip() or ident


def parse_hidden(consts: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    cur = None
    for line in (POKERED / "data" / "events" / "hidden_events.asm").read_text().splitlines():
        m = re.match(r"\s*hidden_events_for\s+(\w+)", line)
        if m:
            cur = consts.get(m.group(1), (None,))[0]
            continue
        m = re.match(r"\s*hidden_(event|text_predef)\s+(\d+),\s*(\d+),\s*(\w+),\s*(\w+)", line)
        if not m or cur is None:
            continue
        kind, x, y, fn, arg = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        if fn in SKIP:
            continue
        name = NAMES.get(fn) or _humanize(arg if kind == "text_predef" else fn)
        out.setdefault(cur, []).append({"name": name, "x": x, "y": y, "kind": "object",
                                        "face": FACING.get(arg) if fn == "BillsHousePC" else None})
    return out


def parse_signs(consts: dict, name_to_const: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for f in sorted((POKERED / "data" / "maps" / "objects").glob("*.asm")):
        mid = consts.get(name_to_const.get(f.stem, ""), (None,))[0]
        if mid is None:
            continue
        for m in re.finditer(r"bg_event\s+(\d+),\s*(\d+),\s*TEXT_\w+?_(\w+)", f.read_text()):
            label = m.group(3).lower().replace("_", " ")
            out.setdefault(mid, []).append({"name": f"sign ({label})", "x": int(m.group(1)),
                                            "y": int(m.group(2)), "kind": "sign", "face": None})
    return out


def main() -> None:
    consts = parse_map_constants()
    _c2n, name_to_const = build_name_index()
    objs = parse_signs(consts, name_to_const)
    for mid, lst in parse_hidden(consts).items():
        objs.setdefault(mid, []).extend(lst)
    OUT.write_text(json.dumps({str(k): v for k, v in sorted(objs.items())}, indent=0))
    print(f"wrote {OUT.relative_to(REPO)}: {sum(map(len, objs.values()))} objects on {len(objs)} maps")
    print("Bill's House (88):", objs.get(88))


if __name__ == "__main__":
    main()
