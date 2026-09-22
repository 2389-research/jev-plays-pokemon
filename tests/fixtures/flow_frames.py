"""Flow-gate fixtures — REAL frames mined from recorded runs (runs/*/log.jsonl), not invented.

Each `screen_text` is an actual game-produced string the emulator decoded during a real run; `expect`
is the true control flow. Battle frames are excluded (the gate only runs outside battle, which is
short-circuited deterministically). `ram_menu_open` mirrors the RAM menu signal the router cross-checks.

Harvested with, e.g.:
    grep-and-count context.screen_text over runs/*/log.jsonl where context.in_battle is false,
    grouped by the logged context.kind (dialog / menu / overworld).
"""

FRAMES = [
    # overworld — no text box (5592 real frames)
    {"name": "overworld_empty", "screen_text": "", "has_text": False,
     "ram_menu_open": False, "expect": "navigate"},

    # dialogue — real NPC / event / sign lines the game showed
    {"name": "dialog_private_property", "screen_text": "This is private property!", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},
    {"name": "dialog_go_through", "screen_text": "You can go through here!", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},
    {"name": "dialog_got_parcel", "screen_text": "HARPER got OAK PARCEL!", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},
    {"name": "dialog_nurse_need_pokemon", "screen_text": "OK. Wel need your POKMON.", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},
    {"name": "dialog_rival", "screen_text": "?????: Okay! Il make my", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},
    # a real MID-PRINT partial read (the text box was still scrolling text in) — still a dialogue frame
    {"name": "dialog_partial_print", "screen_text": "This is pr", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},

    # menu — the real nurse YES/NO prompt (logged as menu: a cursor was up). RAM says menu.open.
    {"name": "menu_heal_yesno", "screen_text": "Shall we heal your POKMON?", "has_text": True,
     "ram_menu_open": True, "expect": "menu"},

    # regression: the all-lowercase continuation that BROKE the has_upper heuristic. Its real frame
    # was ZEROED by that very bug (the detector returned no-text), so this exact string is
    # reconstructed from the game rather than mined — the one case we can't harvest, by definition.
    {"name": "dialog_all_lowercase", "screen_text": "strong, they can protect me!", "has_text": True,
     "ram_menu_open": False, "expect": "dialogue"},
]
