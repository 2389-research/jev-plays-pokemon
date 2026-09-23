"""L1 tiered goals + notepad — pure bookkeeping (spec docs/superpowers/specs/2026-09-23-l1-tiered-goals-design.md).

Goals are L1's own primary/secondary/tertiary horizons (text + optional map-independent criterion);
the harness only validates, detects real changes (a reworded echo is not one), and keeps the
one-level `interrupted` note. Nothing here calls a model.
"""
from __future__ import annotations

from pokemon_agent.agent import goals as G
from pokemon_agent.agent.plan import AgentPlan, Goal, Goals

PARCEL = 70
B = 0xD16B   # party mon 1 base (hp @+1..2, level @+0x21, max hp @+0x22..23)


class Ram:
    def __init__(self, *, hp=20, maxhp=20, level=6, badges=0, items=()):
        m = {0xD163: 1, 0xD356: badges, B: 7, B + 1: hp >> 8, B + 2: hp & 0xFF, B + 0x21: level,
             B + 0x22: maxhp >> 8, B + 0x23: maxhp & 0xFF, 0xD31D: len(items)}
        for i, iid in enumerate(items):
            m[0xD31E + 2 * i], m[0xD31F + 2 * i] = iid, 1
        m[0xD31E + 2 * len(items)] = 0xFF
        self.mem = m

    def read_memory(self, addr, bank=None):
        return self.mem.get(addr, 0)


def _plan(primary="Earn the Boulder Badge", secondary="Deliver the parcel, then go to Pewter",
          tertiary="", t_dw=None, interrupted=None, notepad="", catch=None):
    p = AgentPlan(goals=Goals(primary=Goal(text=primary, done_when="badges>=1"),
                              secondary=Goal(text=secondary),
                              tertiary=Goal(text=tertiary, done_when=t_dw)),
                  notepad=notepad)
    if interrupted:
        p.interrupted = interrupted
    if catch is not None:
        p.battle_goals = {"catch": catch}
    return p


# ---- criterion grammar ------------------------------------------------------------------------
def test_allowed_criteria_are_kept():
    for dw in ("has_item:Oak's Parcel", "no_item:Oak's Parcel", "level>=12", "badges>=1",
               "hp_frac>=0.9", "verify:did the guard let us pass?"):
        assert G.clean_criterion(dw) == dw


def test_map_relative_and_garbage_criteria_are_dropped():
    for dw in ("on_map", "talked", "has_item:Nonexistent Thing", "beat brock", "", None, 7):
        assert G.clean_criterion(dw) is None


def test_clean_goal_keeps_text_drops_bad_criterion_and_caps_text():
    g = G.clean_goal({"text": "Reach Pewter City", "done_when": "on_map"})
    assert g == Goal(text="Reach Pewter City", done_when=None)
    assert len(G.clean_goal({"text": "x" * 500}).text) == G.GOAL_TEXT_MAX


def test_clean_goal_coerces_a_bare_string_and_rejects_other_types():
    assert G.clean_goal("Heal at the Pokémon Center") == Goal(text="Heal at the Pokémon Center")
    assert G.clean_goal(5) is None and G.clean_goal(["x"]) is None


# ---- goal status --------------------------------------------------------------------------------
def test_goal_status_met_unmet_none_unchecked():
    ram = Ram(hp=10, maxhp=20, level=6, items=(PARCEL,))
    assert G.goal_status(Goal(text="x"), ram) == "none"
    assert G.goal_status(Goal(text=""), ram) == "none"
    assert G.goal_status(Goal(text="x", done_when="verify:is it done?"), ram) == "unchecked"
    assert G.goal_status(Goal(text="x", done_when="has_item:Oak's Parcel"), ram) == "met"
    assert G.goal_status(Goal(text="x", done_when="no_item:Oak's Parcel"), ram) == "unmet"
    assert G.goal_status(Goal(text="x", done_when="hp_frac>=1.0"), ram) == "unmet"
    assert G.goal_status(Goal(text="x", done_when="level>=5"), ram) == "met"
    assert G.goal_status(Goal(text="x", done_when="badges>=1"), ram) == "unmet"


def test_statuses_covers_every_tier():
    st = G.statuses(_plan(tertiary="heal", t_dw="hp_frac>=1.0"), Ram(hp=5))
    assert st == {"primary": "unmet", "secondary": "none", "tertiary": "unmet"}


# ---- change detection ---------------------------------------------------------------------------
# a real captured no-op DECIDE (runs/brock-continue-20260922-2217 step 128): legacy keys + "catch": []
CAPTURED_NOOP = {"assessment": "Active step q9 (deliver Oak's Parcel) is still valid", "add": [], "remove": [],
                 "mission": "Defeat Brock in Pewter City to earn the Boulder Badge",
                 "milestone": "Deliver Oak's Parcel in Pallet Town, then travel north via Route 2 and "
                              "Viridian Forest to Pewter City and beat Brock",
                 "catch": []}


def test_captured_noop_with_empty_catch_is_no_change():
    assert G.detect_change(_plan(), CAPTURED_NOOP, step_edit=False) is None


def test_normalized_echo_is_no_change():
    p = _plan(tertiary="Heal at the Viridian Pokémon Center", t_dw="hp_frac>=1.0", notepad="a\nb")
    prop = {"goals": {"tertiary": {"text": "  heal at the viridian   pokémon center ", "done_when": "hp_frac>=1.0"},
                      "primary": {"text": "EARN THE BOULDER BADGE", "done_when": "badges>=1"}},
            "notepad": "a\nb", "catch": []}
    assert G.detect_change(p, prop, step_edit=False) is None


def test_criterion_only_change_counts_but_invalid_criterion_diff_does_not():
    p = _plan(tertiary="Grind", t_dw="level>=10")
    ch = G.detect_change(p, {"goals": {"tertiary": {"text": "Grind", "done_when": "level>=12"}}}, step_edit=False)
    assert ch is not None and ch.goals == {"tertiary": Goal(text="Grind", done_when="level>=12")}
    p2 = _plan(tertiary="Grind")
    assert G.detect_change(p2, {"goals": {"tertiary": {"text": "Grind", "done_when": "on_map"}}},
                           step_edit=False) is None


def test_tier_merge_absent_unchanged_and_empty_text_clears():
    p = _plan(tertiary="Heal", t_dw="hp_frac>=1.0")
    ch = G.detect_change(p, {"goals": {"tertiary": {"text": ""}}}, step_edit=False)
    assert ch.goals == {"tertiary": Goal()}
    G.apply_change(p, ch, pre_status={"tertiary": "unmet"})
    assert p.goals.tertiary == Goal() and p.goals.primary.text == "Earn the Boulder Badge"


def test_catch_rules():
    p = _plan(catch=["Pidgey"])
    assert G.detect_change(p, {"catch": []}, step_edit=True) is None          # echo [] never erases
    assert G.detect_change(p, {"catch": ["pidgey"]}, step_edit=False) is None  # same set
    assert G.detect_change(p, {"catch": ["Rattata"]}, step_edit=False).catch == ["Rattata"]
    assert G.detect_change(p, {"catch": "clear"}, step_edit=False).catch == "clear"
    assert G.detect_change(_plan(), {"catch": "clear"}, step_edit=False) is None  # nothing to clear


def test_interrupted_output_only_empty_or_null_is_meaningful():
    p = _plan(interrupted=Goal(text="Cross Viridian Forest"))
    assert G.detect_change(p, {"interrupted": {"text": "Cross Viridian Forest", "status": "none"}},
                           step_edit=False) is None
    assert G.detect_change(p, {"interrupted": ""}, step_edit=False).drop_interrupted
    assert G.detect_change(p, {"interrupted": None}, step_edit=False).drop_interrupted
    assert G.detect_change(_plan(), {"interrupted": ""}, step_edit=False) is None   # nothing to drop


def test_legacy_keys_only_with_a_step_edit_and_goals_tier_wins():
    p = _plan()
    legacy = {"milestone": "Get to Viridian Forest"}
    assert G.detect_change(p, legacy, step_edit=False) is None
    ch = G.detect_change(p, legacy, step_edit=True)
    assert ch.goals == {"secondary": Goal(text="Get to Viridian Forest")}
    both = {"milestone": "old back-filled text", "goals": {"secondary": {"text": "Beat Brock"}}}
    assert G.detect_change(p, both, step_edit=True).goals == {"secondary": Goal(text="Beat Brock")}


# ---- notepad ------------------------------------------------------------------------------------
def test_notepad_replace_and_unchanged():
    p = _plan(notepad="old")
    ch = G.detect_change(p, {"notepad": "new line"}, step_edit=False)
    G.apply_change(p, ch, pre_status={})
    assert p.notepad == "new line" and not p.notepad_truncated
    assert G.detect_change(p, {"notepad": "New Line "}, step_edit=False) is None


def test_notepad_truncates_at_a_line_boundary_and_flags():
    text = "\n".join(f"line {i:04d} " + "x" * 40 for i in range(60))
    out, cut = G.truncate_notepad(text)
    assert cut and len(out) <= G.NOTEPAD_MAX_CHARS and text.startswith(out) and text[len(out)] == "\n"


def test_notepad_single_long_line_hard_cuts():
    out, cut = G.truncate_notepad("y" * 3000)
    assert cut and out == "y" * G.NOTEPAD_MAX_CHARS


# ---- interrupted bookkeeping --------------------------------------------------------------------
def _divert(p, text, dw=None, pre="unmet"):
    ch = G.detect_change(p, {"goals": {"tertiary": {"text": text, "done_when": dw}}}, step_edit=False)
    G.apply_change(p, ch, pre_status={"tertiary": pre})
    return p


def test_diversion_records_the_paused_focus():
    p = _plan(tertiary="Cross Viridian Forest")
    _divert(p, "Heal at the Viridian Pokémon Center", "hp_frac>=1.0")
    assert p.interrupted == Goal(text="Cross Viridian Forest")


def test_forest_heal_shop_keeps_only_the_first_pause():
    p = _plan(tertiary="Cross Viridian Forest")
    _divert(p, "Heal", "hp_frac>=1.0")
    _divert(p, "Buy Potions at the Viridian Mart", "has_item:Potion")
    assert p.interrupted.text == "Cross Viridian Forest"


def test_return_with_normalized_same_wording_clears():
    p = _plan(tertiary="Heal", t_dw="hp_frac>=1.0", interrupted=Goal(text="Cross Viridian Forest"))
    _divert(p, "cross  viridian forest", pre="unmet")
    assert p.interrupted == Goal()


def test_return_with_different_wording_clears_only_if_the_diversion_was_met():
    p = _plan(tertiary="Heal", t_dw="hp_frac>=1.0", interrupted=Goal(text="Cross Viridian Forest"))
    _divert(p, "Walk north through the forest to Pewter", pre="unmet")
    assert p.interrupted.text == "Cross Viridian Forest"
    p2 = _plan(tertiary="Heal", t_dw="hp_frac>=1.0", interrupted=Goal(text="Cross Viridian Forest"))
    _divert(p2, "Walk north through the forest to Pewter", pre="met")
    assert p2.interrupted == Goal()


def test_met_tertiary_replaced_is_not_recorded_as_paused():
    p = _plan(tertiary="Heal", t_dw="hp_frac>=1.0")
    _divert(p, "Cross Viridian Forest", pre="met")
    assert p.interrupted == Goal()


def test_clearing_the_tertiary_never_sets_interrupted():
    p = _plan(tertiary="Cross Viridian Forest")
    _divert(p, "", pre="unmet")
    assert p.interrupted == Goal()


def test_set_and_explicit_clear_in_the_same_decide_ends_empty():
    p = _plan(tertiary="Cross Viridian Forest")
    ch = G.detect_change(p, {"goals": {"tertiary": {"text": "Grind on Route 2"}}, "interrupted": ""},
                         step_edit=False)
    G.apply_change(p, ch, pre_status={"tertiary": "none"})
    assert p.interrupted == Goal() and p.goals.tertiary.text == "Grind on Route 2"


def test_interrupted_own_criterion_met_clears_at_review():
    p = _plan(tertiary="Shop", interrupted=Goal(text="Get to level 10", done_when="level>=10"))
    assert G.refresh_interrupted(p, Ram(level=11)) is True and p.interrupted == Goal()
    p2 = _plan(interrupted=Goal(text="Get to level 10", done_when="level>=10"))
    assert G.refresh_interrupted(p2, Ram(level=6)) is False and p2.interrupted.text


def test_verify_or_no_criterion_interrupted_never_auto_clears():
    for dw in (None, "verify:did we cross the forest?"):
        p = _plan(interrupted=Goal(text="Cross the forest", done_when=dw))
        assert G.refresh_interrupted(p, Ram()) is False and p.interrupted.text


# ---- AgentPlan model ----------------------------------------------------------------------------
def test_old_checkpoint_seeds_goals_from_mission_milestone_and_drops_tried_failed():
    old = {"mission": "Beat Brock", "milestone": "Deliver the parcel", "tried_failed": ["a", "b"]}
    p = AgentPlan.model_validate(old)
    assert p.goals.primary.text == "Beat Brock" and p.goals.secondary.text == "Deliver the parcel"
    assert p.tried_failed == [] and p.notepad == ""


def test_goals_win_over_stale_mission_on_load():
    p = AgentPlan.model_validate({"mission": "stale", "goals": {"primary": {"text": "Beat Brock"}}})
    assert p.mission == "Beat Brock"


def test_apply_goals_keeps_mission_milestone_in_sync():
    p = _plan()
    p.apply_goals({"primary": Goal(text="Beat Misty"), "secondary": Goal(text="Reach Cerulean")})
    assert p.mission == "Beat Misty" and p.milestone == "Reach Cerulean"


def test_prompt_view_hides_the_notepad_and_ledgers():
    p = _plan(notepad="secret-ish notes")
    view = p.prompt_view()
    assert "notepad" not in view and "tried_failed" not in view and "hypotheses" not in view
    assert view["goals"]["primary"]["text"] == "Earn the Boulder Badge"
