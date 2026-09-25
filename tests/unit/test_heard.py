"""HeardLog: complete messages, who/where, repeats, bounded memory with summaries (spec 2026-09-24).

runs/ss-anne-20260923 heard "That bush in front of the shop is in the way. There might be a way
around." four times; the old log kept typing fragments in a 30-entry ring and L1 never saw them."""
from __future__ import annotations

from pokemon_agent.agent.heard import DialogueParser, HeardLog

# frames exactly as the textbox rows (14, 16) read on the real save (Bill's dialogue)
BILL_FRAMES = [["Call me BILL!", "I'm a true blu"], ["Call me BILL!", "I'm a true blue"],
               ["I'm a true blue", "POKéMANIAC! Hey!"], ["POKéMANIAC! Hey!", "What's with that"]]
BUSH = [["That", ""], ["That bush in front of t", ""], ["That bush in front of the shop", ""],
        ["That bush in front of the shop", "is in the way."], ["is in the way.", ""],
        ["There", ""], ["There might be a way ar", ""], ["There might be a way around.", ""]]


def test_parser_stitches_typing_and_scrolling_into_one_message():
    p = DialogueParser()
    for f in BILL_FRAMES:
        p.feed(f)
    assert p.close() == "Call me BILL! I'm a true blue POKéMANIAC! Hey! What's with that"


def _say(h, frames, step, speaker="Cooltrainer M", at=(15, 18), mid=3):
    for i, f in enumerate(frames):
        h.observe(step + i, map_id=mid, map_name="Cerulean City", active=True, lines=f, speaker=speaker,
                  speaker_xy=at)
    return h.observe(step + len(frames), map_id=mid, map_name="Cerulean City", active=False, lines=[])


def test_a_message_closes_with_speaker_place_and_repeats_are_counted():
    h = HeardLog()
    m = _say(h, BUSH, 436)
    assert m["text"] == "That bush in front of the shop is in the way. There might be a way around."
    assert (m["speaker"], m["at"], m["map"]) == ("Cooltrainer M", [15, 18], 3)
    for k in range(3):
        _say(h, BUSH, 450 + 20 * k)
    assert h.distinct_count() == 1 and h.recent[-1]["count"] == 4
    assert "heard 4x" in h.since(400)[0] and "Cooltrainer M at (15,18) in Cerulean City" in h.since(400)[0]


def test_battle_text_is_ignored_and_a_battle_closes_an_open_message():
    h = HeardLog()
    h.observe(1, map_id=3, map_name="C", active=True, lines=["Hey! You!", ""], speaker="Rocket")
    closed = h.observe(2, map_id=3, map_name="C", active=True, lines=["Wild PIDGEY appeared!", ""], in_battle=True)
    assert closed["text"] == "Hey! You!"
    assert h.observe(3, map_id=3, map_name="C", active=False, lines=[]) is None
    assert h.distinct_count() == 1


def test_bounded_memory_folds_overflow_into_summaries():
    calls = []

    def summarize(kind, previous, messages):
        calls.append((kind, len(messages)))
        return f"{kind} summary of {len(messages)}"
    h = HeardLog(window=50, per_map=3)
    for i in range(6):
        _say(h, [[f"Message number {i} here", ""]], i * 30, speaker=f"NPC{i}")
    assert h.compact(summarize, min_aged=2)
    assert len(h.by_map[3]["messages"]) == 3 and h.by_map[3]["summary"] == "map summary of 3"
    assert h.digest.startswith("digest summary")
    assert ("map", 3) in calls


def test_summary_falls_back_deterministically_and_round_trips():
    h = HeardLog(window=10, per_map=1)
    _say(h, BUSH, 1)
    _say(h, [["Go see BILL on Route 25.", ""]], 100, speaker="Girl")
    h.compact(None, min_aged=1)
    assert "bush" in h.by_map[3]["summary"].lower() and "bush" in h.digest.lower()
    h2 = HeardLog.from_dict(h.to_dict())
    assert h2.digest == h.digest and h2.here(3)["messages"] == h.here(3)["messages"]


GUARD = [["The people here were", "robbed."], ["robbed.", "It's obvious that"], ["It's obvious that", "TEAM ROCKET is behind"],
         ["TEAM ROCKET is behind", "this crime!"]]


def test_a_conversation_reopened_on_the_closing_frame_is_two_messages_not_one():
    """runs/explore-live: A reopened the Guard's text with no idle frame between, and the end of one
    message merged with the start of the next ("...crime! The")."""
    h = HeardLog()
    for i, f in enumerate(GUARD + [["The", ""]] + GUARD):
        h.observe(i, map_id=3, map_name="Cerulean City", active=True, lines=f, speaker="Guard", speaker_xy=(28, 12))
    h.observe(20, map_id=3, map_name="Cerulean City", active=False, lines=[])
    msgs = h.by_map[3]["messages"]
    assert len(msgs) == 1 and msgs[0]["count"] == 2
    assert msgs[0]["text"].endswith("this crime!")


def test_the_same_words_from_different_places_stay_separate_and_ordered():
    """runs/sleeves-surge2: 16 trash cans all say "Nope, there's only trash here." — merged into one entry
    (credited to the first can, heard 81x), L1 couldn't tell which cans it had checked or in what order."""
    h = HeardLog()
    nope = [["Nope, there's only", "trash here."]]
    _say(h, nope, 10, speaker="trash can", at=(1, 9), mid=92)
    _say(h, [["Hey! There's a switch", "under the trash!"]], 20, speaker="trash can", at=(9, 11), mid=92)
    _say(h, nope, 30, speaker="trash can", at=(9, 9), mid=92)
    _say(h, nope, 40, speaker="trash can", at=(1, 9), mid=92)
    msgs = h.here(92)["messages"]
    assert len(msgs) == 3
    assert msgs[0].startswith("step 21: trash can at (9,11)")                # order = last heard
    assert "trash can at (9,9)" in msgs[1] and "trash can at (1,9)" in msgs[2] and "heard 2x" in msgs[2]
