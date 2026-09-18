from pokemon_agent.agent.interaction_memory import InteractionMemory
from pokemon_agent.core.models import InteractAction, MoveAction, Direction, PlayerState


def player(x, y, facing, m=37):
    return PlayerState(x=x, y=y, map_id=m, facing=facing)


def test_interact_with_dialog_marks_talked():
    mem = InteractionMemory()
    p = player(5, 5, "north")  # facing north -> tile (5,4)
    mem.record_action(1, p, InteractAction(), caused_dialog=True)
    assert (37, 5, 4) in mem.talked


def test_interact_without_dialog_marks_empty_not_talked():
    mem = InteractionMemory()
    p = player(5, 5, "north")
    mem.record_action(1, p, InteractAction(), caused_dialog=False)
    assert (37, 5, 4) in mem.empty_tiles
    assert (37, 5, 4) not in mem.talked


def test_move_does_not_mark():
    mem = InteractionMemory()
    mem.record_action(1, player(5, 5, "north"), MoveAction(direction=Direction.NORTH), caused_dialog=False)
    assert len(mem.talked) == 0 and len(mem.empty_tiles) == 0


def test_annotate_flags_talked_npc():
    mem = InteractionMemory()
    p = player(5, 5, "north")
    mem.record_action(1, p, InteractAction(), caused_dialog=True)  # talked to (37,5,4)
    npcs = [{"x": 5, "y": 4, "facing": "south"}, {"x": 9, "y": 2, "facing": "left"}]
    out = mem.annotate_npcs(p, npcs)
    assert out[0]["talked_to"] is True   # the one we conversed with
    assert out[1]["talked_to"] is False  # a different NPC


def test_dialog_logged_and_deduped():
    mem = InteractionMemory()
    p = player(5, 5, "north")
    gs = {"dialog_active": True, "screen_text": "OAK: Hello there!"}
    mem.record_dialog(1, p, gs)
    mem.record_dialog(2, p, gs)  # same text -> not duplicated
    assert len(mem.dialog_log) == 1
    mem.record_dialog(3, p, {"dialog_active": True, "screen_text": "Choose a POKEMON"})
    assert len(mem.dialog_log) == 2
    assert "Choose a POKEMON" in mem.summary()["recent_dialog"]
