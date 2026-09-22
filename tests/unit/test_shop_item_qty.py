"""_shop_item_qty resolves what to buy from a directive — including the realistic parsed form where
`has_item` is an item ID (not a name). Regression for the demo bug where id 11 was passed as '11'
(or 70 as 'Oaks Parcel') and matched nothing on the shelf."""
from pokemon_agent.agent.plan import Directive, Intent
from pokemon_agent.agent.reason_loop import ReasoningLoop


def test_has_item_id_maps_to_name():
    # _parse_done_when stores has_item as an item ID (resolve_item_id) — 11 == Antidote.
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc"}, success={"has_item": 11})
    assert ReasoningLoop._shop_item_qty(d) == ("Antidote", 1)


def test_target_item_and_qty_win():
    d = Directive(intent=Intent.SHOP, target={"kind": "map", "map": 56, "item": "Potion", "qty": 5},
                  success={"has_item": 20})
    assert ReasoningLoop._shop_item_qty(d) == ("Potion", 5)


def test_numeric_string_id_also_maps():
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc"}, success={"has_item": "11"})
    assert ReasoningLoop._shop_item_qty(d)[0] == "Antidote"


def test_none_when_no_item_goal():
    d = Directive(intent=Intent.TALK_TO, target={"kind": "npc"}, success={"talked_on_map": 1})
    assert ReasoningLoop._shop_item_qty(d)[0] is None
