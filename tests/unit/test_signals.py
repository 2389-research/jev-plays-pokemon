from pokemon_agent.agent.signals import party_hp_frac, needs_emergency_heal, min_level


def test_hp_frac_and_emergency():
    party = [{"hp": 2, "max_hp": 26}, {"hp": 0, "max_hp": 20}]
    assert 0.0 < party_hp_frac(party) < 0.1
    assert needs_emergency_heal(party) is True            # a fainted member
    healthy = [{"hp": 25, "max_hp": 26}]
    assert needs_emergency_heal(healthy) is False


def test_emergency_on_low_frac():
    assert needs_emergency_heal([{"hp": 3, "max_hp": 30}]) is True   # frac 0.1 < 0.15
    assert needs_emergency_heal([{"hp": 20, "max_hp": 30}]) is False


def test_min_level_and_empty_party():
    assert min_level([{"level": 8}, {"level": 5}]) == 5
    assert party_hp_frac([]) == 1.0                        # empty party -> treat as full (no false emergency)
    assert needs_emergency_heal([]) is False
    assert min_level([]) is None
