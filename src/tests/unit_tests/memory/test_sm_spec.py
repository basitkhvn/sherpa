import json

import pytest

from sherpa_ai.actions.empty import EmptyAction
from sherpa_ai.memory.sm_spec import SMSpecError, load_sm_spec, validate_sm_spec

SPEC = {
    "name": "demo",
    "initial": "A",
    "states": [{"name": "A", "description": "start"}, {"name": "B"}, {"name": "C"}],
    "transitions": [
        {"trigger": "a_to_b", "source": "A", "dest": "B", "action": "act"},
        {"trigger": "b_to_c", "source": "B", "dest": "C", "conditions": ["yes"]},
    ],
}


def test_valid_spec_has_no_errors():
    assert validate_sm_spec(SPEC, {"act": EmptyAction()}, {"yes": lambda: True}) == []


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda s: s.__setitem__("initial", "Z"), "initial state"),
        (lambda s: s["states"].append({"name": "A"}), "duplicate state"),
        (lambda s: s["transitions"][0].__setitem__("dest", "Q"), "not a declared state"),
        (lambda s: s["transitions"].append(dict(s["transitions"][0])), "duplicate trigger"),
        (lambda s: s["transitions"][0].__setitem__("action", "nope"), "unknown action"),
        (lambda s: s["transitions"][1].__setitem__("conditions", ["nope"]), "unknown condition"),
    ],
)
def test_invalid_specs(mutate, fragment):
    spec = json.loads(json.dumps(SPEC))
    mutate(spec)
    errs = validate_sm_spec(spec, {"act": EmptyAction()}, {"yes": lambda: True})
    assert any(fragment in e for e in errs), errs


def test_non_dict():
    assert validate_sm_spec([]) == ["spec must be a JSON object"]


def test_load_builds_machine(tmp_path):
    path = tmp_path / "sm.json"
    path.write_text(json.dumps(SPEC))
    sm = load_sm_spec(path, {"act": EmptyAction()}, {"yes": lambda: True})
    assert sm.state == "A"
    assert sm.explicit_transitions == {"a_to_b", "b_to_c"}
    sm.sm.set_state("B")
    sm.b_to_c()  # guarded by the registered condition
    assert sm.state == "C"


def test_load_raises_with_all_errors():
    spec = json.loads(json.dumps(SPEC))
    spec["initial"] = "Z"
    spec["transitions"][0]["dest"] = "Q"
    with pytest.raises(SMSpecError) as e:
        load_sm_spec(spec, {"act": EmptyAction()}, {"yes": lambda: True})
    assert len(e.value.errors) == 2


def test_two_loaded_machines_do_not_share_triggers():
    a = load_sm_spec(SPEC, {"act": EmptyAction()}, {"yes": lambda: True})
    b = load_sm_spec({"initial": "A", "states": [{"name": "A"}, {"name": "B"}],
                      "transitions": [{"trigger": "only_b", "source": "A", "dest": "B"}]})
    assert b.explicit_transitions == {"only_b"} and "a_to_b" in a.explicit_transitions
