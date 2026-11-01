"""Load and validate a JSON state-machine spec into a :class:`SherpaStateMachine`.

Spec format::

    {"name": "demo", "initial": "A",
     "states": [{"name": "A", "description": "start"}, {"name": "B"}],
     "transitions": [{"trigger": "go", "source": "A", "dest": "B",
                      "action": "my_action", "conditions": ["is_ready"]}]}

``action`` names are looked up in an *action registry* (name -> ``BaseAction``);
``conditions`` names in a *condition registry* (name -> callable).
"""

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from sherpa_ai.actions.base import BaseAction
from sherpa_ai.memory.state_machine import SherpaStateMachine


class SMSpecError(ValueError):
    """Raised when a state-machine spec is invalid."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_sm_spec(
    spec: Any,
    action_registry: Optional[Mapping[str, BaseAction]] = None,
    condition_registry: Optional[Mapping[str, Callable]] = None,
) -> list[str]:
    """Check a spec and return a list of error messages (empty means valid).

    Checks: required fields, unique state names, initial state exists,
    transition endpoints exist, unique trigger per source state, and (when
    registries are given) that referenced actions/conditions are registered.

    Args:
        spec: The parsed JSON spec.
        action_registry: Known action names, or None to skip action checks.
        condition_registry: Known condition names, or None to skip condition checks.

    Example:
        >>> validate_sm_spec({"initial": "A", "states": [{"name": "A"}], "transitions": []})
        []
    """
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    errors: list[str] = []
    states = spec.get("states")
    transitions = spec.get("transitions", [])
    if not isinstance(states, list) or not states:
        return ["'states' must be a non-empty list"]
    if not isinstance(transitions, list):
        return ["'transitions' must be a list"]

    names: set[str] = set()
    for s in states:
        n = s.get("name") if isinstance(s, dict) else None
        if not isinstance(n, str) or not n:
            errors.append(f"state without a valid 'name': {s!r}")
        elif n in names:
            errors.append(f"duplicate state name '{n}'")
        else:
            names.add(n)
    if spec.get("initial") not in names:
        errors.append(f"initial state '{spec.get('initial')}' is not a declared state")

    seen: set[tuple[str, str]] = set()
    for i, t in enumerate(transitions):
        if not isinstance(t, dict) or not all(k in t for k in ("trigger", "source", "dest")):
            errors.append(f"transition {i} needs 'trigger', 'source' and 'dest'")
            continue
        for end in ("source", "dest"):
            if t[end] not in names:
                errors.append(f"transition '{t['trigger']}': {end} '{t[end]}' is not a declared state")
        key = (t["source"], t["trigger"])
        if key in seen:
            errors.append(f"duplicate trigger '{t['trigger']}' from state '{t['source']}'")
        seen.add(key)
        if action_registry is not None and t.get("action") is not None:
            if t["action"] not in action_registry:
                errors.append(f"transition '{t['trigger']}': unknown action '{t['action']}'")
        if condition_registry is not None:
            for c in t.get("conditions", []):
                if c not in condition_registry:
                    errors.append(f"transition '{t['trigger']}': unknown condition '{c}'")
    return errors


def load_sm_spec(
    spec: Union[dict, str, Path],
    action_registry: Optional[Mapping[str, BaseAction]] = None,
    condition_registry: Optional[Mapping[str, Callable]] = None,
    sm_cls: Optional[type] = None,
) -> SherpaStateMachine:
    """Build a :class:`SherpaStateMachine` from a spec dict or a JSON file path.

    Raises:
        SMSpecError: if the spec is invalid (all problems are listed in ``errors``).

    Example:
        >>> sm = load_sm_spec({"initial": "A", "states": [{"name": "A"}, {"name": "B"}],
        ...                    "transitions": [{"trigger": "go", "source": "A", "dest": "B"}]})
        >>> sm.state
        'A'
    """
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    action_registry = action_registry or {}
    condition_registry = condition_registry or {}
    errors = validate_sm_spec(spec, action_registry, condition_registry)
    if errors:
        raise SMSpecError(errors)

    kwargs = {"sm_cls": sm_cls} if sm_cls else {}
    machine = SherpaStateMachine(
        name=spec.get("name", ""),
        states=[{"name": s["name"], "description": s.get("description", "")} for s in spec["states"]],
        initial=spec["initial"],
        **kwargs,
    )
    for t in spec.get("transitions", []):
        conds = [condition_registry[c] for c in t.get("conditions", [])] or None
        action = action_registry[t["action"]] if t.get("action") is not None else None
        machine.update_transition(t["trigger"], t["source"], t["dest"], conditions=conds, action=action)
    return machine
