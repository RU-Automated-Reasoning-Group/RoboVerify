"""Learn guards whose witness is unique on each recorded loop-head scene."""

import itertools
from dataclasses import dataclass

from synthesis.predicates.enumerate import SearchResult, enumerate_separator
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import free_names


def loop_guard_synthesis(
    positive, exits, variables, scope, *, language=None, candidates=()
):
    """Positive items are (scene, {bound variable: chosen object}); exits have no witness."""
    if not positive or not exits:
        return SearchResult("insufficient_examples")
    examples = []
    for scene, chosen in positive:
        if set(chosen) != set(variables):
            raise ValueError("Every guard witness must bind all guard variables")
        for assignment in itertools.product(scene.positions, repeat=len(variables)):
            aliases = dict(scene.bindings, **dict(zip(variables, assignment)))
            label = all(aliases[v] == chosen[v] for v in variables)
            examples.append(
                (Scene(scene.positions, aliases, scene.entry_positions), label)
            )
    for scene in exits:
        for assignment in itertools.product(scene.positions, repeat=len(variables)):
            examples.append(
                (
                    Scene(
                        scene.positions,
                        dict(scene.bindings, **dict(zip(variables, assignment))),
                        scene.entry_positions,
                    ),
                    False,
                )
            )
    for candidate in candidates:
        if free_names(candidate) <= set(scope) | set(variables) and all(
            evaluate(candidate, scene) == label for scene, label in examples
        ):
            return SearchResult("found", candidate, examined=1)
    return enumerate_separator(
        examples,
        tuple(sorted(set(scope) | set(variables))),
        mode="guard",
        language=language,
    )


LoopGuardSynthesis = loop_guard_synthesis
