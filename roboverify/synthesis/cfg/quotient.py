"""Flat adjacent-fragment quotient with witnessed bindings and unique guards."""

from copy import deepcopy
from dataclasses import dataclass

from synthesis.api.instructions import (
    Assign,
    MoveByName,
    PickByName,
    PickPlaceByName,
    Put,
    ReleaseByName,
    Skip,
)
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.graph import Edge, Node
from synthesis.cfg.kleene import (
    Letter,
    Template,
    anti_unify,
    carried_bindings,
    encode_fragment,
    match_template,
)
from synthesis.cfg.physical import name_operands
from synthesis.cfg.refine import scene_at
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.scope import scope as graph_scope
from synthesis.predicates.guard import loop_guard_synthesis
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import conjunction, ref, substitute


@dataclass
class Repetition:
    start: int
    width: int
    substitutions: tuple
    template: object


def find_repetition(labels):
    word = (
        tuple(labels)
        if labels and isinstance(labels[0], Letter)
        else encode_fragment(labels)
    )
    for width in range(1, len(word) // 2 + 1):
        for start in range(len(word) - 2 * width + 1):
            template = anti_unify(
                word[start : start + width], word[start + width : start + 2 * width]
            )
            if template is None or not template.first:
                continue
            try:
                expected_update = carried_bindings(template)
            except ValueError:
                continue
            substitutions = [template.first, template.second]
            cursor = start + 2 * width
            while cursor + width <= len(word):
                mapping = match_template(template, word[cursor : cursor + width])
                if mapping is None:
                    break
                try:
                    update = carried_bindings(Template(template.word, substitutions[-1], mapping))
                except ValueError:
                    break
                if update != expected_update:
                    break
                substitutions.append(mapping)
                cursor += width
            return Repetition(start, width, tuple(substitutions), template)
    return None


def extract_iterations(segments_by_node, start, width, count):
    """Group contiguous recorded segments, keeping original absolute indices."""
    groups = []
    for iteration in range(count):
        columns = segments_by_node[
            start + iteration * width : start + (iteration + 1) * width
        ]
        keys = [(s.demo_idx, s.t_start) for s in columns[0]]
        if any(len(column) != len(keys) for column in columns):
            raise ValueError("Unaligned iteration demonstrations")
        result = []
        for row in zip(*columns):
            if any(
                s.demo_idx != row[0].demo_idx or s.trace is not row[0].trace
                for s in row
            ):
                raise ValueError("Mismatched demonstration provenance")
            if any(a.t_end != b.t_start for a, b in zip(row, row[1:])):
                raise ValueError("Iteration fragments are not contiguous")
            result.append(
                DemoSegment(
                    row[0].demo_idx,
                    row[0].t_start,
                    row[-1].t_end,
                    row[0].trace,
                    row[0].bindings,
                    row[0],
                )
            )
        groups.append(result)
    return groups


def _rename_instruction(instruction, names):
    result = deepcopy(instruction)
    fields = {
        Put: ("upper_block", "base_block"),
        Assign: ("left", "right"),
        PickByName: ("grab_box_name",),
        MoveByName: ("target_box_name_x", "target_box_name_y", "target_box_name_z"),
        ReleaseByName: ("release_box_name",),
        PickPlaceByName: (
            "grab_box_name",
            "target_box_name_x",
            "target_box_name_y",
            "target_box_name_z",
        ),
    }
    if isinstance(result, Skip):
        return result
    if type(result) not in fields:
        raise ValueError(f"Cannot generalize instruction {type(result).__name__}")
    for field in fields[type(result)]:
        old = getattr(result, field)
        setattr(result, field, names.get(old, old))
    return result


def quotient(cfg, *, language=None, infer_invariant=None):
    """One flat collapse. Missing data/guard/invariant leaves the graph unchanged."""
    cfg.validate_structure()
    word = []
    for node in cfg.order:
        edge = cfg.outgoing(node)[0]
        word.append(
            Letter(
                (
                    edge.binding_condition
                    if edge.binding_condition is not None
                    else edge.label
                ),
                edge.binds,
                isinstance(cfg.nodes[node].region, LoopRegion),
            )
        )
    repetition = find_repetition(word)
    if repetition is None:
        return False
    r = repetition
    carry, rebound = carried_bindings(r.template)
    if not carry or not rebound:
        return False
    occupied = set(graph_scope(cfg)[cfg.order[r.start]])

    def fresh(preferred):
        name = preferred
        while name in occupied:
            name += "_loop"
        occupied.add(name)
        return name

    names = {name: fresh("b" if i == 0 else f"b{i}") for i, name in enumerate(carry)}
    names.update(
        {
            name: fresh("b_prime" if i == 0 else f"b_prime{i}")
            for i, name in enumerate(rebound)
        }
    )
    # Reject cyclic parallel updates: sequential Assign would change their meaning.
    updates = tuple((names[a], names[b]) for a, b in carry.items())
    if any(
        right in {a for a, _ in updates[:i]} for i, (_, right) in enumerate(updates)
    ):
        return False
    try:
        groups = extract_iterations(
            [cfg.demos.for_node(n) for n in cfg.order],
            r.start,
            r.width,
            len(r.substitutions),
        )
    except ValueError:
        return False
    if not groups or any(not group for group in groups):
        return False
    entries = {s.demo_idx: s.t_start for s in groups[0]}
    for group in groups:
        for segment in group:
            segment.entry_index = entries[segment.demo_idx]
    positive, exits, loop_segments = [], [], []
    for iteration, (group, mapping) in enumerate(zip(groups, r.substitutions)):
        for segment in group:
            head = scene_at(segment, segment.t_start)
            try:
                bindings = {
                    names[key]: head.bindings[value.value]
                    for key, value in mapping.items()
                }
            except KeyError:
                return False
            head = Scene(
                head.positions, dict(head.bindings, **bindings), head.entry_positions
            )
            positive.append(
                (head, {names[key]: bindings[names[key]] for key in rebound})
            )
            loop_segments.append(
                DemoSegment(
                    segment.demo_idx,
                    segment.t_start,
                    segment.t_end,
                    segment.trace,
                    dict(segment.bindings, **bindings),
                    segment,
                    entries[segment.demo_idx],
                )
            )
            if iteration == len(groups) - 1:
                final = scene_at(segment, segment.t_end)
                for left, right in updates:
                    bindings[left] = bindings[right]
                exits.append(
                    Scene(
                        final.positions,
                        dict(final.bindings, **bindings),
                        final.entry_positions,
                    )
                )
    scope = set(graph_scope(cfg)[cfg.order[r.start]]) | set(names[key] for key in carry)
    hint = conjunction(
        *(
            substitute(
                letter.predicate, {key: ref(value) for key, value in names.items()}
            )
            for letter in r.template.word
        )
    )
    learned = loop_guard_synthesis(
        positive,
        exits,
        tuple(names[k] for k in rebound),
        scope,
        language=language,
        candidates=(hint,),
    )
    if not learned or infer_invariant is None:
        return False
    invariant = infer_invariant(loop_segments, learned.term, scope)
    if invariant is None:
        return False
    inverse = {value.value: names[key] for key, value in r.template.first.items()}
    body = []
    for node in cfg.order[r.start : r.start + r.width]:
        region = cfg.nodes[node].region
        if not isinstance(region, BlockRegion):
            return False
        try:
            # Numeric candidates may become named physical loop bodies, but do not
            # acquire a relational summary merely by matching demonstrations.
            rows = cfg.demos.for_node(node)
            aliases = {}
            for key, variable in inverse.items():
                ids = {scene_at(row, row.t_start).bindings[key] for row in rows}
                if len(ids) == 1:
                    aliases[ids.pop()] = variable
            for key in sorted(scope - set(names.values())):
                ids = {scene_at(row, row.t_start).bindings[key] for row in rows}
                if len(ids) == 1:
                    aliases.setdefault(ids.pop(), key)
            body.append(
                BlockRegion(
                    (
                        None
                        if region.symbolic is None
                        else tuple(
                            _rename_instruction(i, inverse) for i in region.symbolic
                        )
                    ),
                    tuple(
                        _rename_instruction(name_operands(i, aliases), inverse)
                        for i in region.physical
                    ),
                )
            )
        except (KeyError, ValueError):
            return False
    init = tuple((names[key], r.template.first[key].value) for key in carry)
    posts = tuple(
        substitute(letter.predicate, {key: ref(value) for key, value in names.items()})
        for letter in r.template.word
    )
    body_demos = []
    for slot in range(r.width):
        examples = []
        for iteration, mapping in enumerate(r.substitutions):
            original = cfg.demos.for_node(
                cfg.order[r.start + iteration * r.width + slot]
            )
            for segment in original:
                head = scene_at(segment, segment.t_start)
                bindings = {
                    names[key]: head.bindings[value.value]
                    for key, value in mapping.items()
                }
                examples.append(
                    DemoSegment(
                        segment.demo_idx,
                        segment.t_start,
                        segment.t_end,
                        segment.trace,
                        dict(segment.bindings, **bindings),
                        segment,
                        entries[segment.demo_idx],
                    )
                )
        body_demos.append(tuple(examples))
    region = LoopRegion(
        learned.term,
        tuple(names[k] for k in rebound),
        tuple(body),
        init,
        updates,
        invariant,
        tuple(len(groups) for _ in groups[0]),
        posts,
        tuple(body_demos),
        require_unique_guard=True,
    )
    first, last = r.start, r.start + r.width * len(groups)
    removed = cfg.order[first:last]
    new = removed[0] + ".loop"
    incoming, outgoing = cfg.incoming(removed[0])[0], cfg.outgoing(removed[-1])[0]
    edge_index = cfg.edges.index(incoming)
    cfg.edges[edge_index : edge_index + len(removed) + 1] = [
        Edge(
            incoming.source,
            new,
            incoming.label,
            incoming.binds,
            incoming.kills,
            incoming.binding_condition,
        ),
        Edge(
            new,
            outgoing.target,
            outgoing.label,
            outgoing.binds,
            outgoing.kills,
            outgoing.binding_condition,
        ),
    ]
    cfg.order[first:last] = [new]
    for node in removed:
        del cfg.nodes[node]
        del cfg.demos.segments[node]
    cfg.nodes[new] = Node(new, region)
    cfg.demos.segments[new] = [
        DemoSegment(
            first.demo_idx,
            first.t_start,
            last.t_end,
            first.trace,
            first.bindings,
            first,
        )
        for first, last in zip(groups[0], groups[-1])
    ]
    return True


Quotient = quotient
ExtractIterations = extract_iterations
