"""Flat adjacent-fragment quotient with witnessed bindings and unique guards."""

from copy import copy, deepcopy
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
from synthesis.cfg.demos import DemoAssignment, DemoSegment
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.iterations import extract_iterations
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
from synthesis.predicates.term import conjunction, exists, negate, ref, substitute


@dataclass
class Repetition:
    start: int
    width: int
    substitutions: tuple
    template: object


def find_repetition(labels, excluded=frozenset()):
    word = (
        tuple(labels)
        if labels and isinstance(labels[0], Letter)
        else encode_fragment(labels)
    )
    for width in range(1, len(word) // 2 + 1):
        for start in range(len(word) - 2 * width + 1):
            if (start, width) in excluded:
                continue
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
                    update = carried_bindings(
                        Template(template.word, substitutions[-1], mapping)
                    )
                except ValueError:
                    break
                if update != expected_update:
                    break
                substitutions.append(mapping)
                cursor += width
            return Repetition(start, width, tuple(substitutions), template)
    return None


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


def _fold(cfg, repetition, language, infer_invariant):
    r = repetition
    carry, rebound = carried_bindings(r.template)
    if not carry or not rebound:
        return False
    available = set(graph_scope(cfg)[cfg.order[r.start]])
    occupied = set(available)

    def fresh(preferred):
        name = preferred
        while name in occupied:
            name += "_loop"
        occupied.add(name)
        return name

    names = {key: fresh("b" if i == 0 else f"b{i}") for i, key in enumerate(carry)}
    names.update(
        {
            key: fresh("b_prime" if i == 0 else f"b_prime{i}")
            for i, key in enumerate(rebound)
        }
    )
    updates = tuple((names[a], names[b]) for a, b in carry.items())
    if any(
        right in {a for a, _ in updates[:i]} for i, (_, right) in enumerate(updates)
    ):
        return False
    init = tuple((names[key], r.template.first[key].value) for key in carry)
    posts = tuple(
        substitute(letter.predicate, {key: ref(value) for key, value in names.items()})
        for letter in r.template.word
    )
    rebound_names = tuple(names[key] for key in rebound)
    examples = cfg.demos.for_node(cfg.order[r.start])
    if not examples:
        return False
    extracted = []
    for row in examples:
        try:
            bindings = dict(row.bindings)
            initial = scene_at(row, row.t_start)
            bindings.update({left: initial.bindings[right] for left, right in init})
            suffix = DemoSegment(
                row.demo_idx,
                row.t_start,
                len(row.trace.states) - 1,
                row.trace,
                bindings,
                row,
                row.t_start,
            )
            iterations = extract_iterations(suffix, posts, rebound_names, updates)
        except (KeyError, ValueError):
            return False
        if not iterations.bodies:
            return False
        extracted.append(iterations)
    positive, heads, exits = [], [], []
    body_demos = [[] for _ in posts]
    loop_demos = []
    for row, iterations in zip(examples, extracted):
        for body in iterations.bodies:
            head = body[0]
            heads.append(head)
            positive.append(
                (
                    scene_at(head, head.t_start),
                    {n: head.bindings[n] for n in rebound_names},
                )
            )
            for slot, segment in enumerate(body):
                body_demos[slot].append(segment)
        terminal = iterations.terminal
        exits.append(scene_at(terminal, terminal.t_start))
        loop_demos.append(
            DemoSegment(
                row.demo_idx,
                row.t_start,
                terminal.t_end,
                row.trace,
                iterations.bodies[0][0].bindings,
                row,
                row.t_start,
                terminal.bindings,
            )
        )
    loop_scope = available | {left for left, _ in init}
    learned = loop_guard_synthesis(
        positive,
        exits,
        rebound_names,
        loop_scope,
        language=language,
        candidates=(conjunction(*posts),),
    )
    if not learned:
        return False
    terminals = tuple(it.terminal for it in extracted)
    invariant = (
        None
        if infer_invariant is None
        else infer_invariant(heads + list(terminals), learned.term, loop_scope)
    )
    inverse = {value.value: names[key] for key, value in r.template.first.items()}
    body = []
    for slot, node in enumerate(cfg.order[r.start : r.start + r.width]):
        old = cfg.nodes[node].region
        region = BlockRegion(None)
        if isinstance(old, BlockRegion):
            try:
                aliases = {}
                for name in sorted(loop_scope | set(rebound_names)):
                    ids = {
                        scene_at(
                            it.bodies[0][slot], it.bodies[0][slot].t_start
                        ).bindings[name]
                        for it in extracted
                    }
                    if len(ids) == 1:
                        aliases.setdefault(ids.pop(), name)
                region = BlockRegion(
                    (
                        None
                        if old.symbolic is None
                        else tuple(
                            _rename_instruction(i, inverse) for i in old.symbolic
                        )
                    ),
                    tuple(
                        _rename_instruction(name_operands(i, aliases), inverse)
                        for i in old.physical
                    ),
                )
            except (KeyError, ValueError):
                # Structural discovery does not require a reusable controller.
                pass
        body.append(region)
    body_names = [f"body{i}" for i in range(len(body))]
    body_edges = [Edge("entry", body_names[0], learned.term)]
    body_edges.extend(
        Edge(name, body_names[i + 1] if i + 1 < len(body_names) else "exit", posts[i])
        for i, name in enumerate(body_names)
    )
    body_cfg = RelationalCFG(
        dict(zip(body_names, (Node(n, b) for n, b in zip(body_names, body)))),
        body_edges,
        body_names,
        DemoAssignment(dict(zip(body_names, body_demos))),
        learned.term,
        posts[-1],
        initial_scope=frozenset(loop_scope | set(rebound_names)),
    )
    region = LoopRegion(
        learned.term,
        rebound_names,
        tuple(body),
        init,
        updates,
        invariant,
        tuple(len(it.bodies) for it in extracted),
        posts,
        tuple(tuple(rows) for rows in body_demos),
        require_unique_guard=True,
        body_cfg=body_cfg,
        exit_demos=terminals,
    )
    first, last = r.start, r.start + r.width * len(r.substitutions)
    removed = cfg.order[first:last]
    new = removed[0] + ".loop"
    incoming, outgoing = cfg.incoming(removed[0])[0], cfg.outgoing(removed[-1])[0]
    exit_label = negate(exists(rebound_names, learned.term))
    trial = copy(cfg)
    trial.nodes, trial.order, trial.edges = (
        dict(cfg.nodes),
        list(cfg.order),
        list(cfg.edges),
    )
    trial.demos = DemoAssignment(dict(cfg.demos.segments))
    index = trial.edges.index(incoming)
    trial.edges[index : index + len(removed) + 1] = [
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
            cfg.postcondition if outgoing.target == cfg.exit else exit_label,
        ),
    ]
    trial.order[first:last] = [new]
    for name in removed:
        del trial.nodes[name]
        del trial.demos.segments[name]
    trial.nodes[new] = Node(new, region)
    trial.demos.segments[new] = loop_demos
    # The loop may explain more of a demo than the original detected pair.
    # Repartition its continuation from the newly recovered terminal head.
    cursors = [it.terminal.t_end for it in extracted]
    for name in trial.order[first + 1 :]:
        rows = []
        old_rows = trial.demos.for_node(name)
        if len(old_rows) != len(examples):
            return False
        post = trial.outgoing(name)[0].label
        for i, (old, it) in enumerate(zip(old_rows, extracted)):
            bindings = dict(old.bindings, **it.terminal.bindings)
            remaining = DemoSegment(
                old.demo_idx,
                cursors[i],
                len(old.trace.states) - 1,
                old.trace,
                bindings,
                old,
                old.entry_index,
            )
            end = (
                remaining.t_end
                if trial.outgoing(name)[0].target == cfg.exit
                else next(
                    (
                        t
                        for t in range(cursors[i] + 1, remaining.t_end + 1)
                        if evaluate(post, scene_at(remaining, t))
                    ),
                    None,
                )
            )
            if end is None:
                return False
            rows.append(
                DemoSegment(
                    old.demo_idx,
                    cursors[i],
                    end,
                    old.trace,
                    bindings,
                    old,
                    old.entry_index,
                )
            )
            cursors[i] = end
        trial.demos.segments[name] = rows
    from synthesis.cfg.validate import validate_cfg

    if not validate_cfg(trial):
        return False
    cfg.__dict__.update(trial.__dict__)
    return True


def quotient(cfg, *, language=None, infer_invariant=None):
    """Collapse flat repetitions to a fixed point; failed matches leave no edits."""
    cfg.validate_structure()
    changed, excluded = False, set()
    while True:
        word = [
            Letter(
                e.binding_condition if e.binding_condition is not None else e.label,
                e.binds,
                isinstance(cfg.nodes[name].region, LoopRegion),
            )
            for name in cfg.order
            for e in cfg.outgoing(name)
        ]
        repetition = find_repetition(word, excluded)
        if repetition is None:
            return changed
        if _fold(cfg, repetition, language, infer_invariant):
            changed, excluded = True, set()
        else:
            excluded.add((repetition.start, repetition.width))


Quotient = quotient
ExtractIterations = extract_iterations
