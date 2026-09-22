"""Validate temporal ordering over absolute demonstration indices."""


def first_true(segment, predicate, scene_at):
    return next(
        (
            t
            for t in range(segment.t_start, segment.t_end + 1)
            if predicate(scene_at(segment, t))
        ),
        None,
    )


def last_true(segment, predicate, scene_at):
    return next(
        (
            t
            for t in range(segment.t_end, segment.t_start - 1, -1)
            if predicate(scene_at(segment, t))
        ),
        None,
    )


def validate_split(starts, finishes, *, entry=False):
    """Require first-new <= last-old for ordinary transitions.

    Refinement separately enforces strictly interior split boundaries.
    """
    if not starts or set(starts) != set(finishes):
        return False
    return all(
        start is not None
        and finishes[key] is not None
        and (start > finishes[key] and start != 0 if entry else start <= finishes[key])
        for key, start in starts.items()
    )


def validate_cfg(cfg):
    """Validate the complete recorded partition, including neighboring blocks.

    Ordinary transitions require first-new <= last-old, as clarified for
    POPL section 3.5. Persistent truth and equality at the boundary are allowed.
    Bindings and absolute boundaries are checked on both sides of each cut.
    """
    from collections import Counter

    from synthesis.cfg.refine import scene_at
    from synthesis.predicates.scene import evaluate

    cfg.validate_structure()
    for name in cfg.order:
        incoming, outgoing = cfg.incoming(name)[0], cfg.outgoing(name)[0]
        rows = cfg.demos.for_node(name)
        if not rows:
            return False
        for row in rows:
            try:
                if not evaluate(incoming.label, scene_at(row, row.t_start)):
                    return False
                if incoming.binding_condition is not None and not evaluate(
                    incoming.binding_condition, scene_at(row, row.t_start)
                ):
                    return False
                if not evaluate(outgoing.label, scene_at(row, row.t_end)):
                    return False
                first = first_true(
                    row, lambda state: evaluate(outgoing.label, state), scene_at
                )
                if incoming.source != cfg.entry:
                    last = last_true(
                        row, lambda state: evaluate(incoming.label, state), scene_at
                    )
                    if not validate_split({0: first}, {0: last}):
                        return False
                if outgoing.target != cfg.exit:
                    if first != row.t_end or row.t_end <= row.t_start:
                        return False
            except (KeyError, ValueError):
                return False
        if outgoing.target != cfg.exit:
            following = cfg.demos.for_node(outgoing.target)
            ends = Counter((id(r.trace), r.demo_idx, r.t_end) for r in rows)
            starts = Counter((id(r.trace), r.demo_idx, r.t_start) for r in following)
            if ends != starts:
                return False
    return True


Validate = validate_cfg
