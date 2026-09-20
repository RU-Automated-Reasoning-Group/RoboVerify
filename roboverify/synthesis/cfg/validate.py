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
    """Reject absent witnesses, i_s <= i_f, and entry splits at timestep zero."""
    if not starts or set(starts) != set(finishes):
        return False
    return all(
        start is not None
        and finishes[key] is not None
        and start > finishes[key]
        and (not entry or start != 0)
        for key, start in starts.items()
    )


Validate = validate_split
