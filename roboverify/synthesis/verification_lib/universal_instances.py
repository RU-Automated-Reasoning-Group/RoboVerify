"""Weaken conjunctive premises to quantifier-free universal instances.

Every returned formula follows from the input premises. There is no domain
closure: UNSAT with a violation proves the original unbounded violation UNSAT.
SAT/UNKNOWN here are inconclusive and must fall back to the original query.
"""

from itertools import islice, product

import z3


def is_quantifier_free(expr, cache=None):
    cache = {} if cache is None else cache
    key = expr.get_id()
    if key not in cache:
        cache[key] = not z3.is_quantifier(expr) and all(
            is_quantifier_free(child, cache) for child in expr.children()
        )
    return cache[key]


def universal_instance_relaxation(premises, candidates, *, max_instances=256):
    """Return a necessary condition, or None when no quantified premise occurs.

    Keep quantifier-free conjuncts and instantiate positive, top-level forall
    conjuncts with quantifier-free bodies. Drop other quantified conjuncts in
    full, including negative universals and nested/alternating quantifiers.
    Recursively instantiating those without polarity handling could strengthen
    the premises and would be unsound for this proof shortcut.
    """
    if max_instances < 0:
        raise ValueError("Instance budget must be nonnegative")
    candidates = tuple(dict.fromkeys(candidates))
    cache = {}
    result = []
    quantified = False
    remaining = max_instances

    def conjuncts(expr):
        if z3.is_and(expr):
            for child in expr.children():
                yield from conjuncts(child)
        else:
            yield expr

    for part in (c for premise in premises for c in conjuncts(premise)):
        if is_quantifier_free(part, cache):
            result.append(part)
            continue
        quantified = True
        if not (
            z3.is_quantifier(part)
            and part.is_forall()
            and is_quantifier_free(part.body(), cache)
        ):
            continue
        choices = [
            tuple(c for c in candidates if c.sort() == part.var_sort(i))
            for i in range(part.num_vars())
        ]
        for values in islice(product(*choices), remaining):
            result.append(z3.substitute_vars(part.body(), *reversed(values)))
            remaining -= 1
    return tuple(result) if quantified else None
