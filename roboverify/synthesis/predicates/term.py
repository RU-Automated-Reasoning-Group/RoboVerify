"""Immutable, interned formulas; named binders canonicalize alpha-equivalence."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Term:
    op: str
    args: tuple = ()
    value: object = None

    @property
    def depth(self):
        return 1 + max((arg.depth for arg in self.args), default=0)

    @property
    def quantifier_count(self):
        return int(self.op in ("exists", "forall")) + sum(a.quantifier_count for a in self.args)

    def __str__(self):
        if self.op == "ref":
            return str(self.value)
        if self.op == "bool":
            return str(self.value)
        if self.op in ("exists", "forall"):
            return f"{self.op} {','.join(self.value)}.({self.args[0]})"
        return f"{self.op}({', '.join(map(str, self.args))})"


@lru_cache(maxsize=100000)
def _term(op, args=(), value=None):
    return Term(op, args, value)


def ref(name):
    if not isinstance(name, str) or not name or name.startswith("@"):
        raise ValueError("Reference names must be nonempty and cannot start with @")
    return _term("ref", value=name)


def boolean(value):
    if not isinstance(value, bool):
        raise TypeError("Boolean literal required")
    return _term("bool", value=value)


def atom(name, left, right):
    if name not in {"ON_star", "ON_star_zero", "Higher", "Scattered", "ON", "eq"}:
        raise ValueError(f"Unknown predicate: {name}")
    if left.op != "ref" or right.op != "ref":
        raise TypeError("Predicate arguments must be object references")
    return _term(name, (left, right))


def negate(term):
    return _term("not", (term,))


def conjunction(*terms):
    return _term("and", tuple(terms)) if terms else boolean(True)


def disjunction(*terms):
    return _term("or", tuple(terms)) if terms else boolean(False)


def implies(left, right):
    return _term("implies", (left, right))


def canonical(term, env=None, depth=0):
    env = {} if env is None else env
    if term.op == "ref":
        return _term("ref", value=env.get(term.value, term.value))
    if term.op in ("forall", "exists"):
        names = tuple(f"@{depth+i}" for i in range(len(term.value)))
        nested = dict(env, **dict(zip(term.value, names)))
        body = canonical(term.args[0], nested, depth + len(names))
        return _term(term.op, (body,), names)
    return _term(term.op, tuple(canonical(a, env, depth) for a in term.args), term.value)


def quantify(kind, names, body):
    names = tuple(names)
    if kind not in ("exists", "forall") or not names or len(set(names)) != len(names):
        raise ValueError("Quantifiers require a kind and distinct binders")
    for name in names:
        ref(name)
    return canonical(_term(kind, (body,), names))


def exists(names, body):
    return quantify("exists", names, body)


def forall(names, body):
    return quantify("forall", names, body)


def free_names(term, bound=frozenset()):
    if term.op == "ref":
        return frozenset() if term.value in bound else frozenset([term.value])
    if term.op in ("forall", "exists"):
        bound = bound | frozenset(term.value)
    return frozenset().union(*(free_names(a, bound) for a in term.args))


def substitute(term, mapping, bound=frozenset()):
    """Capture-free substitution of free object references, not formula holes."""
    def walk(node, hidden):
        if node.op == "ref":
            return node if node.value in hidden else mapping.get(node.value, node)
        if node.op in ("forall", "exists"):
            hidden = hidden | frozenset(node.value)
        return _term(node.op, tuple(walk(a, hidden) for a in node.args), node.value)
    return canonical(walk(term, bound))


def to_z3(term, context, bindings=None):
    import z3
    bindings = {} if bindings is None else bindings
    if term.op == "ref":
        return bindings[term.value] if term.value in bindings else context.get_consts(term.value)
    if term.op == "bool":
        return z3.BoolVal(term.value)
    if term.op in ("exists", "forall"):
        variables = [z3.Const(name, context.BoxSort) for name in term.value]
        body = to_z3(term.args[0], context, dict(bindings, **dict(zip(term.value, variables))))
        return (z3.Exists if term.op == "exists" else z3.ForAll)(variables, body)
    args = [to_z3(a, context, bindings) for a in term.args]
    operations = {"and": z3.And, "or": z3.Or, "not": z3.Not, "implies": z3.Implies,
                  "eq": lambda a, b: a == b}
    if term.op in operations:
        return operations[term.op](*args)
    if term.op == "ON":
        a, b = args
        mid = z3.FreshConst(context.BoxSort, "direct_middle")
        return z3.And(a != b, context.ON_star(a, b),
                      z3.ForAll([mid], z3.Implies(z3.And(context.ON_star(a, mid), context.ON_star(mid, b)), z3.Or(mid == a, mid == b))))
    return getattr(context, term.op)(*args)
