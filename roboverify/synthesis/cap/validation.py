"""Deterministic syntax, type, value-provenance, and effect checking.

This accepts a deliberately specified Python subset, not arbitrary Python.
Missing helpers are obligations during local checking, never implicit validity.
"""

import ast
import math
from dataclasses import dataclass

from .contracts import (
    ACTIONS,
    API_NAMES,
    PREDICATES,
    QUERIES,
    Bundle,
    ConstraintFailure,
    Diagnostic,
)


@dataclass(frozen=True)
class Value:
    kind: str
    actions: bool = False


UNKNOWN = Value("unknown")
BOOL = Value("bool")
OBJECT = Value("object")
NUMBER = Value("number")
NONE = Value("none")


@dataclass(frozen=True)
class MissingCall:
    name: str
    signature: str
    role: str


FORBIDDEN = {
    "get_obj_pos",
    "get_bbox",
    "get_observation",
    "get_obs",
    "env",
    "sim",
    "observation",
    "desired_goal",
    "reward",
    "is_success",
    "eval",
    "exec",
    "open",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "vars",
    "type",
    "compile",
    "input",
}
ALLOWED = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.Expr,
    ast.If,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Call,
    ast.Lambda,
    ast.List,
    ast.Tuple,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
)


def diagnostic(node, rule, message, unit="policy"):
    return Diagnostic(
        rule, message, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), unit
    )


def parse(source, unit="policy"):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ConstraintFailure(
            [Diagnostic("SYNTAX", exc.msg, exc.lineno or 0, exc.offset or 0, unit)]
        ) from exc
    if len(list(ast.walk(tree))) > 10000:
        raise ConstraintFailure(
            [Diagnostic("SOURCE_SIZE", "Source exceeds 10000 AST nodes", unit=unit)]
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (
            node.id.startswith("_") or node.id in FORBIDDEN
        ):
            raise ConstraintFailure(
                [diagnostic(node, "FORBIDDEN_ACCESS", node.id, unit)]
            )
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            location = node
            while not hasattr(location, "lineno") and location in parents:
                location = parents[location]
            raise ConstraintFailure(
                [diagnostic(location, "UNSUPPORTED_SYNTAX", type(node).__name__, unit)]
            )
        if isinstance(node, ast.Name) and (
            node.id.startswith("_") or node.id in FORBIDDEN
        ):
            raise ConstraintFailure(
                [diagnostic(node, "FORBIDDEN_ACCESS", node.id, unit)]
            )
        if isinstance(node, (ast.FunctionDef, ast.arg)):
            name = node.name if isinstance(node, ast.FunctionDef) else node.arg
            if name in API_NAMES or name in FORBIDDEN or name.startswith("_"):
                raise ConstraintFailure(
                    [diagnostic(node, "PROTECTED_BINDING", name, unit)]
                )
        if isinstance(node, ast.FunctionDef):
            if (
                not isinstance(parents.get(node), ast.Module)
                or node.decorator_list
                or node.returns
            ):
                raise ConstraintFailure(
                    [
                        diagnostic(
                            node,
                            "FUNCTION_SHAPE",
                            "Only plain top-level function definitions are supported",
                            unit,
                        )
                    ]
                )
        if isinstance(node, ast.arguments) and (
            node.defaults
            or node.kw_defaults
            or node.vararg
            or node.kwarg
            or node.kwonlyargs
            or node.posonlyargs
        ):
            raise ConstraintFailure(
                [
                    diagnostic(
                        node,
                        "SIGNATURE",
                        "Use positional parameters without defaults",
                        unit,
                    )
                ]
            )
        if isinstance(node, ast.arg) and node.annotation:
            raise ConstraintFailure(
                [diagnostic(node, "SIGNATURE", "Annotations are not supported", unit)]
            )
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.keywords
        ):
            raise ConstraintFailure(
                [
                    diagnostic(
                        node,
                        "DYNAMIC_CALL",
                        "Use direct calls with positional arguments",
                        unit,
                    )
                ]
            )
        if isinstance(node, ast.Lambda):
            parent = parents.get(node)
            if not (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id in QUERIES
            ):
                raise ConstraintFailure(
                    [
                        diagnostic(
                            node,
                            "LAMBDA_CONTEXT",
                            "Lambdas are only query arguments",
                            unit,
                        )
                    ]
                )
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ConstraintFailure(
                    [diagnostic(node, "ASSIGNMENT", "Assign to one local name", unit)]
                )
            if node.targets[0].id in API_NAMES:
                raise ConstraintFailure(
                    [diagnostic(node, "PROTECTED_BINDING", node.targets[0].id, unit)]
                )
    return tree


def definitions(bundle):
    functions = {}
    trees = {}
    for unit, source in [("policy", bundle.source), *bundle.helpers.items()]:
        tree = parse(source, unit)
        trees[unit] = tree
        if unit != "policy" and (
            len(tree.body) != 1
            or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != unit
        ):
            raise ConstraintFailure(
                [
                    Diagnostic(
                        "HELPER_DEFINITION",
                        f"Expected exactly def {unit}(...)",
                        unit=unit,
                    )
                ]
            )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in functions:
                    raise ConstraintFailure(
                        [diagnostic(node, "DUPLICATE_HELPER", node.name, unit)]
                    )
                functions[node.name] = (node, unit)
    return trees, functions


class Checker:
    def __init__(self, bundle, *, complete=True, role="body", inputs=None):
        self.bundle, self.complete, self.role = bundle, complete, role
        self.trees, self.functions = definitions(bundle)
        self.inputs = inputs or {"b0": OBJECT}
        self.globals = dict(self.inputs)
        self.stack = []
        self.unit = "policy"
        self.missing = {}

    def fail(self, node, rule, message):
        raise ConstraintFailure([diagnostic(node, rule, message, self.unit)])

    def expect(self, node, value, kind):
        if value.kind != kind and not (value.kind == "unknown" and not self.complete):
            self.fail(node, "TYPE", f"Expected {kind}, got {value.kind}")

    def condition(self, node, env):
        value = self.expr(node, env, "condition")
        self.expect(node, value, "bool")
        if value.actions:
            self.fail(
                node,
                "ACTION_IN_CONDITION",
                "Condition depends on an action-producing helper",
            )
        return value

    def expr(self, node, env, role="body"):
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return BOOL
            if type(node.value) in (int, float) and math.isfinite(node.value):
                return NUMBER
            if node.value is None:
                return NONE
            self.fail(
                node,
                "CONSTANT",
                "Only finite numeric, Boolean, and None constants are supported",
            )
        if isinstance(node, ast.Name):
            if node.id in self.functions or node.id in API_NAMES:
                self.fail(
                    node, "FUNCTION_ALIAS", "Functions may only be used in direct calls"
                )
            if node.id not in env:
                self.fail(node, "UNBOUND_NAME", node.id)
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) != 3:
                self.fail(
                    node,
                    "VECTOR",
                    "Only three-element action-offset vectors are supported",
                )
            values = [self.expr(n, env) for n in node.elts]
            for n, value in zip(node.elts, values):
                self.expect(n, value, "number")
            return Value("vector", any(v.actions for v in values))
        if isinstance(node, ast.UnaryOp):
            value = (
                self.condition(node.operand, env)
                if isinstance(node.op, ast.Not)
                else self.expr(node.operand, env)
            )
            self.expect(
                node, value, "bool" if isinstance(node.op, ast.Not) else "number"
            )
            return value
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left, env), self.expr(node.right, env)
            self.expect(node.left, a, "number")
            self.expect(node.right, b, "number")
            return Value("number", a.actions or b.actions)
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                self.condition(item, env)
            return BOOL
        if isinstance(node, ast.Compare):
            for item in [node.left, *node.comparators]:
                value = self.expr(item, env, "condition")
                if value.kind != "object" and not (
                    value.kind == "unknown" and not self.complete
                ):
                    self.fail(
                        item,
                        "IDENTITY_COMPARISON",
                        "== and != compare object handles only; combine Boolean conditions with and/or/not",
                    )
                if value.actions:
                    self.fail(
                        item,
                        "ACTION_IN_CONDITION",
                        "Identity operand depends on an action",
                    )
            return BOOL
        if not isinstance(node, ast.Call):
            self.fail(node, "EXPRESSION", type(node).__name__)
        name = node.func.id
        if name in QUERIES:
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Lambda):
                self.fail(node, "QUERY", "Queries require one lambda expression")
            query = node.args[0]
            params = [a.arg for a in query.args.args]
            if not params or (name == "get" and len(params) != 1):
                self.fail(
                    node, "QUERY", "get binds one object; quantifiers bind one or more"
                )
            self.condition(query.body, env | dict.fromkeys(params, OBJECT))
            return OBJECT if name == "get" else BOOL
        args = [self.expr(arg, env, role) for arg in node.args]
        if name in API_NAMES:
            expected = (
                ["object", "object"]
                if name in PREDICATES
                else (
                    ["object"]
                    if name == "Pick"
                    else (
                        ["object", "number"]
                        if name == "Release"
                        else ["object", "object", "object", "vector"]
                    )
                )
            )
            if len(args) != len(expected):
                self.fail(node, "ARITY", f"{name} expects {len(expected)} arguments")
            for arg, value, kind in zip(node.args, args, expected):
                self.expect(arg, value, kind)
            acting = name in ACTIONS or any(a.actions for a in args)
            if role == "condition" and acting:
                self.fail(node, "ACTION_IN_CONDITION", name)
            return Value("none" if name in ACTIONS else "bool", acting)
        if name in env:
            self.fail(node, "DYNAMIC_CALL", f"{name} is a value, not a function")
        if name not in self.functions:
            if self.complete:
                self.fail(node, "UNRESOLVED_HELPER", name)
            previous = self.missing.get(name)
            required = (
                "condition"
                if role == "condition" or (previous and previous.role == "condition")
                else "body"
            )
            self.missing[name] = MissingCall(name, ast.unparse(node), required)
            return UNKNOWN
        if name in self.stack:
            self.fail(node, "DEPENDENCY_CYCLE", " -> ".join(self.stack + [name]))
        fn, unit = self.functions[name]
        if len(args) != len(fn.args.args):
            self.fail(node, "ARITY", f"{name} expects {len(fn.args.args)} arguments")
        old_unit = self.unit
        self.unit = unit
        self.stack.append(name)
        try:
            local = self.globals | dict(zip((a.arg for a in fn.args.args), args))
            _, returns, acting, terminated = self.block(
                fn.body, local, role, in_function=True
            )
            if not terminated:
                returns.append(NONE)
            result = merge_values(returns)
            result = Value(result.kind, result.actions or acting)
            if role == "condition" and result.actions:
                self.fail(node, "ACTION_IN_CONDITION", name)
            return result
        finally:
            self.stack.pop()
            self.unit = old_unit

    def block(self, statements, env, role, *, in_function=False, loop=0):
        env, returns, acting = dict(env), [], False
        for stmt in statements:
            if isinstance(stmt, ast.FunctionDef):
                continue
            if isinstance(stmt, ast.Assign):
                name = stmt.targets[0].id
                if name in self.functions:
                    self.fail(stmt, "PROTECTED_BINDING", name)
                value = self.expr(stmt.value, env, role)
                env[name] = value
                acting |= value.actions
            elif isinstance(stmt, ast.Expr):
                acting |= self.expr(stmt.value, env, role).actions
            elif isinstance(stmt, ast.Return):
                if not in_function:
                    self.fail(stmt, "RETURN", "Top-level return is not supported")
                value = self.expr(stmt.value, env, role) if stmt.value else NONE
                returns.append(value)
                return env, returns, acting or value.actions, True
            elif isinstance(stmt, (ast.If, ast.While)):
                self.condition(stmt.test, env)
                left, lr, la, lt = self.block(
                    stmt.body,
                    env,
                    role,
                    in_function=in_function,
                    loop=loop + isinstance(stmt, ast.While),
                )
                right, rr, ra, rt = self.block(
                    stmt.orelse, env, role, in_function=in_function, loop=loop
                )
                # A while body may run zero times. The type must remain valid on re-entry.
                if isinstance(stmt, ast.While):
                    merged = merge_env(env, left)
                    self.condition(stmt.test, merged)
                    self.block(
                        stmt.body, merged, role, in_function=in_function, loop=loop + 1
                    )
                    env = merge_env(right, merged)
                else:
                    env = (
                        right
                        if lt and not rt
                        else left if rt and not lt else merge_env(left, right)
                    )
                returns.extend(lr + rr)
                acting |= la or ra
                if isinstance(stmt, ast.If) and lt and rt:
                    return env, returns, acting, True
            elif isinstance(stmt, (ast.Break, ast.Continue)):
                if not loop:
                    self.fail(stmt, "LOOP_CONTROL", "break/continue requires a loop")
                # Remaining code is still checked conservatively.
            elif not isinstance(stmt, ast.Pass):
                self.fail(stmt, "STATEMENT", type(stmt).__name__)
            if not in_function:
                self.globals = dict(env)
        return env, returns, acting, False

    def check(self):
        self.block(self.trees["policy"].body, self.inputs, self.role)
        # Uncalled definitions still need valid bodies and API signatures.
        # Their parameter types remain generic until a caller specializes them.
        original_complete, original_unit = self.complete, self.unit
        try:
            self.complete = False
            for name, (fn, unit) in self.functions.items():
                self.unit = unit
                self.stack = [name]
                local = self.globals | dict.fromkeys(
                    (a.arg for a in fn.args.args), UNKNOWN
                )
                self.block(fn.body, local, "body", in_function=True)
        finally:
            self.complete, self.unit, self.stack = original_complete, original_unit, []
        if original_complete and self.missing:
            raise ConstraintFailure(
                [Diagnostic("UNRESOLVED_HELPER", name) for name in self.missing]
            )
        return list(self.missing.values())


def merge_values(values):
    if not values:
        return NONE
    kinds = {v.kind for v in values}
    return Value(
        next(iter(kinds)) if len(kinds) == 1 else "mixed",
        any(v.actions for v in values),
    )


def merge_env(a, b):
    return {name: merge_values([a[name], b[name]]) for name in a.keys() & b.keys()}


def validate(bundle, *, complete=True, role="body", inputs=None):
    return Checker(bundle, complete=complete, role=role, inputs=inputs).check()


def validate_helper(source, name, known, role):
    """Check a helper locally before its caller supplies concrete argument types."""
    checker = Checker(Bundle("", known | {name: source}), complete=False, role=role)
    fn, checker.unit = checker.functions[name]
    checker.stack = [name]
    checker.block(
        fn.body,
        checker.inputs | dict.fromkeys((a.arg for a in fn.args.args), UNKNOWN),
        role,
        in_function=True,
    )


def missing_calls(source, helpers=None, role="body"):
    """Syntactic discovery includes uncalled definitions and indirect guard uses."""
    trees, functions = definitions(Bundle(source, helpers or {}))
    calls, assignments, conditions = {}, {}, set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assignments[node.targets[0].id] = node.value
            if isinstance(node, (ast.If, ast.While)):
                conditions.update(
                    n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
                )
            if (
                isinstance(node, ast.Call)
                and node.func.id not in API_NAMES
                and node.func.id not in functions
            ):
                calls[node.func.id] = node
    changed = True
    while changed:
        before = set(conditions)
        for name in list(conditions):
            if name in assignments:
                conditions.update(
                    n.id for n in ast.walk(assignments[name]) if isinstance(n, ast.Name)
                )
        changed = before != conditions
    return [
        MissingCall(
            name,
            ast.unparse(node),
            "condition" if role == "condition" or name in conditions else "body",
        )
        for name, node in calls.items()
    ]
