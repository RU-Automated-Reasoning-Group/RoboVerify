"""Deterministic syntax, type, value-provenance, and effect checking.

This accepts a deliberately specified Python subset, not arbitrary Python.
Missing helpers are obligations during local checking, never implicit validity.
"""

import ast
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .contracts import (
    ACTIONS,
    API_NAMES,
    GOAL_API_NAMES,
    GOAL_POSITION_QUERIES,
    GOAL_PREDICATES,
    POSITION_QUERIES,
    PREDICATES,
    PROTECTED_INPUTS,
    QUERIES,
    TRAVERSALS,
    Bundle,
    CapError,
    ConstraintFailure,
    Diagnostic,
    validate_condition_mode,
)

TYPE_KINDS = frozenset(
    {"object", "goal", "number", "bool", "position", "vector", "none"}
)
MAX_ANALYSIS_STEPS = 100000
# A scoped override of MAX_ANALYSIS_STEPS (RunConfig.max_analysis_steps); None
# keeps the module default.
_ANALYSIS_STEP_LIMIT = ContextVar("analysis_step_limit", default=None)


@contextmanager
def analysis_step_limit(steps):
    """Run DSL analyses in this scope with a work budget of ``steps``.

    ``None`` keeps ``MAX_ANALYSIS_STEPS``. The budget bounds checker work only;
    the verdict of an analysis that finishes does not depend on it.
    """
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("The analysis step limit must be a positive integer")
    token = _ANALYSIS_STEP_LIMIT.set(steps)
    try:
        yield
    finally:
        _ANALYSIS_STEP_LIMIT.reset(token)


def current_analysis_step_limit():
    steps = _ANALYSIS_STEP_LIMIT.get()
    return MAX_ANALYSIS_STEPS if steps is None else steps


class AnalysisLimit(CapError):
    """An unfinished analysis is not a source violation or execution approval."""


@dataclass(frozen=True, init=False)
class Value:
    # Strings are concrete kinds; integers identify constrained generic parameters.
    types: frozenset
    actions: bool

    def __init__(self, kind, actions=False, *, types=None):
        object.__setattr__(self, "types", frozenset({kind} if types is None else types))
        object.__setattr__(self, "actions", actions)

    @property
    def kind(self):
        if len(self.types) == 1:
            item = next(iter(self.types))
            return item if isinstance(item, str) else "unknown"
        return "mixed"

    def with_actions(self, actions):
        return Value(self.kind, self.actions or actions, types=self.types)


@dataclass
class Flow:
    normal: dict | None
    returns: list = field(default_factory=list)
    breaks: list = field(default_factory=list)
    continues: list = field(default_factory=list)
    actions: bool = False


UNKNOWN = Value("unknown")
BOOL = Value("bool")
OBJECT = Value("object")
GOAL = Value("goal")
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
    ast.Subscript,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)


def diagnostic(node, rule, message, unit="policy"):
    return Diagnostic(
        rule, message, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), unit
    )


def parse(source, unit="policy", protected=API_NAMES):
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
            if name in protected or name in FORBIDDEN or name.startswith("_"):
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
        if isinstance(node, ast.Subscript) and not (
            isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
            and node.slice.value in (0, 1, 2)
        ):
            raise ConstraintFailure(
                [
                    diagnostic(
                        node,
                        "POSITION_INDEX",
                        "Use literal position index 0, 1 or 2",
                        unit,
                    )
                ]
            )
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ConstraintFailure(
                    [diagnostic(node, "ASSIGNMENT", "Assign to one local name", unit)]
                )
            if node.targets[0].id in protected:
                raise ConstraintFailure(
                    [diagnostic(node, "PROTECTED_BINDING", node.targets[0].id, unit)]
                )
    return tree


def protected_names(api_names, inputs=()):
    """API names and fixed goal inputs a policy may not rebind.

    Goal names are protected only where the task exposes them, so tower tasks
    accept exactly the programs they accepted before the goal interface.
    """
    available = set(api_names) & GOAL_API_NAMES
    return API_NAMES | available | (PROTECTED_INPUTS & set(inputs))


def definitions(bundle, protected=API_NAMES):
    functions = {}
    trees = {}
    for unit, source in [("policy", bundle.source), *bundle.helpers.items()]:
        tree = parse(source, unit, protected)
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
    def __init__(
        self,
        bundle,
        *,
        complete=True,
        role="body",
        inputs=None,
        condition_mode="dsl_only",
        api_names=API_NAMES,
    ):
        self.condition_mode = validate_condition_mode(condition_mode)
        if (
            bundle.condition_mode is not None
            and bundle.condition_mode != condition_mode
        ):
            raise ConstraintFailure(
                [Diagnostic("CONDITION_MODE", "Bundle and checker modes differ")]
            )
        self.bundle, self.complete, self.role = bundle, complete, role
        self.inputs = inputs or {"b0": OBJECT}
        self.api_set = frozenset(api_names)
        # Goal APIs are recognised only where the task exposes them; elsewhere a
        # call such as AtGoal(x) remains an ordinary (unresolved) helper call.
        self.known_api = API_NAMES | (self.api_set & GOAL_API_NAMES)
        self.handle_kinds = (
            frozenset({"object", "goal"})
            if self.api_set & GOAL_API_NAMES
            else frozenset({"object"})
        )
        self.trees, self.functions = definitions(
            bundle, protected_names(api_names, self.inputs)
        )
        for unit, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and node.func.id in API_NAMES - set(
                    api_names
                ):
                    raise ConstraintFailure(
                        [diagnostic(node, "API_UNAVAILABLE", node.func.id, unit)]
                    )
        self.globals = dict(self.inputs)
        self.stack = []
        self.unit = "policy"
        self.missing = {}
        self.analysis_steps = 0
        self.max_analysis_steps = current_analysis_step_limit()
        self.type_parents = {}
        self.type_domains = {}
        self.type_revision = 0

    def fail(self, node, rule, message):
        raise ConstraintFailure([diagnostic(node, rule, message, self.unit)])

    def step(self):
        self.analysis_steps += 1
        if self.analysis_steps > self.max_analysis_steps:
            raise AnalysisLimit("DSL analysis work budget exhausted")

    def generic_inputs(self, names):
        result = {}
        for name in names:
            ident = len(self.type_parents)
            self.type_parents[ident] = ident
            self.type_domains[ident] = TYPE_KINDS
            result[name] = Value("unknown", types={ident})
        return result

    def type_root(self, ident):
        root = ident
        while self.type_parents[root] != root:
            root = self.type_parents[root]
        while self.type_parents[ident] != ident:
            parent = self.type_parents[ident]
            self.type_parents[ident] = root
            ident = parent
        return root

    def domain(self, item):
        if isinstance(item, int):
            return self.type_domains[self.type_root(item)]
        if item == "unknown" and not self.complete:
            return TYPE_KINDS  # An unresolved local call remains a separate obligation.
        return frozenset({item})

    def possible_kind(self, value):
        kinds = frozenset().union(*(self.domain(t) for t in value.types))
        if len(kinds) == 1:
            return next(iter(kinds))
        return "unknown" if kinds == TYPE_KINDS else "mixed"

    def constrain(self, node, values, allowed):
        # Every alternative must support the required kind. Linked parameters
        # retain this requirement through aliases, returns, comparisons and joins.
        self.step()
        common = frozenset(allowed)
        items = frozenset().union(*(v.types for v in values))
        for item in items:
            common &= self.domain(item)
        if not common:
            expected = " or ".join(sorted(allowed))
            actual = "/".join(self.possible_kind(v) for v in values)
            self.fail(node, "TYPE", f"Expected {expected}, got {actual}")
        roots = sorted({self.type_root(t) for t in items if isinstance(t, int)})
        if roots:
            root = roots[0]
            if self.type_domains[root] != common:
                self.type_domains[root] = common
                self.type_revision += 1
            for other in roots[1:]:
                self.type_parents[other] = root
                self.type_revision += 1

    def expect(self, node, value, kind):
        self.constrain(node, [value], {kind})

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
        self.step()
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
            if node.id in self.functions or node.id in self.known_api:
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
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value, env, role)
            self.expect(node.value, value, "position")
            return Value("number", value.actions)
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
            operands = [node.left, *node.comparators]
            values = [self.expr(item, env, "condition") for item in operands]
            for item, value in zip(operands, values):
                if value.actions:
                    self.fail(
                        item,
                        "ACTION_IN_CONDITION",
                        "Comparison operand depends on an action",
                    )
            for op, left, right in zip(node.ops, values, values[1:]):
                identity = isinstance(op, (ast.Eq, ast.NotEq))
                if self.condition_mode == "dsl_only":
                    if not identity or any(
                        not self.domain(t) & self.handle_kinds
                        for value in (left, right)
                        for t in value.types
                    ):
                        self.fail(
                            node,
                            "NUMERIC_CONDITION",
                            "dsl_only permits only object identity comparisons; numerical conditions require coordinate_predicates",
                        )
                    self.constrain(node, [left, right], self.handle_kinds)
                elif identity:
                    self.constrain(node, [left, right], self.handle_kinds | {"number"})
                else:
                    if any(
                        self.domain(t) in ({"object"}, {"goal"})
                        for v in (left, right)
                        for t in v.types
                    ):
                        self.fail(
                            node,
                            "IDENTITY_COMPARISON",
                            (
                                "Objects support only == and !="
                                if "goal" not in self.handle_kinds
                                else "Object and goal handles support only == and !="
                            ),
                        )
                    self.constrain(node, [left, right], {"number"})
            return BOOL
        if not isinstance(node, ast.Call):
            self.fail(node, "EXPRESSION", type(node).__name__)
        name = node.func.id
        if name in TRAVERSALS and name in self.known_api:
            self.fail(
                node,
                "TRAVERSAL_TARGET",
                f"{name}(x) is a statement on one goal variable, not a value",
            )
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
        if name in POSITION_QUERIES or (
            name in GOAL_POSITION_QUERIES and name in self.known_api
        ):
            expected = (
                []
                if name == "get_gripper_position"
                else ["goal"] if name in GOAL_POSITION_QUERIES else ["object"]
            )
            if len(args) != len(expected):
                self.fail(node, "ARITY", f"{name} expects {len(expected)} arguments")
            for arg, value, kind in zip(node.args, args, expected):
                self.expect(arg, value, kind)
            acting = any(value.actions for value in args)
            if role == "condition" and acting:
                self.fail(node, "ACTION_IN_CONDITION", name)
            return Value("position", acting)
        if name in GOAL_PREDICATES and name in self.known_api:
            expected = ["object"] if name == "AtGoal" else ["goal"]
            if len(args) != 1:
                self.fail(node, "ARITY", f"{name} expects 1 argument")
            self.expect(node.args[0], args[0], expected[0])
            acting = args[0].actions
            if role == "condition" and acting:
                self.fail(node, "ACTION_IN_CONDITION", name)
            return Value("bool", acting)
        if name in API_NAMES:
            # Move/Release references are blocks, or goals where the task has them.
            reference = "reference"
            expected = (
                ["object", "object"]
                if name in PREDICATES
                else (
                    ["object"]
                    if name == "Pick"
                    else (
                        [reference, "number"]
                        if name == "Release"
                        else [reference, reference, reference, "vector"]
                    )
                )
            )
            if len(args) != len(expected):
                self.fail(node, "ARITY", f"{name} expects {len(expected)} arguments")
            for arg, value, kind in zip(node.args, args, expected):
                if kind == reference:
                    self.constrain(arg, [value], self.handle_kinds)
                else:
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
            result = result.with_actions(acting or any(arg.actions for arg in args))
            if role == "condition" and result.actions:
                self.fail(node, "ACTION_IN_CONDITION", name)
            return result
        finally:
            self.stack.pop()
            self.unit = old_unit

    def block(self, statements, env, role, *, in_function=False, loop=0):
        flow = self.block_flow(
            statements, env, role, in_function=in_function, loop=loop
        )
        return flow.normal or dict(env), flow.returns, flow.actions, flow.normal is None

    def block_flow(self, statements, env, role, *, in_function=False, loop=0):
        flow = Flow(dict(env))
        dead_env = dict(env)
        for stmt in statements:
            self.step()
            if isinstance(stmt, ast.FunctionDef):
                continue
            reachable = flow.normal is not None
            current = dict(flow.normal if reachable else dead_env)
            if not in_function:
                self.globals = dict(current)
            step = self.statement_flow(stmt, current, role, in_function, loop)
            dead_env = step.normal if step.normal is not None else current
            if reachable:
                flow.normal = step.normal
                flow.returns.extend(step.returns)
                flow.breaks.extend(step.breaks)
                flow.continues.extend(step.continues)
                flow.actions |= step.actions
            # Unreachable source is still checked, but cannot overwrite a real
            # break/continue/return state or introduce a loop back edge.
        if not in_function and flow.normal is not None:
            self.globals = dict(flow.normal)
        return flow

    def statement_flow(self, stmt, env, role, in_function, loop):
        if isinstance(stmt, ast.Assign):
            name = stmt.targets[0].id
            if name in self.functions:
                self.fail(stmt, "PROTECTED_BINDING", name)
            value = self.expr(stmt.value, env, role)
            env[name] = value
            return Flow(env, actions=value.actions)
        if isinstance(stmt, ast.Expr):
            call = stmt.value
            if (
                isinstance(call, ast.Call)
                and call.func.id in TRAVERSALS
                and call.func.id in self.known_api
            ):
                return self.traversal_flow(stmt, call, env, role)
            value = self.expr(stmt.value, env, role)
            return Flow(env, actions=value.actions)
        if isinstance(stmt, ast.Return):
            if not in_function:
                self.fail(stmt, "RETURN", "Top-level return is not supported")
            value = self.expr(stmt.value, env, role) if stmt.value else NONE
            return Flow(None, returns=[value], actions=value.actions)
        if isinstance(stmt, ast.If):
            self.condition(stmt.test, env)
            left = self.block_flow(
                stmt.body, env, role, in_function=in_function, loop=loop
            )
            right = self.block_flow(
                stmt.orelse, env, role, in_function=in_function, loop=loop
            )
            return Flow(
                join_envs([left.normal, right.normal]),
                left.returns + right.returns,
                left.breaks + right.breaks,
                left.continues + right.continues,
                left.actions or right.actions,
            )
        if isinstance(stmt, ast.While):
            return self.loop_flow(stmt, env, role, in_function, loop)
        if isinstance(stmt, (ast.Break, ast.Continue)):
            if not loop:
                self.fail(stmt, "LOOP_CONTROL", "break/continue requires a loop")
            if isinstance(stmt, ast.Break):
                return Flow(None, breaks=[dict(env)])
            return Flow(None, continues=[dict(env)])
        if isinstance(stmt, ast.Pass):
            return Flow(env)
        self.fail(stmt, "STATEMENT", type(stmt).__name__)

    def traversal_flow(self, stmt, call, env, role):
        """MoveRight(x)/MoveDown(x) rebind one goal variable in place (x := x.r/d)."""
        name = call.func.id
        if role == "condition":
            self.fail(stmt, "ACTION_IN_CONDITION", f"{name} updates a goal cursor")
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.fail(stmt, "TRAVERSAL_TARGET", f"{name} takes one goal variable name")
        target = call.args[0].id
        if target in PROTECTED_INPUTS:
            self.fail(stmt, "TRAVERSAL_TARGET", f"{target} cannot be advanced")
        if target in self.functions or target in self.known_api:
            self.fail(stmt, "TRAVERSAL_TARGET", f"{target} is not a goal variable")
        if target not in env:
            self.fail(stmt, "UNBOUND_NAME", target)
        self.expect(call.args[0], env[target], "goal")
        env[target] = GOAL
        return Flow(env)

    def loop_flow(self, stmt, entry, role, in_function, loop):
        head = dict(entry)
        while True:
            self.step()
            before_types = self.type_revision
            if not in_function:
                self.globals = dict(head)
            self.condition(stmt.test, head)
            body = self.block_flow(
                stmt.body, head, role, in_function=in_function, loop=loop + 1
            )
            # Entry contributes the zero-iteration path. Only fallthrough and
            # continue states feed the next iteration; break/return do not.
            merged = join_envs([head, entry, body.normal, *body.continues])
            if merged == head and before_types == self.type_revision:
                break
            head = merged
        otherwise = self.block_flow(
            stmt.orelse, head, role, in_function=in_function, loop=loop
        )
        return Flow(
            join_envs([otherwise.normal, *body.breaks]),
            body.returns + otherwise.returns,
            otherwise.breaks,
            otherwise.continues,
            body.actions or otherwise.actions,
        )

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
                local = self.globals | self.generic_inputs(a.arg for a in fn.args.args)
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
    return Value(
        "mixed",
        any(v.actions for v in values),
        types=frozenset().union(*(v.types for v in values)),
    )


def merge_env(a, b):
    return {name: merge_values([a[name], b[name]]) for name in a.keys() & b.keys()}


def join_envs(environments):
    available = [env for env in environments if env is not None]
    if not available:
        return None
    result = dict(available[0])
    for env in available[1:]:
        result = merge_env(result, env)
    return result


def validate(
    bundle,
    *,
    complete=True,
    role="body",
    inputs=None,
    condition_mode="dsl_only",
    api_names=API_NAMES,
):
    return Checker(
        bundle,
        complete=complete,
        role=role,
        inputs=inputs,
        condition_mode=condition_mode,
        api_names=api_names,
    ).check()


def validate_helper(
    source,
    name,
    known,
    role,
    *,
    condition_mode="dsl_only",
    api_names=API_NAMES,
    inputs=None,
):
    """Check a helper locally before its caller supplies concrete argument types."""
    checker = Checker(
        Bundle("", known | {name: source}),
        complete=False,
        role=role,
        inputs=inputs,
        condition_mode=condition_mode,
        api_names=api_names,
    )
    fn, checker.unit = checker.functions[name]
    checker.stack = [name]
    checker.block(
        fn.body,
        checker.inputs | checker.generic_inputs(a.arg for a in fn.args.args),
        role,
        in_function=True,
    )

    return list(checker.missing.values())


def missing_calls(source, helpers=None, role="body", api_names=API_NAMES):
    """Syntactic discovery includes uncalled definitions and indirect guard uses."""
    known = API_NAMES | (set(api_names) & GOAL_API_NAMES)
    trees, functions = definitions(Bundle(source, helpers or {}), known)
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
                and node.func.id not in known
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
