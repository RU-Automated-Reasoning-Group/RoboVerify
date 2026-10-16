"""Small explicit interpreter for the checked Python subset; never uses exec."""

import ast
import math
import operator

from .contracts import (
    ACTIONS,
    BudgetExceeded,
    CapError,
    ObjectHandle,
    Position3,
    validate_condition_mode,
)
from .validation import definitions, validate


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class BreakLoop(Exception):
    pass


class ContinueLoop(Exception):
    pass


class Runtime:
    def __init__(self, adapter, budget, logger, condition_mode="dsl_only"):
        self.condition_mode = validate_condition_mode(condition_mode)
        self.adapter, self.budget, self.logger = adapter, budget, logger
        self.functions = {}
        self.global_values = {}
        self.depth = 0
        self.condition_depth = 0
        self.unit = "policy"

    def execute(self, bundle, arguments=None, *, role="body", return_name=None):
        from .validation import OBJECT

        arguments = arguments or {"b0": self.adapter.b0}
        if any(not isinstance(v, ObjectHandle) for v in arguments.values()):
            raise CapError("Module inputs must be object handles")
        validate(
            bundle,
            inputs=dict.fromkeys(arguments, OBJECT),
            role=role,
            condition_mode=self.condition_mode,
        )
        trees, self.functions = definitions(bundle)
        self.global_values = dict(arguments)
        self.condition_depth = int(role == "condition")
        self.logger.emit("execution_start", checkpoint_id=bundle.checkpoint_id)
        self.block(trees["policy"].body, self.global_values)
        result = self.global_values.get(return_name) if return_name else None
        self.logger.emit("execution_end", result=result)
        return result, dict(self.global_values)

    def numeric(self, value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise CapError("Expected a finite action parameter")
        return float(value)

    def condition(self, node, env):
        self.condition_depth += 1
        try:
            value = self.expr(node, env)
            if type(value) is not bool:
                raise CapError("Condition did not return a Boolean")
            return value
        finally:
            self.condition_depth -= 1

    def expr(self, node, env):
        self.budget.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expr(item, env) for item in node.elts]
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value, env)
            if not (
                isinstance(value, Position3)
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and node.slice.value in (0, 1, 2)
            ):
                raise CapError("Only literal Position3 component reads are supported")
            return self.numeric(value[node.slice.value])
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self.condition(node.operand, env)
            value = self.numeric(self.expr(node.operand, env))
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = self.numeric(self.expr(node.left, env)), self.numeric(
                self.expr(node.right, env)
            )
            if isinstance(node.op, ast.Add):
                result = a + b
            elif isinstance(node.op, ast.Sub):
                result = a - b
            elif isinstance(node.op, ast.Mult):
                result = a * b
            else:
                result = a / b
            return self.numeric(result)
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self.condition(item, env)
                if isinstance(node.op, ast.And) and not value:
                    return False
                if isinstance(node.op, ast.Or) and value:
                    return True
            return isinstance(node.op, ast.And)
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, right_node in zip(node.ops, node.comparators):
                right = self.expr(right_node, env)
                if isinstance(left, ObjectHandle) and isinstance(right, ObjectHandle):
                    if not isinstance(op, (ast.Eq, ast.NotEq)):
                        raise CapError("Objects support only identity comparisons")
                    equal = self.adapter.call("eq", [left, right])
                    result = equal if isinstance(op, ast.Eq) else not equal
                elif self.condition_mode == "coordinate_predicates":
                    operations = {
                        ast.Eq: operator.eq,
                        ast.NotEq: operator.ne,
                        ast.Lt: operator.lt,
                        ast.LtE: operator.le,
                        ast.Gt: operator.gt,
                        ast.GtE: operator.ge,
                    }
                    result = operations[type(op)](
                        self.numeric(left), self.numeric(right)
                    )
                else:
                    raise CapError("Numeric conditions require coordinate_predicates")
                if not result:
                    return False
                left = right
            return True
        if isinstance(node, ast.Lambda):
            names = [a.arg for a in node.args.args]

            def query(*objects):
                self.budget.tick()
                return self.condition(node.body, env | dict(zip(names, objects)))

            query.arity = len(names)
            return query
        if isinstance(node, ast.Call):
            args = [self.expr(arg, env) for arg in node.args]
            name = node.func.id
            if name not in self.functions:
                if self.condition_depth and name in ACTIONS:
                    raise CapError("Action attempted during condition evaluation")
                return self.adapter.call(name, args)
            fn, unit = self.functions[name]
            self.depth += 1
            self.budget.depth(self.depth)
            parent = self.logger.emit(
                "helper_enter", helper=name, depth=self.depth, arguments=args
            )
            previous = self.unit
            self.unit = unit
            try:
                local = self.global_values | dict(
                    zip((a.arg for a in fn.args.args), args)
                )
                with self.logger.scoped(
                    parent_event_id=parent, helper=name, depth=self.depth
                ):
                    try:
                        self.block(fn.body, local)
                        result = None
                    except Returned as returned:
                        result = returned.value
                self.logger.emit(
                    "helper_return", helper=name, result=result, parent_event_id=parent
                )
                return result
            finally:
                self.unit = previous
                self.depth -= 1
        raise CapError(f"Unsupported expression at runtime: {type(node).__name__}")

    def block(self, statements, env):
        for stmt in statements:
            self.budget.tick()
            if isinstance(stmt, ast.FunctionDef):
                continue
            with self.logger.scoped(source_unit=self.unit, source_line=stmt.lineno):
                self.logger.emit(
                    "statement_start", statement=ast.unparse(stmt).splitlines()[0]
                )
                if isinstance(stmt, ast.Assign):
                    name = stmt.targets[0].id
                    env[name] = self.expr(stmt.value, env)
                    self.logger.emit("assignment", name=name, value=env[name])
                elif isinstance(stmt, ast.Expr):
                    value = self.expr(stmt.value, env)
                    self.logger.emit("expression_result", value=value)
                elif isinstance(stmt, ast.Return):
                    raise Returned(self.expr(stmt.value, env) if stmt.value else None)
                elif isinstance(stmt, ast.If):
                    selected = self.condition(stmt.test, env)
                    self.logger.emit("branch", selected=selected)
                    self.block(stmt.body if selected else stmt.orelse, env)
                elif isinstance(stmt, ast.While):
                    iteration = 0
                    while self.condition(stmt.test, env):
                        self.budget.tick()
                        self.logger.emit("loop_iteration", iteration=iteration)
                        iteration += 1
                        try:
                            self.block(stmt.body, env)
                        except BreakLoop:
                            break
                        except ContinueLoop:
                            continue
                    else:
                        self.block(stmt.orelse, env)
                    self.logger.emit("loop_exit", iterations=iteration)
                elif isinstance(stmt, ast.Break):
                    raise BreakLoop()
                elif isinstance(stmt, ast.Continue):
                    raise ContinueLoop()
                elif not isinstance(stmt, ast.Pass):
                    raise CapError(f"Unsupported statement: {type(stmt).__name__}")
                self.logger.emit("statement_end")
