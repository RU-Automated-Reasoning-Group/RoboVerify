import itertools
from typing import Any, Callable, Dict, List, Set, Tuple

import z3


def z3_to_python_expr(expr: z3.ExprRef) -> Any:
    # ---------- Quantifiers ----------
    if isinstance(expr, z3.QuantifierRef):
        if expr.is_forall():
            op = "ForAll"
        elif expr.is_exists():
            op = "Exists"
        else:
            raise ValueError(f"Unknown quantifier: {expr}")

        vars = [expr.var_name(i) for i in range(expr.num_vars())]

        return {"op": op, "vars": vars, "body": z3_to_python_expr(expr.body())}

    # ---------- Bound variable (inside quantifiers) ----------
    if z3.is_var(expr):
        # Z3 bound vars are De Bruijn indexed; we'll name them by the var_name for simplicity
        # var_name(i) works only in Quantifier body, already handled
        return {"op": "Var", "name": str(expr)}

    # ---------- Literals ----------
    if z3.is_true(expr):
        return True
    if z3.is_false(expr):
        return False
    if z3.is_int_value(expr):
        return {"op": "Const", "value": expr.as_long()}

    # ---------- Variable (uninterpreted constant) ----------
    if (
        z3.is_const(expr)
        and expr.num_args() == 0
        and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED
    ):
        return {"op": "Var", "name": expr.decl().name()}

    # ---------- Function call ----------
    if (
        z3.is_app(expr)
        and expr.num_args() > 0
        and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED
    ):
        return {
            "op": "Func",
            "name": expr.decl().name(),
            "args": [z3_to_python_expr(expr.arg(i)) for i in range(expr.num_args())],
        }

    # ---------- Other applications (Boolean logic / comparisons) ----------
    if not z3.is_app(expr):
        raise ValueError(f"Unsupported expression: {expr}")

    decl = expr.decl()
    kind = decl.kind()
    args = [z3_to_python_expr(a) for a in expr.children()]

    # Boolean logic
    if kind == z3.Z3_OP_AND:
        return {"op": "And", "args": args}
    if kind == z3.Z3_OP_OR:
        return {"op": "Or", "args": args}
    if kind == z3.Z3_OP_NOT:
        return {"op": "Not", "args": args}

    # Comparisons
    if kind == z3.Z3_OP_EQ:
        return {"op": "Eq", "args": args}
    if kind == z3.Z3_OP_LE:
        return {"op": "Le", "args": args}
    if kind == z3.Z3_OP_LT:
        return {"op": "Lt", "args": args}
    if kind == z3.Z3_OP_GE:
        return {"op": "Ge", "args": args}
    if kind == z3.Z3_OP_GT:
        return {"op": "Gt", "args": args}

    # Arithmetic
    if kind == z3.Z3_OP_ADD:
        return {"op": "Add", "args": args}
    if kind == z3.Z3_OP_SUB:
        return {"op": "Sub", "args": args}
    if kind == z3.Z3_OP_MUL:
        return {"op": "Mul", "args": args}
    if kind == z3.Z3_OP_DIV:
        return {"op": "Div", "args": args}

    raise ValueError(f"Unsupported Z3 operation: {decl.name()}")


# --------------------------
# Eval Quantified Expression
# --------------------------
def eval_quantified_expr(
    expr: Any,
    env: Dict[str, Any],
    domain: List,
    function_impls: Dict[str, Callable],
    bound_vars: list = None,
) -> bool:
    if bound_vars is None:
        bound_vars = []

    if expr is True or expr is False:
        return expr

    op = expr.get("op") if isinstance(expr, dict) and "op" in expr else None

    # ---------- Bound Variable ----------
    if op == "Var" and expr["name"].startswith("Var("):
        idx = int(expr["name"][4:-1])
        var_name = bound_vars[-(idx + 1)]
        return env[var_name]

    # ---------- Named Variable ----------
    if op == "Var":
        return env[expr["name"]]

    # ---------- Constant ----------
    if op == "Const":
        return expr["value"]

    # ---------- Function ----------
    if op == "Func":
        args = [
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        ]
        if expr["name"] == "Top":
            return function_impls[expr["name"]](*args, domain)
        return function_impls[expr["name"]](*args)

    # ---------- Boolean ----------
    if op == "And":
        return all(
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        )
    if op == "Or":
        return any(
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        )
    if op == "Not":
        return not eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        )

    # ---------- Comparisons ----------
    if op == "Eq":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) == eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Gt":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) > eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Lt":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) < eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Ge":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) >= eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Le":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) <= eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )

    # ---------- Arithmetic ----------
    if op == "Add":
        return sum(
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        )
    if op == "Sub":
        a0 = eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        )
        a1 = eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
        return a0 - a1
    if op == "Mul":
        res = 1
        for a in expr["args"]:
            res *= eval_quantified_expr(a, env, domain, function_impls, bound_vars)
        return res
    if op == "Div":
        a0 = eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        )
        a1 = eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
        return a0 // a1

    # ---------- Quantifiers ----------
    if op == "ForAll":
        x_vars = expr["vars"]
        x_domains = [domain for _ in x_vars]
        for x_vals in itertools.product(*x_domains):
            new_env = env.copy()
            for i, v in enumerate(x_vars):
                new_env[v] = x_vals[i]
            if not eval_quantified_expr(
                expr["body"], new_env, domain, function_impls, bound_vars + x_vars
            ):
                return False
        return True

    if op == "Exists":
        y_vars = expr["vars"]
        y_domains = [domain for _ in y_vars]
        for y_vals in itertools.product(*y_domains):
            new_env = env.copy()
            for i, v in enumerate(y_vars):
                new_env[v] = y_vals[i]
            if eval_quantified_expr(
                expr["body"], new_env, domain, function_impls, bound_vars + y_vars
            ):
                return True
        return False

    raise ValueError(f"Unknown op: {op}")


# --------------------------
# Witness Map
# --------------------------
def compute_witness_map(
    expr: Any,
    base_env: Dict[str, Any],
    domain: List,
    function_impls: Dict[str, Callable],
) -> Dict[Tuple, Set[Tuple]]:
    """
    Compute witness map for a quantified expression.

    Args:
        expr: Python AST of the expression (ForAll/Exists or Exists)
        base_env: dictionary of free variables with their current values
        domain: list represeting all values in the domain
        function_impls: mapping of function names to Python callables

    Returns:
        witness_map: dict mapping universal assignments (tuple) to set of witness tuples
    """
    if base_env is None:
        base_env = {}

    # Determine universal and existential variables
    if expr.get("op") == "ForAll":
        x_vars = expr["vars"]
        inner = expr["body"]
    else:
        x_vars = []
        inner = expr

    assert (
        inner.get("op") == "Exists"
    ), "compute_witness_map currently expects Exists or ForAll→Exists"
    y_vars = inner["vars"]
    body = inner["body"]

    # Prepare domain lists for bound variables
    x_domains = [domain for v in x_vars] if x_vars else [()]

    witness_map: Dict[Tuple, Set[Tuple]] = {}

    # Enumerate all universal assignments
    for x_vals in itertools.product(*x_domains):
        # Start from base environment (free variables)
        env = base_env.copy()
        # Assign universal variables
        for i, v in enumerate(x_vars):
            env[v] = x_vals[i]

        witnesses: Set[Tuple] = set()

        # Enumerate all possible candidates for existential variables
        for y_vals in itertools.product(*(domain for v in y_vars)):
            for i, v in enumerate(y_vars):
                env[v] = y_vals[i]

            # Evaluate the body with current env
            if eval_quantified_expr(
                body, env, domain, function_impls, bound_vars=x_vars + y_vars
            ):
                witnesses.add(tuple(y_vals))

        witness_map[x_vals] = witnesses

    return witness_map


# --------------------------
# Merge helpers
# --------------------------
def can_merge(w1: Dict[Tuple, Set[Tuple]], w2: Dict[Tuple, Set[Tuple]]) -> bool:
    # Check that both have the same set of keys
    if set(w1.keys()) != set(w2.keys()):
        return False

    # Check that for each key, there is at least one overlapping witness
    for x in w1:
        if len(w1[x].intersection(w2[x])) == 0:
            return False
    return True


def merge_bodies(e1: Any, e2: Any) -> Any:
    """Merge two ASTs after checking they can be merged"""
    if e1.get("op") == "ForAll":
        x_vars = e1["vars"]
        y_vars = e1["body"]["vars"]
        b1 = e1["body"]["body"]
        b2 = e2["body"]["body"]
        return {
            "op": "ForAll",
            "vars": x_vars,
            "body": {
                "op": "Exists",
                "vars": y_vars,
                "body": {"op": "And", "args": [b1, b2]},
            },
        }
    else:
        y_vars = e1["vars"]
        b1 = e1["body"]
        b2 = e2["body"]
        return {"op": "Exists", "vars": y_vars, "body": {"op": "And", "args": [b1, b2]}}


# --------------------------
# Merge one round
# --------------------------
def merge_once_multi_env(
    exprs: List[Any],
    domain: List,
    function_impls: Dict[str, Callable],
    base_envs: List[Dict[str, Any]],
) -> List[Any]:
    """
    Attempt one round of merging expressions, considering multiple base_envs.
    Merge only if merge succeeds for all base_envs.
    """
    merged = [False] * len(exprs)
    new_exprs = []

    i = 0
    while i < len(exprs):
        if merged[i]:
            i += 1
            continue

        e1 = exprs[i]
        merged_flag = False

        for j in range(i + 1, len(exprs)):
            if merged[j]:
                continue
            e2 = exprs[j]

            # Check merge across all base_envs
            can_merge_all_envs = True
            for base_env in base_envs:
                w1 = compute_witness_map(e1, base_env, domain, function_impls)
                w2 = compute_witness_map(e2, base_env, domain, function_impls)

                if not can_merge(w1, w2):
                    can_merge_all_envs = False
                    break

            if can_merge_all_envs:
                # Merge expressions
                merged_expr = merge_bodies(e1, e2)
                new_exprs.append(merged_expr)
                merged[i] = True
                merged[j] = True
                merged_flag = True
                break  # merge only one pair at a time

        if not merged_flag:
            new_exprs.append(e1)
            merged[i] = True

        i += 1

    return new_exprs


# --------------------------
# Fixpoint merge
# --------------------------
def fixpoint_merge_multi_env(
    exprs: List[Any],
    domain: List,
    function_impls: Dict[str, Callable],
    base_envs: List[Dict[str, Any]],
) -> List[Any]:
    """
    Repeatedly merge expressions until no more merges can happen.
    """
    prev_len = -1
    current_exprs = exprs
    while prev_len != len(current_exprs):
        prev_len = len(current_exprs)
        current_exprs = merge_once_multi_env(
            current_exprs, domain, function_impls, base_envs
        )
    return current_exprs


def python_expr_to_z3(
    expr: Any, var_map: Dict[str, z3.ExprRef], func_map: Dict[str, z3.FuncDeclRef]
) -> z3.ExprRef:
    """
    Rebuild a Z3 expression from Python AST.

    Args:
        expr: Python AST node
        var_map: dictionary mapping variable names to Z3 variables
        func_map: dictionary mapping function names to Z3 functions

    Returns:
        z3.ExprRef representing the expression
    """

    if expr is True:
        return z3.BoolVal(True)
    if expr is False:
        return z3.BoolVal(False)

    op = expr.get("op") if isinstance(expr, dict) and "op" in expr else None

    # ---------- Variable ----------
    if op == "Var":
        name = expr["name"]
        if name.startswith("Var("):
            # Should not appear here; use var_map
            raise ValueError(f"Unexpected De Bruijn var in AST: {name}")
        return var_map[name]

    # ---------- Constant ----------
    if op == "Const":
        return z3.IntVal(expr["value"])

    # ---------- Function ----------
    if op == "Func":
        args = [python_expr_to_z3(a, var_map, func_map) for a in expr["args"]]
        return func_map[expr["name"]](*args)

    # ---------- Boolean ----------
    if op == "And":
        return z3.And([python_expr_to_z3(a, var_map, func_map) for a in expr["args"]])
    if op == "Or":
        return z3.Or([python_expr_to_z3(a, var_map, func_map) for a in expr["args"]])
    if op == "Not":
        return z3.Not(python_expr_to_z3(expr["args"][0], var_map, func_map))

    # ---------- Comparisons ----------
    if op == "Eq":
        return python_expr_to_z3(
            expr["args"][0], var_map, func_map
        ) == python_expr_to_z3(expr["args"][1], var_map, func_map)
    if op == "Gt":
        return python_expr_to_z3(
            expr["args"][0], var_map, func_map
        ) > python_expr_to_z3(expr["args"][1], var_map, func_map)
    if op == "Lt":
        return python_expr_to_z3(
            expr["args"][0], var_map, func_map
        ) < python_expr_to_z3(expr["args"][1], var_map, func_map)
    if op == "Ge":
        return python_expr_to_z3(
            expr["args"][0], var_map, func_map
        ) >= python_expr_to_z3(expr["args"][1], var_map, func_map)
    if op == "Le":
        return python_expr_to_z3(
            expr["args"][0], var_map, func_map
        ) <= python_expr_to_z3(expr["args"][1], var_map, func_map)

    # ---------- Arithmetic ----------
    if op == "Add":
        return sum([python_expr_to_z3(a, var_map, func_map) for a in expr["args"]])
    if op == "Sub":
        a0 = python_expr_to_z3(expr["args"][0], var_map, func_map)
        a1 = python_expr_to_z3(expr["args"][1], var_map, func_map)
        return a0 - a1
    if op == "Mul":
        res = python_expr_to_z3(expr["args"][0], var_map, func_map)
        for a in expr["args"][1:]:
            res *= python_expr_to_z3(a, var_map, func_map)
        return res
    if op == "Div":
        a0 = python_expr_to_z3(expr["args"][0], var_map, func_map)
        a1 = python_expr_to_z3(expr["args"][1], var_map, func_map)
        return a0 / a1

    # ---------- Quantifiers ----------
    if op == "ForAll":
        x_vars = expr["vars"]
        body = expr["body"]
        # Create Z3 bound variables
        z3_vars = [z3.Int(v) for v in x_vars]
        # Update variable map for body
        new_var_map = var_map.copy()
        for v, z in zip(x_vars, z3_vars):
            new_var_map[v] = z
        return z3.ForAll(z3_vars, python_expr_to_z3(body, new_var_map, func_map))

    if op == "Exists":
        y_vars = expr["vars"]
        body = expr["body"]
        z3_vars = [z3.Int(v) for v in y_vars]
        new_var_map = var_map.copy()
        for v, z in zip(y_vars, z3_vars):
            new_var_map[v] = z
        return z3.Exists(z3_vars, python_expr_to_z3(body, new_var_map, func_map))

    raise ValueError(f"Unknown op: {op}")
