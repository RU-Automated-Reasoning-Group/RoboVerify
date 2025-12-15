import itertools
from copy import deepcopy
from typing import Dict, List, Set, Tuple

import sympy
import z3
from synthesis.decision_tree import on_star_implentation
from synthesis.verification_lib.highlevel_verification_lib import (
    BoxSort,
    ON_star,
    get_consts,
    highlevel_z3_solver,
)


def add_all_pairs(vocabulary, r, all_vars):
    for v1 in all_vars:
        for v2 in all_vars:
            if str(v1) != str(v2):
                if isinstance(r, str) and r == "equality":
                    vocabulary.append(v1 == v2)
                elif r == ON_star:
                    vocabulary.append(r(v1, v2))
                else:
                    assert False, f"unknown relation r: {r}"


def compute_omega_k(k: int, relations: List, constants: List) -> Tuple[List, List]:
    """compute the finite vocabulary Omage(k)
    k is the number of nested forall quantifiers
    relations is the list of binary relation to consider
    constants is the list of constants in the existing program
    """
    universally_quantified_vars = [get_consts(f"ux{i}") for i in range(1, k + 1)]
    all_vars = universally_quantified_vars + constants
    omega_inv = []
    for r in relations:
        add_all_pairs(omega_inv, r, all_vars)
    return omega_inv, universally_quantified_vars


def compute_dataset(
    states: List[Dict],
    omega_k: List,
    universal_quantified_vars: List,
    constants: List,
    constants_mapping: Dict,
) -> Set:
    """Compute the dataset for each state and for each relation in the vocabulary
    Each universally quantified variable needs to be bind into one of the existing blocks in the states
    """
    dataset = set()
    for state in states:
        all_objects = list(state.keys())
        for assignment in itertools.product(
            all_objects, repeat=len(universal_quantified_vars)
        ):
            var_mapping = {
                var: assignment[idx]
                for idx, var in enumerate(universal_quantified_vars)
            }
            d = compute_data(
                state,
                omega_k,
                var_mapping,
                constants_mapping,
            )
            print("var_mapping", var_mapping)
            print("data", d)
            dataset.add(d)
    return dataset


def compute_data(
    state,
    omega_k: List,
    var_mapping: Dict,
    constants_mapping: Dict,
) -> Tuple:
    data = []
    for predicate in omega_k:
        if "ON_star" in str(predicate):
            arg0, arg1 = predicate.arg(0), predicate.arg(1)
            block1_name = (
                var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            )
            block2_name = (
                var_mapping[arg1] if arg1 in var_mapping else constants_mapping[arg1]
            )
            assert isinstance(block1_name, str) and isinstance(block2_name, str)
            data.append(
                on_star_implentation(
                    state[block1_name],
                    state[block2_name],
                )
            )
        else:
            # equality
            arg0, arg1 = predicate.arg(0), predicate.arg(1)
            block1_name = (
                var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            )
            block2_name = (
                var_mapping[arg1] if arg1 in var_mapping else constants_mapping[arg1]
            )
            assert isinstance(block1_name, str) and isinstance(block2_name, str)
            data.append(block1_name == block2_name)
    return tuple(data)


def compute_S_U(dataset: Set, omega_k: List, index: int):
    S, U = set(), set()

    rest_omega_k = deepcopy(omega_k)
    target = rest_omega_k.pop(index)

    assert 0 <= index < len(omega_k)
    for d in dataset:
        assert len(d) == len(omega_k)
        val = d[index]
        if val:
            S.add(d[:index] + d[index + 1 :])
        else:
            U.add(d[:index] + d[index + 1 :])
    return S, U, target, rest_omega_k


def learn_from_partition(S: Set, U: Set):
    """Encode the constraint system of the learning program using z3 for samples in S and U
    returns the indexes selected
    """
    import pdb

    pdb.set_trace()
    n = None
    for d in S:
        n = len(d)
        break

    # sel_i ∈ {0,1}
    sel = [z3.Int(f"sel_{i}") for i in range(n)]

    opt = z3.Optimize()

    # Constraint φ2: 0 ≤ sel_i ≤ 1  (binary)
    for i in range(n):
        opt.add(sel[i] >= 0, sel[i] <= 1)

    # Constraint φ1:
    # For every pair (s, u), ∨ (s(Π_i) ≠ u(Π_i) ∧ sel_i = 1)
    for si, s_data in enumerate(S):
        for ui, u_data in enumerate(U):
            disj = []
            for i in range(n):
                if s_data[i] != u_data[i]:
                    disj.append(sel[i] == 1)
                # otherwise this predicate can't distinguish s and u
            opt.add(z3.Or(*disj))
    opt.add(sel[1] == 0)
    # Objective φc: minimize Σ_i sel_i
    opt.minimize(z3.Sum(sel))

    # Solve
    result = opt.check()
    if result == z3.sat:
        model = opt.model()
        chosen = [i for i in range(n) if model[sel[i]].as_long() == 1]
        return chosen
    else:
        return None


def construct_truth_table_and_extract_expression(
    current_S: Set, current_U: Set, num_selected: int, adding_to_S: bool = True
):
    """Construct the truth table for all possible valuations, reject those in U and accept the rest in S, then extact a minimized the logic expression for this

    Current_S contains data from S projected to selected indices
    Current_U contains data from U projected to selected indices
    adding_to_S is not used for now
    """
    assert len(current_S & current_U) == 0

    accepted_values = []
    rejected_values = []
    for assignment in itertools.product((0, 1), repeat=num_selected):
        if assignment in current_U:
            rejected_values.append(assignment)
        else:
            accepted_values.append(assignment)

    good = all([s_data in accepted_values for s_data in current_S])
    if not good:
        raise ValueError("Not all assignment from current_S are accepted")

    var_symbols = sympy.symbols(f"term0:{num_selected}")
    return sympy.SOPform(var_symbols, accepted_values), var_symbols


def project_to_selected(dataset: Set, selected: List) -> Set:
    projected_dataset = set()
    for d in dataset:
        projected_dataset.add(tuple(d[i] for i in selected))
    return projected_dataset


def sympy_to_z3(expr, z3_terms):
    """
    Convert a SymPy Boolean expression into a Z3 Boolean expression.

    SymPy symbols must be named: term0, term1, ...
    z3_terms[i] corresponds to sympy symbol term{i}.
    """

    # Constants
    if expr is sympy.true:
        return z3.BoolVal(True)

    if expr is sympy.false:
        return z3.BoolVal(False)

    # Atomic symbol: term<i>
    if isinstance(expr, sympy.Symbol):
        name = expr.name
        if not name.startswith("term"):
            raise ValueError(f"Unexpected symbol name: {name}")
        idx = int(name[4:])
        return z3_terms[idx]

    # Negation
    if isinstance(expr, sympy.Not):
        return z3.Not(sympy_to_z3(expr.args[0], z3_terms))

    # Conjunction
    if isinstance(expr, sympy.And):
        return z3.And(*(sympy_to_z3(arg, z3_terms) for arg in expr.args))

    # Disjunction
    if isinstance(expr, sympy.Or):
        return z3.Or(*(sympy_to_z3(arg, z3_terms) for arg in expr.args))

    raise TypeError(f"Unsupported SymPy expression type: {type(expr)}")


def implication_sop_to_clauses_z3(M, N):
    """
    Transform (A OR B OR C) => N into:
        [ (Not(A) OR N), (Not(B) OR N), (Not(C) OR N) ]

    Parameters
    ----------
    M : z3.BoolRef
        Expected to be a disjunction (Or) or a single Boolean term
    N : z3.BoolRef

    Returns
    -------
    list[z3.BoolRef]
    """

    # Handle trivial cases first
    if z3.is_true(M):
        # True => N  ≡ N
        return [N]

    if z3.is_false(M):
        # False => N  ≡ True (no constraints)
        return []

    # Extract disjuncts
    if z3.is_or(M):
        disjuncts = M.children()
    else:
        # Single disjunct case: M = A
        disjuncts = [M]

    # Build clauses: (¬A ∨ N)
    clauses = [z3.Or(z3.Not(A), N) for A in disjuncts]

    return clauses


def add_universal_quantifiers(clauses: List, universal_quantified_vars: List):
    """Adding universal quantifiers for all vars in universal_quantified_vars"""
    result = []
    for clause in clauses:
        result.append(z3.ForAll([*universal_quantified_vars], clause))
    return result


def check_tautology(clause) -> bool:
    """Check whether clause can be directly derived from the axioms we already have"""
    solver = z3.Solver()

    # add all axioms
    highlevel_verification = highlevel_z3_solver()
    highlevel_verification.add_axiom(solver)

    solver.add(z3.Not(clause))
    result = solver.check()
    if result == z3.unsat:
        # can be derived from axioms
        return True
    elif result == z3.sat:
        return False
    else:
        assert False, f"unknown z3 result {result}"


def loop_inference(
    states: List,
    k: int,
    relations: List,
    constants: List,
    constants_mapping: Dict,
    index: int,
):
    import pdb

    pdb.set_trace()
    omega_inv, universal_quantified_vars = compute_omega_k(k, relations, constants)

    a, y = universal_quantified_vars
    b0, b, b_prime = constants
    omega_inv = [ON_star(a, b0), ON_star(y, a), ON_star(a, b), a == y, b == a]

    dataset = compute_dataset(
        states, omega_inv, universal_quantified_vars, constants, constants_mapping
    )
    full_S, full_U, target, reduced_omega = compute_S_U(dataset, omega_inv, index)

    reduced_S = full_S - full_U
    reduced_U = full_U - full_S

    # learn phi in phi => target
    pdb.set_trace()
    phi_selected_idxs = learn_from_partition(reduced_S, full_U)
    selected_omega = [reduced_omega[i] for i in phi_selected_idxs]
    phi, sympy_vars = construct_truth_table_and_extract_expression(
        current_S=project_to_selected(reduced_S, phi_selected_idxs),
        current_U=project_to_selected(full_U, phi_selected_idxs),
        num_selected=len(phi_selected_idxs),
    )
    print(phi)
    z3_phi = sympy_to_z3(phi, z3_terms=selected_omega)
    print("z3_phi", z3_phi)
    clauses = implication_sop_to_clauses_z3(z3_phi, target)
    universal_quantified_clauses = add_universal_quantifiers(
        clauses, universal_quantified_vars
    )
    print(universal_quantified_clauses)
    useful_invariant = [
        x for x in universal_quantified_clauses if not check_tautology(x)
    ]
    print("useful invariant", useful_invariant)
    # learn phi_prime in target => phi_prime
    # phi_prime_selected_idx = learn_from_partition(full_S, reduced_U)
    # phi_prime = construct_truth_table_and_extract_expression()


if __name__ == "__main__":
    # try the example from the ara proposal
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 5.0, 0.0],
        }
    ]
    k = 2
    relations = [ON_star, "equality"]
    b0, b, b_prime = get_consts("b0"), get_consts("b"), get_consts("b_prime")
    constants = [b0, b, b_prime]
    constants_mapping = {b0: "x1", b: "x3", b_prime: "x4"}
    index = 0
    loop_inference(states, k, relations, constants, constants_mapping, index)
