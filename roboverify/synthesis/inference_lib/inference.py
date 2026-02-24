import itertools
import pdb
from copy import deepcopy
from typing import Dict, List, Set, Tuple

import sympy
import z3

from synthesis.inference_lib import quant_enum_merge
from synthesis.util import on
from synthesis.verification_lib.highlevel_verification_lib import (  # b9,; b10,; b11,; b12,
    BoxSort,
    ON_star,
    ON_star_zero,
    Top,
    get_consts,
    highlevel_z3_solver,
)

z3.set_option("smt.core.minimize", "true")


def add_all_pairs(vocabulary, r, all_vars):
    for v1 in all_vars:
        for v2 in all_vars:
            if str(v1) != str(v2):
                if isinstance(r, str) and r == "equality":
                    vocabulary.append(v1 == v2)
                elif r == ON_star:
                    vocabulary.append(r(v1, v2))
                elif r == ON_star_zero:
                    vocabulary.append(r(v1, v2))
                else:
                    assert False, f"unknown relation r: {r}"


def add_univariable_predicate(vocabulary, r, all_vars):
    for v in all_vars:
        vocabulary.append(r(v))


def forall_exists_compute_omega(
    universally_quantified_vars: List,
    existential_quantified_vars: int,
    relations: List,
    constants: List,
):
    all_vars = universally_quantified_vars + existential_quantified_vars + constants
    omega_inv = []
    for r in relations:
        if str(r) == "Top":
            add_univariable_predicate(omega_inv, r, all_vars)
        else:
            add_all_pairs(omega_inv, r, all_vars)
    return omega_inv


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


def forall_exists_compute_dataset(
    states_zero: List[dict],
    states: List[Dict],
    omega_k: List,
    universal_quantified_vars: List,
    existential_quantified_vars: List,
    constatns: List,
    constants_mappings: List[Dict],
    all_witness_permutations: List,
) -> List[Set]:
    """assuming there is only a single state for now. For each witness permutation, we can
    obtain one dataset. In the future, when we have multiple state, we need to consider
    different state might have different witness permutations as well
    """

    all_datasets = []
    for witness_assignment in all_witness_permutations:
        dataset = set()
        state, state_zero, mapping = states[0], states_zero[0], constants_mappings[0]

        existential_var_mapping = {
            var: witness_assignment[idx]
            for idx, var in enumerate(existential_quantified_vars)
        }

        all_objects = list(state.keys())
        for universal_assignment in itertools.product(
            all_objects, repeat=len(universal_quantified_vars)
        ):
            # universal_var_mapping maps universal quantified variables to its corresponding real block
            universal_var_mapping = {
                var: universal_assignment[idx]
                for idx, var in enumerate(universal_quantified_vars)
            }
            var_mapping = {**universal_var_mapping, **existential_var_mapping}
            d = compute_data(state_zero, state, omega_k, var_mapping, mapping)
            print("var_mapping", var_mapping)
            print("omega_k", omega_k)
            print("data", d)
            dataset.add(d)

        all_datasets.append(dataset)
    return all_datasets


def compute_dataset(
    states_zero: List[Dict],
    states: List[Dict],
    omega_k: List,
    universal_quantified_vars: List,
    constants: List,
    constants_mappings: List[Dict],
) -> Set:
    """Compute the dataset for each state and for each relation in the vocabulary
    Each universally quantified variable needs to be bind into one of the existing blocks in the states
    """
    dataset = set()
    for state_zero, state, mapping in zip(states_zero, states, constants_mappings):
        all_objects = list(state.keys())
        for assignment in itertools.product(
            all_objects, repeat=len(universal_quantified_vars)
        ):
            # var_mapping maps universal quantified variables to its corresponding real block
            var_mapping = {
                var: assignment[idx]
                for idx, var in enumerate(universal_quantified_vars)
            }
            d = compute_data(
                state_zero,
                state,
                omega_k,
                var_mapping,
                mapping,
            )
            print("var_mapping", var_mapping)
            print("omega_k", omega_k)
            print("data", d)
            dataset.add(d)
    return dataset


def compute_data(
    state_zero,
    state,
    omega_k: List,
    var_mapping: Dict,
    constants_mapping: Dict,
) -> Tuple:
    data = []
    for predicate in omega_k:
        print("predicate:", predicate)
        if str(predicate).startswith("Top"):
            arg0 = predicate.arg(0)
            block1_name = (
                var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            )
            top_flag = True
            for other_block_name in state:
                if block1_name != other_block_name:
                    if on.on_star_implementation(
                        state[other_block_name], state[block1_name]
                    ):
                        top_flag = False
                        break
            data.append(top_flag)
        elif str(predicate).startswith("ON_star") and not str(predicate).startswith(
            "ON_star_zero"
        ):
            arg0, arg1 = predicate.arg(0), predicate.arg(1)
            block1_name = (
                var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            )
            block2_name = (
                var_mapping[arg1] if arg1 in var_mapping else constants_mapping[arg1]
            )
            if not (isinstance(block1_name, str) and isinstance(block2_name, str)):
                pdb.set_trace()
            assert isinstance(block1_name, str) and isinstance(block2_name, str)
            # if block1_name == "tbl" or block2_name == "tbl":
            # pdb.set_trace()
            data.append(
                on.on_star_implementation(
                    state[block1_name],
                    state[block2_name],
                )
            )
        elif str(predicate).startswith("ON_star_zero"):
            arg0, arg1 = predicate.arg(0), predicate.arg(1)
            block1_name = (
                var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            )
            block2_name = (
                var_mapping[arg1] if arg1 in var_mapping else constants_mapping[arg1]
            )
            assert isinstance(block1_name, str) and isinstance(block2_name, str)
            data.append(
                on.on_star_implementation(
                    state_zero[block1_name],
                    state_zero[block2_name],
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
    n = None
    for d in S:
        n = len(d)
        break

    if n is None:
        for d in U:
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
    # opt.add(sel[1] == 0)
    # Objective φc: minimize Σ_i sel_i
    opt.minimize(z3.Sum(sel))

    # Solve
    result = opt.check()
    if result == z3.sat:
        model = opt.model()
        print("model is", model)
        chosen = [i for i in range(n) if model[sel[i]].as_long() == 1]
        return chosen
    else:
        pdb.set_trace()


def construct_truth_table_and_extract_expression_for_phi(
    current_S: Set, current_U: Set, num_selected: int
):
    """Construct the truth table for all possible valuations, reject those in U and accept the rest in S, then extact a minimized the logic expression for this

    Current_S contains data from S projected to selected indices
    Current_U contains data from U projected to selected indices

    Current strategy: reject all sample from current_U and accepting all the rest,
    include those in current_S
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


def construct_truth_table_and_extract_expression_for_phi_prime(
    current_S: Set, current_U: Set, num_selected: int
):
    """Construct the truth table for all possible valuations, reject those in U and accept the rest in S, then extact a minimized the logic expression for this

    Current_S contains data from S projected to selected indices
    Current_U contains data from U projected to selected indices

    Current strategy: accept all sample from current_S and reject all the rest,
    include those in current_U
    """
    assert len(current_S & current_U) == 0

    accepted_values = []
    rejected_values = []
    for assignment in itertools.product((0, 1), repeat=num_selected):
        if assignment in current_S:
            accepted_values.append(assignment)
        else:
            rejected_values.append(assignment)

    good = all([u_data in rejected_values for u_data in current_U])
    if not good:
        raise ValueError("Not all assignment from current_U are rejected")

    var_symbols = sympy.symbols(f"term0:{num_selected}")
    return sympy.POSform(var_symbols, accepted_values), var_symbols


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


def implication_pos_to_clauses_z3(target, N):
    """
    Given Z3 expressions `target` and `N` (N in product-of-sums form),
    return a list of clauses: [Not(target) Or Xi] for each sum term Xi in N.
    """

    not_target = z3.Not(target)

    # If N is an AND, extract its arguments
    if z3.is_and(N):
        clauses = list(N.children())
    else:
        # Otherwise treat N as a single clause
        clauses = [N]

    # Build Not(target) OR Xi for each clause Xi
    result = [z3.Or(not_target, clause) for clause in clauses]

    return result


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


def add_universal_quantifiers(clauses: List, universal_quantified_vars: List) -> List:
    """Adding universal quantifiers for all vars in universal_quantified_vars"""
    result = []
    for clause in clauses:
        result.append(z3.ForAll([*universal_quantified_vars], clause))
    return result


def add_universal_and_existential_quantifiers(
    clauses: List, universal_quantified_vars: List, existential_quantified_vars: List
) -> List:
    """Adding universal quantifiers for all vars in universal_quantified_vars,
    and adding existential quantifiers for all vars in existential_quantified_vars"""
    result = []
    for clause in clauses:
        result.append(
            z3.ForAll(
                [*universal_quantified_vars],
                z3.Exists([*existential_quantified_vars], clause),
            )
        )
    return result


def check_tautology(clause) -> bool:
    """Check whether clause can be directly derived from the axioms we already have
    Returns True if the caluse is a tautology
    """
    solver = z3.Solver()

    # add all axioms
    highlevel_verification = highlevel_z3_solver()
    highlevel_verification.add_axiom(solver)
    highlevel_verification.add_axiom_on_star_zero(solver)

    solver.add(z3.Not(clause))
    result = solver.check()
    if result == z3.unsat:
        # can be derived from axioms
        return True
    elif result == z3.sat:
        return False
    else:
        assert False, f"unknown z3 result {result}"


def forall_exists_loop_inference_by_index(
    states_zero: List,
    states: List,
    constants: List,
    constants_mappings: List[Dict],
    index: int,
    omega_inv: List,
    universal_quantified_vars: List,
    existential_quantified_vars: List,
    all_witness_permutations: List,
):
    print(
        f"=========== learning with index = {index} with target predicate {omega_inv[index]}"
    )

    all_datasets: List = forall_exists_compute_dataset(
        states_zero,
        states,
        omega_inv,
        universal_quantified_vars,
        existential_quantified_vars,
        constants,
        constants_mappings,
        all_witness_permutations,
    )
    all_full_S, all_full_U, all_target, all_reduced_omega = [], [], [], []
    all_reduced_S, all_reduced_U = [], []
    for dataset in all_datasets:
        full_S, full_U, target, reduced_omega = compute_S_U(dataset, omega_inv, index)
        all_full_S.append(full_S)
        all_full_U.append(full_U)
        all_target.append(target)
        all_reduced_omega.append(reduced_omega)

        reduced_S = full_S - full_U
        reduced_U = full_U - full_S
        all_reduced_S.append(reduced_S)
        all_reduced_U.append(reduced_U)

    iter = 0
    all_invariant = []
    for full_S, full_U, reduced_S, reduced_U, reduced_omega in zip(
        all_full_S, all_full_U, all_reduced_S, all_reduced_U, all_reduced_omega
    ):
        print("learn with dataset index", iter)
        iter += 1
        # learn phi in phi => target
        phi_selected_idxs = learn_from_partition(reduced_S, full_U)
        selected_phi_omega = [reduced_omega[i] for i in phi_selected_idxs]
        phi, phi_sympy_vars = construct_truth_table_and_extract_expression_for_phi(
            current_S=project_to_selected(reduced_S, phi_selected_idxs),
            current_U=project_to_selected(full_U, phi_selected_idxs),
            num_selected=len(phi_selected_idxs),
        )
        print("phi", phi)
        z3_phi = sympy_to_z3(phi, z3_terms=selected_phi_omega)
        print("z3_phi", z3_phi)
        phi_clauses = implication_sop_to_clauses_z3(z3_phi, target)
        forall_exists_quantified_phi_clauses = (
            add_universal_and_existential_quantifiers(
                phi_clauses, universal_quantified_vars, existential_quantified_vars
            )
        )
        print("forall_exists phi clauses", forall_exists_quantified_phi_clauses)
        useful_invariant_with_phi = [
            z3.simplify(x)
            for x in forall_exists_quantified_phi_clauses
            # if not check_tautology(x)
        ]
        print("useful invariant using phi", useful_invariant_with_phi)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # learn phi_prime in target => phi_prime
        phi_prime_selected_idxs = learn_from_partition(full_S, reduced_U)
        selected_phi_prime_omega = [reduced_omega[i] for i in phi_prime_selected_idxs]
        phi_prime, phi_prime_sympy_vars = (
            construct_truth_table_and_extract_expression_for_phi_prime(
                current_S=project_to_selected(full_S, phi_prime_selected_idxs),
                current_U=project_to_selected(reduced_U, phi_prime_selected_idxs),
                num_selected=len(phi_prime_selected_idxs),
            )
        )
        print("phi_prime", phi_prime)
        z3_phi_prime = sympy_to_z3(phi_prime, z3_terms=selected_phi_prime_omega)
        print("z3_phi_prime", z3_phi_prime)
        phi_prime_clauses = implication_pos_to_clauses_z3(target, z3_phi_prime)
        forall_exists_quantified_phi_prime_clauses = (
            add_universal_and_existential_quantifiers(
                phi_prime_clauses,
                universal_quantified_vars,
                existential_quantified_vars,
            )
        )
        print(
            "forall_exists quantified phi prime clauses",
            forall_exists_quantified_phi_prime_clauses,
        )
        useful_invariant_with_phi_prime = [
            z3.simplify(x)
            for x in forall_exists_quantified_phi_prime_clauses
            # if not check_tautology(x)
        ]
        print("useful invariant using phi prime", useful_invariant_with_phi_prime)
        all_invariant.extend(
            useful_invariant_with_phi + useful_invariant_with_phi_prime
        )
        print("total length", len(all_invariant))
    all_invariant_with_exists = filter_exists(all_invariant)
    return all_invariant_with_exists


def filter_exists(expressions):
    return [e for e in expressions if contains_exists(e)]


def contains_exists(expr: z3.ExprRef) -> bool:
    # If this node itself is a quantifier
    if z3.is_quantifier(expr):
        # If it's an existential → keep it
        if expr.is_exists():
            return True

        # Otherwise recurse into its body
        return contains_exists(expr.body())

    # Recurse through children of any other expression
    for child in expr.children():
        if contains_exists(child):
            return True

    return False


def loop_inference_by_index(
    states_zero: List,
    states: List,
    constants: List,
    constants_mappings: List[Dict],
    index: int,
    omega_inv: List,
    universal_quantified_vars: List,
):
    print(
        f"=========== learning with index = {index} with target predicate {omega_inv[index]}"
    )

    dataset = compute_dataset(
        states_zero,
        states,
        omega_inv,
        universal_quantified_vars,
        constants,
        constants_mappings,
    )
    full_S, full_U, target, reduced_omega = compute_S_U(dataset, omega_inv, index)

    print("S", full_S)
    print("U", full_U)
    reduced_S = full_S - full_U
    reduced_U = full_U - full_S
    print("full_S", len(full_S))
    print("full_U", len(full_U))
    print("reduced_S", len(reduced_S))
    print("reduced_U", len(reduced_U))

    # learn phi in phi => target
    phi_selected_idxs = learn_from_partition(reduced_S, full_U)
    selected_phi_omega = [reduced_omega[i] for i in phi_selected_idxs]
    phi, phi_sympy_vars = construct_truth_table_and_extract_expression_for_phi(
        current_S=project_to_selected(reduced_S, phi_selected_idxs),
        current_U=project_to_selected(full_U, phi_selected_idxs),
        num_selected=len(phi_selected_idxs),
    )
    print("phi", phi)
    z3_phi = sympy_to_z3(phi, z3_terms=selected_phi_omega)
    print("z3_phi", z3_phi)
    phi_clauses = implication_sop_to_clauses_z3(z3_phi, target)
    universal_quantified_phi_clauses = add_universal_quantifiers(
        phi_clauses, universal_quantified_vars
    )
    print("universal quantified phi clauses", universal_quantified_phi_clauses)
    useful_invariant_with_phi = [
        z3.simplify(x)
        for x in universal_quantified_phi_clauses
        if not check_tautology(x)
    ]
    print("useful invariant using phi", useful_invariant_with_phi)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # learn phi_prime in target => phi_prime
    phi_prime_selected_idxs = learn_from_partition(full_S, reduced_U)
    selected_phi_prime_omega = [reduced_omega[i] for i in phi_prime_selected_idxs]
    phi_prime, phi_prime_sympy_vars = (
        construct_truth_table_and_extract_expression_for_phi_prime(
            current_S=project_to_selected(full_S, phi_prime_selected_idxs),
            current_U=project_to_selected(reduced_U, phi_prime_selected_idxs),
            num_selected=len(phi_prime_selected_idxs),
        )
    )
    print("phi_prime", phi_prime)
    z3_phi_prime = sympy_to_z3(phi_prime, z3_terms=selected_phi_prime_omega)
    print("z3_phi_prime", z3_phi_prime)
    phi_prime_clauses = implication_pos_to_clauses_z3(target, z3_phi_prime)
    universal_quantified_phi_prime_clauses = add_universal_quantifiers(
        phi_prime_clauses, universal_quantified_vars
    )
    print(
        "universal quantified phi prime clauses", universal_quantified_phi_prime_clauses
    )
    useful_invariant_with_phi_prime = [
        z3.simplify(x)
        for x in universal_quantified_phi_prime_clauses
        if not check_tautology(x)
    ]
    print("useful invariant using phi prime", useful_invariant_with_phi_prime)
    all_invariant = useful_invariant_with_phi + useful_invariant_with_phi_prime
    print("total length", len(all_invariant))
    return all_invariant


def get_universal_quantified_vars(n_forall: int) -> List:
    return [get_consts(f"ux{i}") for i in range(1, n_forall + 1)]


def get_existential_quantified_vars(n_exists: int) -> List:
    return [get_consts(f"ex{i}") for i in range(1, n_exists + 1)]


def compute_all_possible_witness_permutations(
    n_exists: int, candidate_witness: List
) -> List[List]:
    """For each universally quantified variable, assign one constant from the input candidate_witness as the witness.
    Return all possible witness permutations.
    """
    if n_exists < 0:
        raise ValueError("n_forall must be non-negative")
    # If there are zero universal vars, there is one empty assignment
    if n_exists == 0:
        return [[]]
    # Cartesian product: allow repetition of candidates across universal vars
    return [list(p) for p in itertools.product(candidate_witness, repeat=n_exists)]


# This function is used to learn forall-exists variants
def forall_exists_loop_inference(
    states_zero: List,
    states: List,
    n_forall: int,
    n_exists: int,
    relations: List,
    constants: List,
    constants_mappings: List[Dict],
):
    universally_quantified_vars = get_universal_quantified_vars(n_forall)
    existential_quantified_vars = get_existential_quantified_vars(n_exists)
    all_objects = list(states[0].keys())
    all_witness_permuatations = compute_all_possible_witness_permutations(
        n_exists, all_objects
    )

    omega_inv = forall_exists_compute_omega(
        universally_quantified_vars, existential_quantified_vars, relations, constants
    )

    # (ux1,) = universally_quantified_vars
    # (ex1,) = existential_quantified_vars
    # b0, b = constants
    # omega_inv = [ON_star(ex1, b0), Top(ex1)]
    print("omega_inv", omega_inv)

    inferred_invariants = []
    for i in range(len(omega_inv)):
        inferred_invariants.extend(
            forall_exists_loop_inference_by_index(
                states_zero,
                states,
                constants,
                constants_mappings,
                i,
                omega_inv,
                universally_quantified_vars,
                existential_quantified_vars,
                all_witness_permuatations,
            )
        )
    print("inferred_invariants count", len(inferred_invariants))

    # filtered_invariants = check_redundancy(inferred_invariants)
    # print("filtered candidates", len(filtered_invariants))
    # print(filtered_invariants)


# This function is used to learn forall only invariants
def loop_inference(
    states_zero: List,
    states: List,
    k: int,
    relations: List,
    constants: List,
    constants_mappings: List[Dict],
):
    omega_inv, universal_quantified_vars = compute_omega_k(k, relations, constants)

    ############### ! temp shortcut starts
    # x, y = universal_quantified_vars
    # b0, b, tbl = constants
    # omega_inv = [
    #     x == tbl,
    #     y == tbl,
    #     ON_star(x, b0),
    #     ON_star(x, y),
    #     ON_star_zero(x, y),
    #     ON_star(b, x),
    #     ON_star_zero(y, x),
    # ]
    ############## ! temp shortcut end

    print("omega_inv", omega_inv)

    inferred_invariants = []
    for i in range(len(omega_inv)):
        inferred_invariants.extend(
            loop_inference_by_index(
                states_zero,
                states,
                constants,
                constants_mappings,
                i,
                omega_inv,
                universal_quantified_vars,
            )
        )
    print("inferred_invariants count", len(inferred_invariants))

    filtered_invariants = check_redundancy(inferred_invariants)
    print("filtered candidates", len(filtered_invariants))
    print(filtered_invariants)

    final_result = z3.And(*filtered_invariants)
    print("final result", final_result)

    print("checking equivalent with ground truth")
    solver = z3.Solver()
    solver.set(unsat_core=True)

    highlevel_verification = highlevel_z3_solver()
    highlevel_verification.add_axiom(solver)
    highlevel_verification.add_axiom_on_star_zero(solver)
    (b0,) = constants
    highlevel_verification.add_unstack_b0_bottom_loop_invarinat(solver, b0)  # not
    # solver.assert_and_track(z3.Not(z3.ForAll([x], z3.Implies(ON_star(x, b0), x != b))), "not_b_neq_b0")
    # solver.assert_and_track(ON_star(x, b0), "on_b0")
    # solver.assert_and_track(
    #     z3.Not(
    #         z3.And(
    #             ON_star(x, b0),
    #             z3.Implies(ON_star(x, y), ON_star_zero(x, y))
    #         )
    #     ), "original_1"
    # )

    def extract_direct_on(model, blocks, names, ON_star):
        direct_on = {n: "table" for n in names}

        for i, (a_name, a) in enumerate(zip(names, blocks)):
            for j, (b_name, b) in enumerate(zip(names, blocks)):

                if a_name == b_name:
                    continue

                if not model.evaluate(ON_star(a, b)):
                    continue

                # check if there exists an intermediate block
                has_middle = False

                for k, (c_name, c) in enumerate(zip(names, blocks)):
                    if c_name in [a_name, b_name]:
                        continue

                    if model.evaluate(ON_star(a, c)) and model.evaluate(ON_star(c, b)):
                        has_middle = True
                        break

                if not has_middle:
                    direct_on[a_name] = b_name

        return direct_on

    def build_stacks(direct_on):
        stacks = []
        visited = set()

        blocks = list(direct_on.keys())

        bases = [b for b, v in direct_on.items() if v == "table"]

        for base in bases:
            if base in visited:
                continue

            stack = [base]
            visited.add(base)
            top = base

            while True:
                above = None
                for b, v in direct_on.items():
                    if v == top and b not in visited:
                        above = b
                        break

                if above:
                    stack.append(above)
                    visited.add(above)
                    top = above
                else:
                    break

            stacks.append(stack)

        return stacks

    def print_stacks(stacks, title):
        print("\n" + title)

        for stack in stacks:
            for block in reversed(stack):
                print(f"   [{block}]")
            print("  --------")
            print("   table\n")

    from PIL import Image, ImageDraw, ImageFont

    def draw_stacks(stacks, filename, title):
        width, height = 800, 400
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        block_w = 70
        block_h = 30
        gap = 60

        draw.text((10, 10), title, fill="black", font=font)

        x = 50
        for stack in stacks:
            y = 350

            for block in stack:
                draw.rectangle(
                    [x, y - block_h, x + block_w, y], outline="black", width=2
                )
                draw.text((x + 15, y - block_h + 5), block, fill="black", font=font)
                y -= block_h + 5

            draw.line([x, y + 5, x + block_w, y + 5], fill="black", width=3)
            x += block_w + gap

        img.save(filename)

    # solver.assert_and_track(
    #     z3.Not(
    #         z3.And(
    #             ON_star(x, b0),
    #             z3.Implies(ON_star_zero(x, y), ON_star(x, y))
    #         )
    #     ), "original_2")

    # solver.assert_and_track(x != tbl, "x_neq_tbl")
    # solver.assert_and_track(y != tbl, "y_neq_tbl")
    # solver.assert_and_track(ON_star(b, x), "b_on_x")
    # solver.assert_and_track(
    #     z3.Not(
    #         z3.And(
    #             ON_star(b, x),
    #             z3.Implies(ON_star(x, y), ON_star_zero(y, x))
    #         )
    #     ),
    #     "original_3"
    # )
    # solver.assert_and_track(
    #     z3.Not(
    #         z3.And(
    #             ON_star(b, x),
    #             z3.Implies(ON_star_zero(y, x), ON_star(x, y))
    #         )
    #     ),
    #     "original_4"
    # )

    for idx, candidate in enumerate(filtered_invariants):
        solver.assert_and_track(candidate, f"term{idx}")
        print(f"term{idx}:", candidate)
    print(solver.check())
    print("Unsat Core:", solver.unsat_core())
    if solver.check() == z3.sat:
        print("constraints satisfiable")
        print("model is")
        print(solver.model())
        blocks = [b9, b10, b11, b12]
        names = ["b9", "b10", "b11", "b12"]

        start_state = extract_direct_on(solver.model(), blocks, names, ON_star_zero)
        current_state = extract_direct_on(solver.model(), blocks, names, ON_star)

        start_stacks = build_stacks(start_state)
        current_stacks = build_stacks(current_state)

        print_stacks(start_stacks, "START STATE")
        print_stacks(current_stacks, "CURRENT STATE")

        draw_stacks(start_stacks, "start_state.png", "Start State")
        draw_stacks(current_stacks, "current_state.png", "Current State")

    return final_result, filtered_invariants


def check_redundancy(candidates: List) -> List:
    print("======= starting check redundancy =======")
    filtered_invariants = []
    for candidate in candidates:
        solver = z3.Solver()
        highlevel_verification = highlevel_z3_solver()
        highlevel_verification.add_axiom(solver)
        highlevel_verification.add_axiom_on_star_zero(solver)

        for existing in filtered_invariants:
            solver.add(existing)
        solver.add(z3.Not(candidate))

        result = solver.check()
        if result == z3.sat:
            # not redundant
            filtered_invariants.append(candidate)
            print("not redundant", candidate)
        elif result == z3.unsat:
            print("redundant candidate", candidate)
        else:
            assert False, f"unknown verification result = {result}"

    return filtered_invariants


def run_proposal_example():
    states_zero: List[Dict] = []
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 5.0, 0.0],
            "x5": [10.0, 10.0, 0.0],
        },
        {"x1": [0.0, 0.0, 0.0], "x2": [5.0, 5.0, 0.0], "x3": [10.0, 10.0, 0.0]},
    ]
    k = 2
    relations = [ON_star, "equality"]
    b0, b = get_consts("b0"), get_consts("b")
    constants = [b0, b]
    constants_mappings = [
        {b0: "x1", b: "x3"},
        {b0: "x1", b: "x1"},
    ]

    return loop_inference(
        states_zero, states, k, relations, constants, constants_mappings
    )


def run_unstack_example():
    states_zero: List[Dict] = [{}, {}, {}, {}]
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [10.0, 0.0, 0.0],
            "x4": [5.0, 0.0, 0.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [15.0, 0.0, 0.0],
            "x3": [10.0, 0.0, 0.0],
            "x4": [5.0, 0.0, 0.0],
        },
    ]
    n_forall = 2
    relations = [ON_star, "equality"]
    b0 = get_consts("b0")
    constants = [b0]
    constants_mappings = [
        {b0: "x1"},
        {b0: "x1"},
        {b0: "x1"},
        {b0: "x1"},
    ]

    return loop_inference(
        states_zero, states, n_forall, relations, constants, constants_mappings
    )


def run_reverse_example():
    states_zero: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": [-100.0, -100.0, -100.0],
        },
    ]
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [5.0, 0.0, 0.0],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.05],
            "x5": [5.0, 0.0, 0.0],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [5.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.05],
            "x5": [5.0, 0.0, 0.0],
            "tbl": [-100.0, -100.0, -100.0],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [5.0, 0.0, 0.15],
            "x3": [5.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.05],
            "x5": [5.0, 0.0, 0.0],
            "tbl": [-100.0, -100.0, -100.0],
        },
    ]
    k = 2
    relations = [ON_star, ON_star_zero, "equality"]
    b0, b = get_consts("b0"), get_consts("b")
    tbl = get_consts("tbl")
    constants = [b0, b, tbl]
    constants_mappings = [
        {b0: "x1", b: "tbl", tbl: "tbl"},
        {b0: "x1", b: "x5", tbl: "tbl"},
        {b0: "x1", b: "x4", tbl: "tbl"},
        {b0: "x1", b: "x3", tbl: "tbl"},
        {b0: "x1", b: "x2", tbl: "tbl"},
    ]

    return loop_inference(
        states_zero, states, k, relations, constants, constants_mappings
    )


def run_forall_exists_example():

    states_zero: List[Dict] = [[]]
    states: List[Dict] = [
        {
            "x1": (0.0, 0.0, 0.0),
            "x2": (0.0, 0.0, 0.05),
            "x3": (0.0, 0.0, 0.1),
            "x4": (5.0, 5.0, 0.0),
            "x5": (10.0, 10.0, 0.0),
        }
    ]
    n_forall = 1
    n_exists = 1
    relations = [ON_star, "equality", Top]
    b0, b = get_consts("b0"), get_consts("b")
    constants = [b0, b]
    constants_mappings = [
        {b0: "x1", b: "x3"},
    ]
    x, y = get_consts("x"), get_consts("y")
    m, n = get_consts("m"), get_consts("n")

    test_expr = z3.ForAll([x, y], z3.Exists([m, n], z3.And(ON_star(x, n), y == m)))
    py_expr = quant_enum_merge.z3_to_python_expr(test_expr)
    domain = [val for _, val in states[0].items()]
    function_imples = {
        "ON_star": on.on_star_implementation,
        "ON_star_zer": on.on_star_implementation,
    }
    env = {"b0": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.1]}
    quant_enum_merge.eval_quantified_expr(py_expr, env, domain, function_imples)
    print(py_expr)
    py_rst = quant_enum_merge.eval_quantified_expr(
        py_expr, env, domain, function_imples
    )
    print(py_rst)

    test_expr_1 = z3.ForAll([x], z3.Exists([y], Top(y)))

    py_witness = quant_enum_merge.compute_witness_map(
        py_expr, env, domain, function_imples
    )
    print(py_witness)
    exit()
    return forall_exists_loop_inference(
        states_zero,
        states,
        n_forall,
        n_exists,
        relations,
        constants,
        constants_mappings,
    )
