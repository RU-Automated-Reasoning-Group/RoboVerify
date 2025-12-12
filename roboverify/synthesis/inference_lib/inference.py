from synthesis.verification_lib.highlevel_verification_lib import (
    ON_star,
    BoxSort,
    get_consts,
)
from synthesis.decision_tree import on, get_block_pos

from typing import List, Tuple, Set, Dict
import itertools
from copy import deepcopy

def add_all_pairs(vocabulary, r, all_vars):
    for v1 in all_vars:
        for v2 in all_vars:
            if str(v1) != str(v2):
                if r == "equality":
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
            block1_name = var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            block2_name = var_mapping[arg1] if arg1 in var_mapping else constants_mapping[arg1]
            assert isinstance(block1_name, str) and isinstance(block2_name, str)
            data.append(
                on(
                    state[block1_name],
                    state[block2_name],
                )
            )
        else:
            # equality
            arg0, arg1 = predicate.arg(0), predicate.arg(1)
            block1_name = var_mapping[arg0] if arg0 in var_mapping else constants_mapping[arg0]
            block2_name = var_mapping[arg1] if arg1 in var_mapping else constants_mapping[arg1]
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
            U.add(d[:index] + d[index+1:])
    return S, U, target, rest_omega_k


def learn_from_partition(
    S,
    U,
):
    """Encode the constraint system of the learning program using z3 for samples in S and U
    returns the indexes selected
    """


def construct_truth_table_and_extract_expression():
    """Construct the truth table for all possible valuations, reject those in U and accept the rest in S, then extact a minimized the logic expression for this"""


def loop_inference(states: List, k: int, relations: List, constants: List, index: int):
    omega_inv, universal_quantified_vars = compute_omega_k(k, relations, constants)
    dataset = compute_dataset(states, omega_inv, universal_quantified_vars, constants)
    full_S, full_U, target, reduced_omega = compute_S_U(dataset, omega_inv, index)

    # learn phi in phi => omega_inv[index]

    reduced_S = full_S - full_U
    reduced_U = full_U - full_S
    phi_selected_idxs = learn_from_partition(reduced_S, full_U)
    phi = construct_truth_table_and_extract_expression(full_U)

    # learn phi_prime in omega_inv[index] => phi_prime
    phi_prime_selected_idx = learn_from_partition(full_S, reduced_U)
    phi_prime = construct_truth_table_and_extract_expression()



if __name__ == "__main__":
    # try the example from the ara proposal
    pass