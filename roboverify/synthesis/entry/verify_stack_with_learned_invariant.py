from z3 import And, Consts, ForAll, Not, Or

from synthesis.api.program import Assign, Program, Put, While
from synthesis.inference_lib.inference import run_proposal_example
from synthesis.verification_lib.highlevel_verification_lib import BoxSort, ON_star


def verify_stack_program_with_learned_invariant():
    """Infer invariant from examples and verify the stack program."""
    learned_invariant, candidate_lists = run_proposal_example()
    _ = candidate_lists

    b_prime, b, n, b0 = Consts("b_prime b n b0", BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            instantiated_cond=And(
                ForAll([n], Or(b_prime == n, Not(ON_star(n, b_prime)))),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=learned_invariant,
        ),
    ]
    program = Program(2, instructions=instructions)

    m, n = Consts("m n", BoxSort)
    precondition = ForAll([m], ForAll([n], Or(m == n, Not(ON_star(n, m)))))

    m, b0 = Consts("m b0", BoxSort)
    postcondition = ForAll([m], ON_star(m, b0))

    program.highlevel_verification(precondition, postcondition)


if __name__ == "__main__":
    verify_stack_program_with_learned_invariant()
