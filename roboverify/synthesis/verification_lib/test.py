from program import *
from z3 import *

def stack_program():
    b_prime, b, n, b0, a = Consts("b_prime b n b0 a", BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            cond=(
                Exists(
                    [b_prime],
                    And(
                        ForAll([n], Or(b_prime == n, Not(ON_star(n, b_prime)))),
                        b_prime != b,
                    ),
                )
            ),
            instantiated_cond=And(
                ForAll([n], Or(b_prime == n, Not(ON_star(n, b_prime)))),
                b_prime != b,
            ),
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=And(
                ForAll(
                    [a],
                    Or(
                        ON_star(a, b0),
                        And(
                            ForAll([n], Or(n == a, Not(ON_star(n, a)))),
                            ForAll([n], higher(n, a)),
                        ),
                    ),
                ),
                ON_star(b, b0),
                ForAll([n], Or(n == b, Not(ON_star(n, b)))),
            ),
        ),
    ]
    p = Program(2, instructions=instructions)

    m, n = Consts("m n", BoxSort)
    precondition = ForAll([m], ForAll([n], Or(m == n, Not(ON_star(n, m)))))

    m, n, b0 = Consts("m n b0", BoxSort)
    postcondition = ForAll([m], ON_star(m, b0))

    # print(p.VC_gen(precondition, postcondition))
    p.highlevel_verification(precondition, postcondition)

if __name__ == "__main__":
    stack_program()
