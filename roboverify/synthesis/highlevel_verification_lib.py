from z3 import (
    And,
    BoolSort,
    Consts,
    DeclareSort,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Solver,
    sat,
    unsat,
)

BoxSort = DeclareSort("Box")
Variable_pools = {}

ON_star = Function("ON_star", BoxSort, BoxSort, BoolSort())
higher = Function("higher", BoxSort, BoxSort, BoolSort())


def get_consts(symbol: str):
    (c,) = Consts(f"{symbol}", BoxSort)
    return c


class highlevel_z3_solver:
    def start_verification(self, vc):
        s = Solver()
        self.add_axiom(s)
        s.add(Not(vc))
        if s.check() == sat:
            print("VC is satisfiable")
        elif s.check() == unsat:
            print("VC is unsatisfiable")

    def add_axiom(self, s: Solver):
        x, y, c = Consts("x y c", BoxSort)
        s.add(
            ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c)))
        )
        s.add(ForAll([x], ON_star(x, x)))
        s.add(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))
                ),
            )
        )
        s.add(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))
                ),
            )
        )
        s.add(ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))))

        s.add(ForAll([x, y, c], Implies(And(higher(x, y), higher(y, c)), higher(x, c))))
        s.add(ForAll([x], higher(x, x)))
        s.add(
            ForAll(
                [x, y, c],
                Implies(
                    And(higher(x, y), higher(x, c)), Or(higher(y, c), higher(c, y))
                ),
            )
        )
