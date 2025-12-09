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
    EnumSort
)

BoxSort, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
# BoxSort = DeclareSort("Box")
Variable_pools = {}

ON_star = Function("ON_star", BoxSort, BoxSort, BoolSort())
higher = Function("higher", BoxSort, BoxSort, BoolSort())
scattered = Function("scattered", BoxSort, BoxSort, BoolSort())


def get_consts(symbol: str):
    (c,) = Consts(f"{symbol}", BoxSort)
    return c


class highlevel_z3_solver:
    def start_verification(self, vc=None):
        s = Solver()
        self.add_axiom(s)
        if vc is not None:
            s.add(Not(vc))
        if s.check() == sat:
            print("VC is satisfiable")
            print(s.model())
            for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                        print(f"ON_star({name1}, {name2}): {s.model().evaluate(ON_star(box1, box2))}")

            for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                    print(f"higher({name1}, {name2}): {s.model().evaluate(higher(box1, box2))}")
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

def top(a):
    n = get_consts("n")
    return ForAll([n], Or(n == a, Not(ON_star(n, a))))

def on_table(a):
    n = get_consts("n")
    return And(ForAll([n], higher(n, a)))
    return ForAll([n], Not(ON_star(a, n)))

if __name__ == "__main__":
    s = Solver()
    hsolver = highlevel_z3_solver()
    hsolver.add_axiom(s)

    b_prime, b, b0, a, b = Consts("b_prime b b0 a b", BoxSort)
    s.add(b_prime != b)
    s.add(top(b_prime))
    s.add(ON_star(b, b0))
    s.add(top(b))
    s.add(ForAll([a], Or(ON_star(a, b0), And(top(a), on_table(a)))))

    s.add(Not(on_table(b0)))
    s.add(Implies(And(on_table(a), on_table(b)), And(higher(a, b), higher(b, a))))
    print(s.check()) 
    print(s.model())

    for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            print(f"ON_star({name1}, {name2}): {s.model().evaluate(ON_star(box1, box2))}")

    for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            print(f"higher({name1}, {name2}): {s.model().evaluate(higher(box1, box2))}")
