from z3 import (
    And,
    BoolSort,
    Consts,
    DeclareSort,
    EnumSort,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Solver,
    sat,
    unsat,
)

# BoxSort, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])
BoxSort = DeclareSort("Box")
Variable_pools = {}

ON_star = Function("ON_star", BoxSort, BoxSort, BoolSort())
ON_star_zero = Function("ON_star_zero", BoxSort, BoxSort, BoolSort())
higher = Function("higher", BoxSort, BoxSort, BoolSort())
scattered = Function("scattered", BoxSort, BoxSort, BoolSort())


def get_consts(symbol: str):
    (c,) = Consts(f"{symbol}", BoxSort)
    return c


class highlevel_z3_solver:
    def start_verification(self, vc=None):
        s = Solver()
        self.add_axiom(s)
        self.add_axiom_on_star_zero(s)
        if vc is not None:
            s.add(Not(vc))
        if s.check() == sat:
            print("VC is satisfiable")
            print(s.model())
            # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            #             print(f"ON_star({name1}, {name2}): {s.model().evaluate(ON_star(box1, box2))}")

            # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            #         print(f"higher({name1}, {name2}): {s.model().evaluate(higher(box1, box2))}")
        elif s.check() == unsat:
            print("VC is unsatisfiable")

    def add_axiom(self, s: Solver):
        x, y, c = Consts("x y c", BoxSort)
        s.assert_and_track(
            ForAll(
                [x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))
            ),
            "on1",
        )
        s.assert_and_track(ForAll([x], ON_star(x, x)), "on2")
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))
                ),
            ),
            "on3",
        )
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))
                ),
            ),
            "on4",
        )
        s.assert_and_track(
            ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))),
            "on5",
        )
        tbl, = Consts("tbl", BoxSort)
        s.assert_and_track(
            ForAll([x], Implies(Or(ON_star(x, tbl), ON_star(tbl, x)), x == tbl)),
            f"on_tbl"
        )

        # s.add(ForAll([x, y, c], Implies(And(higher(x, y), higher(y, c)), higher(x, c))))
        # s.add(ForAll([x], higher(x, x)))
        # s.add(
            # ForAll(
                # [x, y, c],
                # Implies(
                    # And(higher(x, y), higher(x, c)), Or(higher(y, c), higher(c, y))
                # ),
            # )
        # )

    def add_axiom_on_star_zero(self, s: Solver):
        x, y, c = Consts("x y c", BoxSort)
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star_zero(x, y), ON_star_zero(y, c)), ON_star_zero(x, c)
                ),
            ),
            "on_zero_1",
        )
        s.assert_and_track(ForAll([x], ON_star_zero(x, x)), "on_zero_2")
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star_zero(x, y), ON_star_zero(x, c)),
                    Or(ON_star_zero(y, c), ON_star_zero(c, y)),
                ),
            ),
            "on_zero_3",
        )
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(ON_star_zero(x, c), ON_star_zero(y, c)),
                    Or(ON_star_zero(x, y), ON_star_zero(y, x)),
                ),
            ),
            "on_zero_4",
        )
        s.assert_and_track(
            ForAll(
                [x, y], Implies(ON_star_zero(x, y), Implies(ON_star_zero(y, x), x == y))
            ),
            "on_zero_5",
        )
        tbl, = Consts("tbl", BoxSort)
        s.assert_and_track(
            ForAll([x], Implies(Or(ON_star_zero(x, tbl), ON_star_zero(tbl, x)), x == tbl)),
            f"on_zero_tbl"
        )


    def add_reverse_loop_invariant(self, s: Solver, b0, b):
        x, y = Consts("x y", BoxSort)
        # s.assert_and_track(
        #     Not(
        #         ForAll(
        #             [x, y],
        #             Or(
        #                 And(ON_star(x, b0), ON_star(x, y) == ON_star_zero(x, y)),
        #                 And(
        #                     Not(ON_star(x, b0)),
        #                     ON_star(b, x),
        #                     ON_star(x, y) == ON_star_zero(y, x),
        #                 ),
        #             ),
        #         )
        #     ),
        #     "not_original_invariant",
        # )
        tbl, = Consts("tbl", BoxSort)
        s.assert_and_track(
            Not(
                ForAll(
                    [x,y],
                    Implies(
                        And(x != tbl, y != tbl),
                        Or(
                            And(x != b, ON_star(x, b0), ON_star(x, y) == ON_star_zero(x, y)),
                            And(Not(ON_star(x, b0)), b != tbl, ON_star(b, x), ON_star(x, y) == ON_star_zero(y, x))
                        )
                    )
                )
            ),
            "not_original_invariant_with_tbl"
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
            print(
                f"ON_star({name1}, {name2}): {s.model().evaluate(ON_star(box1, box2))}"
            )

    for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            print(f"higher({name1}, {name2}): {s.model().evaluate(higher(box1, box2))}")
