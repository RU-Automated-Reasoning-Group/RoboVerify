from z3 import (
    And,
    DeclareSort,
    Function,
    Implies,
    Not,
    Or,
    RealSort,
    Solver,
    sat,
    unsat,
)

BoxSort = DeclareSort("Box")
x = Function("x", BoxSort, RealSort())
y = Function("y", BoxSort, RealSort())
z = Function("z", BoxSort, RealSort())
