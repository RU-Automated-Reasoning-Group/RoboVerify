from z3 import (
    And,
    sat,
    unsat,
    Solver,
    And,
    Or,
    Not,
    Implies,
    Function,
    DeclareSort,
    RealSort
)

BoxSort = DeclareSort("Box")
x = Function("x", BoxSort, RealSort())
y = Function("y", BoxSort, RealSort())
z = Function("z", BoxSort, RealSort())