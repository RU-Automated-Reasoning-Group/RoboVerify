from z3 import *

# Create solver
solver = Solver()

# Declare sort for boxes
Box, (b0, b1, b2) = EnumSort('Box', ['b0', 'b1', 'b2'])
c, x, y = Consts('c x y', Box)

# Declare the uninterpreted function "ON"
ON = Function('ON', Box, Box, BoolSort())
solver.add(ForAll([x], Not(ON(x, x))))
solver.add(ForAll([x, y], Not(And( ON(x, y),  ON(y, x) ))))
solver.add(ForAll([x, y, c], Implies(And(ON(x, c), ON(y, c)), x == y)))

ON_star = Function("ON_star", Box, Box, BoolSort())
solver.add(ForAll([x, y], ON_star(x, y) == Or(x == y, Exists([c], And(ON(x, c), ON_star(c, y))))))

def top(x):
    return Not(Exists([c], ON(c, x)))

def on_or_top(b):
    return Or(ON_star(b, b0), top(b))

# precondition
solver.add([on_or_top(b) for b in [b0, b1, b2]])
solver.add(ON_star(b1, b0))
solver.add(top(b1))
solver.add(top(b2))


# # put b2 on b1
ON_after = Function('ON_after', Box, Box, BoolSort())
transition_rule = ForAll([x, y], 
    ON_after(x, y) == If(And(x == b2, y == b1), Not(ON(x, y)), ON(x, y))
)
solver.add(transition_rule)

ON_star_after = Function("ON_star_after", Box, Box, BoolSort())
solver.add(ForAll([x, y], ON_star_after(x, y) == Or(x == y, Exists([c], And(ON_after(x, c), ON_star_after(c, y))))))

# # postcondition
postcondition = And([ON_star_after(b, b0) for b in [b0, b1, b2]])
solver.add(Not(postcondition))


# Check satisfiability
if solver.check() == sat:
    print("SAT - The constraints are consistent!")
    print(solver.model())
    import pdb; pdb.set_trace()
    for id1, x in enumerate([b0, b1, b2]):
        for id2, y in enumerate([b0, b1, b2]):
            print(f"ON({id1}, {id2}) = {solver.model().eval(ON(x, y))}")

    print("=" * 50)
    for id1, x in enumerate([b0, b1, b2]):
        for id2, y in enumerate([b0, b1, b2]):
            print(f"ON_star({id1}, {id2}) = {solver.model().eval(ON_star(x, y))}")

    print("=" * 50)
    for id1, x in enumerate([b0, b1, b2]):
        for id2, y in enumerate([b0, b1, b2]):
            print(f"ON_after({id1}, {id2}) = {solver.model().eval(ON_after(x, y))}")

    print("=" * 50)
    for id1, x in enumerate([b0, b1, b2]):
        for id2, y in enumerate([b0, b1, b2]):
            print(f"ON_star_after({id1}, {id2}) = {solver.model().eval(ON_star_after(x, y))}")

else:
    print("UNSAT - The constraints are contradictory!")

