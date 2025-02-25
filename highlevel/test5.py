from z3 import *

# Create solver
solver = Solver()

# Declare sort for boxes
Box, (b0, b1, b2) = EnumSort('Box', ['b0', 'b1', 'b2'])
c, x, y = Consts('c x y', Box)

# Declare the uninterpreted function "ON"
ON_after = Function('ON_after', Box, Box, BoolSort())

solver.add(Not(ON_after(b0, b0)))
solver.add(ON_after(b1, b0))
solver.add(Not(ON_after(b2, b0)))

solver.add(Not(ON_after(b0, b1)))
solver.add(Not(ON_after(b1, b1)))
solver.add(ON_after(b2, b1))

solver.add(Not(ON_after(b0, b2)))
solver.add(Not(ON_after(b1, b2)))
solver.add(Not(ON_after(b2, b2)))


ON_trans_after = TransitiveClosure(ON_after)
ON_star_after = Function("ON_star_after", Box, Box, BoolSort())
solver.add(ForAll([x, y], ON_star_after(x, y) == Or(x == y, ON_trans_after(x, y))))

postcondition = ON_star_after(b2, b2)
# solver.add(postcondition)
solver.add(Not(postcondition))

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
            print(f"ON_trans({id1}, {id2}) = {solver.model().eval(ON_star(x, y))}")

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