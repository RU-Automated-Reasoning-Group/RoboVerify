from z3 import *

# Create solver
solver = Solver()

# Declare sort for boxes
Box, (b0, b1, b2) = EnumSort('Box', ['b0', 'b1', 'b2'])
c, x, y = Consts('c x y', Box)

# Declare the uninterpreted function "ON"
ON_after = Function('ON_after', Box, Box, BoolSort())

solver.add(ON_after(b0, b0))
solver.add(ON_after(b1, b0))         # ON(b1, b0)
solver.add(Not(ON_after(b2, b0)))    # Not ON(b2, b0)

solver.add(Not(ON_after(b0, b1)))    # Not ON(b0, b1)
solver.add(ON_after(b1, b1))
solver.add(ON_after(b2, b1))         # ON(b2, b1)

solver.add(Not(ON_after(b0, b2)))    # Not ON(b0, b2)
solver.add(Not(ON_after(b1, b2)))    # Not ON(b1, b2))
solver.add(ON_after(b2, b2))

# Use the built-in TransitiveClosure function
ON_trans_after = TransitiveClosure(ON_after)

for id1, x in enumerate([b0, b1, b2]):
    for id2, y in enumerate([b0, b1, b2]):
        solver.push()
        postcondition = ON_trans_after(x, y)
        solver.add(postcondition)

        # Check satisfiability
        print(f"checking ON_trans_after({id1}, {id2}):")
        if solver.check() == sat:
            print("SAT")
            # print(solver.model())
        else:
            print("UNSAT")
        
        solver.pop()

        solver.push()
        solver.add(Not(postcondition))
        print(f"checking NOT ON_trans_after({id1}, {id2}):")
        if solver.check() == sat:
            print("SAT")
            # print(solver.model())
        else:
            print("UNSAT")
        
        solver.pop()
        