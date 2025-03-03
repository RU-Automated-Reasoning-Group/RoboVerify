from z3 import *

# Create solver
solver = Solver()

# Declare sort for boxes
Box, (b0, b1, b2) = EnumSort('Box', ['b0', 'b1', 'b2'])
c, x, y = Consts('c x y', Box)

ON = Function('ON', Box, Box, BoolSort())
solver.add(ForAll([x], ON(x, x)))
solver.add(ForAll([x, y], Implies(x != y, Not(And( ON(x, y),  ON(y, x) )))))
solver.add(ForAll([x, y, c], Implies(And(ON(x, c), ON(y, c), x != c, y != c), x == y)))
ON_trans = TransitiveClosure(ON)

def top(x):
    return Not(Exists([c], And(Not(c == x), ON(c, x))))

def on_or_top(b):
    return Or(ON_trans(b, b0), top(b))

solver.add([on_or_top(b) for b in [b0, b1, b2]])
# solver.add(ON_trans(b1, b0))
solver.add(top(b0))
solver.add(top(b1))
solver.add(top(b2))

# solver.add(ON(b0, b0))
ON_after = Function('ON_after', Box, Box, BoolSort())
transition_rule = ForAll([x, y], 
    ON_after(x, y) == If(And(x == b1, y == b0), Not(ON(x, y)), ON(x, y))
)
solver.add(transition_rule)
ON_trans_after = TransitiveClosure(ON_after)

def eval_one(solver, boxes, fun, fun_str):
    print("=" * 50)
    for id1, x in enumerate(boxes):
        for id2, y in enumerate(boxes):
            print(f"{fun_str}({id1}, {id2}) = {solver.model().eval(fun(x, y))}")

def eval_all(solver, boxes, ON, ON_trans, ON_after, ON_trans_after):
    eval_one(solver, boxes, ON, "ON")
    eval_one(solver, boxes, ON_trans, "ON_trans")
    eval_one(solver, boxes, ON_after, "ON_after")
    eval_one(solver, boxes, ON_trans_after, "ON_trans_after")


for id1, x in enumerate([b0, b1, b2]):
    for id2, y in enumerate([b0, b1, b2]):
        solver.push()
        postcondition = ON_trans_after(x, y)
        solver.add(postcondition)

        # Check satisfiability
        print(f"checking ON_trans_after({id1}, {id2}):")
        if solver.check() == sat:
            print("SAT")
            # eval_all(solver, [b0, b1, b2], ON, ON_trans, ON_after, ON_trans_after)
            # print(solver.model())
        else:
            print("UNSAT")
        
        solver.pop()

        solver.push()
        solver.add(Not(postcondition))
        print(f"checking NOT ON_trans_after({id1}, {id2}):")
        if solver.check() == sat:
            print("SAT")
            # eval_all(solver, [b0, b1, b2], ON, ON_trans, ON_after, ON_trans_after)
            # print(solver.model())
        else:
            print("UNSAT")
        
        solver.pop()


