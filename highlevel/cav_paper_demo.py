from z3 import *

solver = Solver()

Node = DeclareSort('Node')
a, b, h = Consts("a b h", Node)
next_fun = Function('next', Node, Node, BoolSort())
next_star = TransitiveClosure(next_fun)

solver.add(next_fun(a, b))
solver.add(next_fun(b, h))
solver.add(Not(next_star(a, h)))
print(solver.check())