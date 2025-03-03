from z3 import *

# Define sorts (types)
Node = DeclareSort('Node')

# Define a relation representing edges in a graph
edge = Function('edge', Node, Node, BoolSort())

# Define the transitive closure of the edge relation
tc_edge = TransitiveClosure(edge)

# Create nodes
a, b, c = Consts('a b c', Node)

# Create a solver
s = Solver()

# Add some edges
s.add(edge(a, b))  # a -> b
s.add(edge(b, c))  # b -> c

# Query: Is a reachable from c?
s.add(Not(tc_edge(a, a)))  # If this is unsat, then a can reach c

# Check satisfiability
if s.check() == unsat:
    print("Yes, a can reach c")
else:
    print("No, a cannot reach c")
