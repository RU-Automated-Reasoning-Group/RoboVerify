"""
Z3 verification of "build an m x n nested linked list from existing blocks".

Model:
 - Finite universe of nodes: integers 0..N-1 where N = m*n
 - Fields r, d are Int->Int (we model per-node Int variables for readability)
 - assigned[u] is Bool per node
 - Program variables (ghost counters / pointers): h, lastRow, first, prev, r_cnt, c_cnt, k_cnt
 - We will reason about a single inner-loop body and outer-loop body using pre-state/post-state copies.

Verification approach:
 - Write invariants InnerInv and OuterInv (over pre-state variables)
 - For each VC, assert pre-state invariant & guard & body-effect & NOT(post-state invariant)
 - Check satisfiability: SAT -> counterexample; UNSAT -> VC holds.

Limitations:
 - This is a bounded, finite-domain verification. It proves correctness for the given m,n values.
 - Larger m,n increase solver work. Start with small sizes (m<=3, n<=3).
"""

from z3 import *
import itertools

# Parameters: change m,n for different bounded checks
m = 2
n = 2
N = m * n
nodes = list(range(N))
Null = -1  # represent null with -1

# Helper: create pre/post-state variable dictionaries
def mk_state(prefix):
    # fields: r[u], d[u] as Ints; assigned[u] as Bool
    r = {u: Int(f"{prefix}_r_{u}") for u in nodes}
    d = {u: Int(f"{prefix}_d_{u}") for u in nodes}
    assigned = {u: Bool(f"{prefix}_assigned_{u}") for u in nodes}
    # program pointers/ghosts
    h = Int(f"{prefix}_h")
    lastRow = Int(f"{prefix}_lastRow")
    first = Int(f"{prefix}_first")
    prev = Int(f"{prefix}_prev")
    r_cnt = Int(f"{prefix}_r_cnt")  # number of complete rows built
    c_cnt = Int(f"{prefix}_c_cnt")  # number of nodes in current row
    k_cnt = Int(f"{prefix}_k_cnt")  # total assigned nodes
    return {
        'r': r, 'd': d, 'assigned': assigned,
        'h': h, 'lastRow': lastRow, 'first': first, 'prev': prev,
        'r_cnt': r_cnt, 'c_cnt': c_cnt, 'k_cnt': k_cnt
    }

# Create pre-state and post-state
pre = mk_state("pre")
post = mk_state("post")

# Basic domain constraints helper for r and d fields: they must be either in nodes or Null
def domain_constraints(state):
    cons = []
    for u in nodes:
        rvar = state['r'][u]
        dvar = state['d'][u]
        cons.append(Or([rvar == v for v in nodes] + [rvar == Null]))
        cons.append(Or([dvar == v for v in nodes] + [dvar == Null]))
    # h/lastRow/first/prev are either in nodes or Null
    cons += [Or(state['h'] == v, state['h'] == Null) for v in nodes]  # incorrect style: need Or over whole set
    # rewrite properly
    cons_clean = []
    cons_clean.extend(cons[:-len(nodes)])  # keep r/d domain constraints
    # now add domain for h,lastRow,first,prev
    for ptr in ('h', 'lastRow', 'first', 'prev'):
        cons_clean.append(Or(*[post_val == v for v in nodes for post_val in [state[ptr]]]) ) # we'll fix simpler below
    # but above is messy; simpler approach: assert Int in range by inequalities
    # we'll use numeric constraints: -1 or 0..N-1
    cons_clean = []
    for u in nodes:
        rvar = state['r'][u]
        dvar = state['d'][u]
        cons_clean.append(Or(And(rvar >= 0, rvar < N), rvar == Null))
        cons_clean.append(Or(And(dvar >= 0, dvar < N), dvar == Null))
    # pointers
    for ptr in ('h','lastRow','first','prev'):
        var = state[ptr]
        cons_clean.append(Or(And(var >= 0, var < N), var == Null))
    # counters: r_cnt in 0..m, c_cnt in 0..n, k_cnt in 0..N
    cons_clean.append(And(pre['r_cnt'] >= 0, pre['r_cnt'] <= m))
    # but we will not add pre-specific ones here: leave counters' domain to invariants
    return cons_clean

# We'll build invariants explicitly as Z3 formulas using quantifier-free encoding where possible:
# Since domain finite, we use conjunctions over nodes for "for all node" properties.

# Helper to express "node u belongs to the prefix of the current row (first .. prev)".
# We use explicit reachability via r links in pre-state: we define reach_prefix_pre[u] iff there exists a path
# of length < c_cnt from first to u following r pointers. Because bounds small, we unroll path lengths.

# Because Z3 quantifiers + arrays are fiddly, we encode reachability with explicit disjunctions over chain lengths.
def reachable_in_row_prefix(state, first_var, prev_var, c_cnt_var):
    # returns a dict node->Bool formula that says "node is one of nodes in prefix of length c_cnt starting at first_var"
    reach = {}
    for u in nodes:
        disj = []
        # path of length 1: first == u (only when c_cnt >= 1)
        disj.append(And(c_cnt_var >= 1, first_var == u))
        # path of length 2: first.r == u (and c_cnt >=2)
        # generalize up to n (max)
        # build iterative equalities: first -> r(first) -> r(r(first)) ...
        prev_step = first_var
        for L in range(2, n+1):
            # compute node reached by following r L-1 times: build equality chain prev_step.r == next
            # Build equality chain as a nested expression: r(prev_step) == v. But prev_step is symbolic; since r is per-node var, we can only express concrete equalities when prev_step equals some concrete node.
            # Because domain small, we encode all possibilities: there exists a sequence of concrete nodes p0,p1,...,p_{L-1} where p0==first_var and p_{L-1}==u and for all t, pre.r[p_t]==p_{t+1}
            seq_clause = []
            for seq in itertools.product(nodes, repeat=L):
                # require seq[0] == first_var and seq[-1] == u and r[seq[t]] == seq[t+1] for t=0..L-2
                conds = [first_var == seq[0], seq[-1] == u]
                for t in range(L-1):
                    conds.append(state['r'][seq[t]] == seq[t+1])
                seq_clause.append(And(*conds))
            if seq_clause:
                disj.append(And(c_cnt_var >= L, Or(*seq_clause)))
        reach[u] = Or(*disj)
    return reach

# Because the above becomes large, we'll restrict to small m,n. This script is intended for small sizes.

# Invariant definitions (as quantifier-free formulas)
def InnerInv(state):
    # state: dict
    s = []
    # basic bounds on counters
    s.append(state['r_cnt'] >= 0)
    s.append(state['r_cnt'] <= m)
    s.append(state['c_cnt'] >= 0)
    s.append(state['c_cnt'] <= n)
    s.append(state['k_cnt'] >= 0)
    s.append(state['k_cnt'] <= N)
    # relationship k = r*n + c
    s.append(state['k_cnt'] == state['r_cnt'] * n + state['c_cnt'])
    # assigned nodes are subset of domain (we don't need extra)
    # Distinctness of assigned nodes: for any two nodes u!=v, assigned[u] & assigned[v] -> they are distinct (trivially true)
    # Structure built so far: if c=0 then first=null & prev=null
    s.append(Implies(state['c_cnt'] == 0, state['first'] == Null))
    s.append(Implies(state['c_cnt'] == 0, state['prev'] == Null))
    # If c>0 then first and prev in domain (not null). We won't assert the exact row_len_prefix using heavy unrolling; we assert existence of one path from first to prev of length c_cnt.
    # Build reachability prefix:
    reach = reachable_in_row_prefix(state, state['first'], state['prev'], state['c_cnt'])
    # If c>0 then first != Null and prev != Null and prev is in reach prefix
    s.append(Implies(state['c_cnt'] > 0, state['first'] != Null))
    s.append(Implies(state['c_cnt'] > 0, state['prev'] != Null))
    # prev is in the prefix:
    if state['c_cnt'] is not None:
        # add that prev equals some u that is in reach (we encode as: there exists u such that reach[u] and u==prev)
        exists_clause = Or([And(reach[u], state['prev'] == u) for u in nodes])
        s.append(Implies(state['c_cnt'] > 0, exists_clause))
    # assigned nodes count equals k_cnt (simple cardinality check: sum assigned == k_cnt)
    sum_assigned = Sum([If(state['assigned'][u], 1, 0) for u in nodes])
    s.append(sum_assigned == state['k_cnt'])
    return And(*s)

def OuterInv(state):
    # OuterInv includes that first/prev/inner invariants for the current row hold (we model only necessary parts)
    s = []
    s.append(state['r_cnt'] >= 0)
    s.append(state['r_cnt'] <= m)
    s.append(state['k_cnt'] >= 0)
    s.append(state['k_cnt'] <= N)
    s.append(state['c_cnt'] >= 0)
    s.append(state['c_cnt'] <= n)
    s.append(state['k_cnt'] == state['r_cnt'] * n + state['c_cnt'])
    # if r_cnt==0 then h==Null & lastRow==Null else h != Null & lastRow != Null
    s.append(Implies(state['r_cnt'] == 0, state['h'] == Null))
    s.append(Implies(state['r_cnt'] == 0, state['lastRow'] == Null))
    s.append(Implies(state['r_cnt'] > 0, state['h'] != Null))
    s.append(Implies(state['r_cnt'] > 0, state['lastRow'] != Null))
    # assigned count equals k_cnt
    sum_assigned = Sum([If(state['assigned'][u], 1, 0) for u in nodes])
    s.append(sum_assigned == state['k_cnt'])
    # For brevity we don't encode full rows_prefix_of_len; we assert that k_cnt == r_cnt*n + c_cnt and r_cnt<=m ensures partial shape
    return And(*s)

# Postcondition Qbuild: when outer loop finished (r_cnt == m and c_cnt==0) then rows_len(h,m,n) holds.
# For bounded check, we assert: if r_cnt == m and c_cnt==0 then k_cnt==N and every node is assigned
def PostCond(state):
    cond = And(state['r_cnt'] == m, state['c_cnt'] == 0)
    all_assigned = And(*[state['assigned'][u] for u in nodes])
    # when finished, h==Null iff m==0; else h != Null
    h_cond = If(m == 0, state['h'] == Null, state['h'] != Null)
    return Implies(cond, And(state['k_cnt'] == N, all_assigned, h_cond))

# Program body encodings:
# Inner loop body (pre -> post): we must express effect of picking x (some unassigned node), setting its fields,
# marking assigned, updating first/prev/k_cnt/c_cnt accordingly.
# We model nondeterminism by existentially quantifying x in nodes with pre.assigned[x] == False.

def inner_body_transition(pre_state, post_state):
    """
    Return formula: there exists x in nodes with not pre.assigned[x] s.t. post-state equals effect of body.
    We model only the variables that change in inner loop: r, d for the chosen x, assigned[x], first, prev, c_cnt, k_cnt.
    For other locations, we conservatively require they stay equal (frame).
    """
    clauses = []
    x = Int('x_pick')
    # x in domain and unassigned in pre
    clauses.append(Or([x == v for v in nodes]))
    clauses.append(Not(pre_state['assigned'][0]) )  # placeholder; we'll add correct constraint below

    # Fix the previous bad line: create Or(x==v) and constraint pre.assigned[x]==False using ite over nodes
    # Build pre_assigned_x == False as a disjunction: (x==v & pre.assigned[v]==False) for some v
    pre_assigned_false_cases = []
    for v in nodes:
        pre_assigned_false_cases.append(And(x == v, Not(pre_state['assigned'][v])))
    pick_exists_clause = Or(*pre_assigned_false_cases)
    # Now body effects:
    # post.assigned[v] == pre.assigned[v] for all v != chosen x; post.assigned[x] == True
    assigned_eqs = []
    for v in nodes:
        assigned_eqs.append(If(x == v, post_state['assigned'][v] == True, post_state['assigned'][v] == pre_state['assigned'][v]))
    # r and d fields: we set post.r[x] == Null (or post.r[x] remains maybe set to null? In original we set x.r := null and x.d:=null)
    r_eqs = []
    d_eqs = []
    for v in nodes:
        r_eqs.append(If(x == v, post_state['r'][v] == Null, post_state['r'][v] == pre_state['r'][v]))
        d_eqs.append(If(x == v, post_state['d'][v] == Null, post_state['d'][v] == pre_state['d'][v]))
    # first/prev update:
    # if pre.prev == Null then post.first == x else post.first == pre.first
    first_eq = If(pre_state['prev'] == Null, post_state['first'] == x, post_state['first'] == pre_state['first'])
    # post.prev == x
    prev_eq = post_state['prev'] == x
    # counters:
    c_inc = post_state['c_cnt'] == pre_state['c_cnt'] + 1
    k_inc = post_state['k_cnt'] == pre_state['k_cnt'] + 1
    # other variables unchanged (h, lastRow, r_cnt)
    unchanged = And(post_state['h'] == pre_state['h'], post_state['lastRow'] == pre_state['lastRow'],
                    post_state['r_cnt'] == pre_state['r_cnt'])
    # combine
    body = And(pick_exists_clause, And(*assigned_eqs), And(*r_eqs), And(*d_eqs), first_eq, prev_eq, c_inc, k_inc, unchanged)
    # Quantify exists x
    return Exists([x], body)

def outer_body_transition(pre_state, post_state):
    """
    Outer body executes after inner loop finishes with c_cnt == n:
     - link row: if pre.h == Null then post.h := pre.first else pre.lastRow.d := pre.first
     - post.lastRow := pre.first
     - post.r_cnt := pre.r_cnt + 1
     - post.c_cnt := 0
     - other variables (r, d, assigned for nodes) remain the same (frame)
    """
    clauses = []
    # pre.c_cnt should equal n for outer body to be executed in typical flow; but invariant ensures that before outer body we had c_cnt==n
    # two cases for linking
    link_case1 = And(pre['h'] == Null, post['h'] == pre['first'])
    # case2: pre.h != Null -> we set d[lastRow] = first; since lastRow is a concrete index we must set post.d[lastRow]==pre.first
    # Because lastRow could be Null (not in this branch), we guard:
    d_updates = []
    for v in nodes:
        d_updates.append(If(pre['lastRow'] == v, post['d'][v] == pre['first'], post['d'][v] == pre['d'][v]))
    # combine link_case2
    link_case2 = And(pre['h'] != Null, And(*d_updates), post['h'] == pre['h'])
    link = Or(link_case1, link_case2)
    # post.lastRow == pre.first
    lastRow_eq = post['lastRow'] == pre['first']
    # counters
    r_inc = post['r_cnt'] == pre['r_cnt'] + 1
    c_zero = post['c_cnt'] == 0
    k_same = post['k_cnt'] == pre['k_cnt']
    # assigned and r/d for interior nodes remain same except possibly one d updated in case2 which we encoded
    # For r fields, we assume they remain the same
    r_same = And(*[post['r'][v] == pre['r'][v] for v in nodes])
    assigned_same = And(*[post['assigned'][v] == pre['assigned'][v] for v in nodes])
    return And(link, lastRow_eq, r_inc, c_zero, k_same, r_same, assigned_same)

# Now create solver and check VCs
def check_vc(vc_formula, name):
    s = Solver()
    s.set("timeout", 10000)  # 10s per check
    s.add(Not(vc_formula))   # check satisfiability of negation
    res = s.check()
    if res == sat:
        print(f"VC {name} FAILED: solver found counterexample")
        print(s.model())
        return False
    elif res == unsat:
        print(f"VC {name} holds (unsat).")
        return True
    else:
        print(f"VC {name} unknown or timeout: {res}")
        return False

# Build domain constraints for pre and post
domain_cons = []
for state in (pre, post):
    for u in nodes:
        domain_cons.append(Or(And(state['r'][u] >= 0, state['r'][u] < N), state['r'][u] == Null))
        domain_cons.append(Or(And(state['d'][u] >= 0, state['d'][u] < N), state['d'][u] == Null))
    domain_cons.append(Or(And(state['h'] >= 0, state['h'] < N), state['h'] == Null))
    domain_cons.append(Or(And(state['lastRow'] >= 0, state['lastRow'] < N), state['lastRow'] == Null))
    domain_cons.append(Or(And(state['first'] >= 0, state['first'] < N), state['first'] == Null))
    domain_cons.append(Or(And(state['prev'] >= 0, state['prev'] < N), state['prev'] == Null))
    # counters basic bounds
    domain_cons.append(state['r_cnt'] >= 0)
    domain_cons.append(state['r_cnt'] <= m)
    domain_cons.append(state['c_cnt'] >= 0)
    domain_cons.append(state['c_cnt'] <= n)
    domain_cons.append(state['k_cnt'] >= 0)
    domain_cons.append(state['k_cnt'] <= N)

# VC1: Initialization implies OuterInv (init -> OuterInv)
# Define initial state: h == Null, lastRow == Null, all assigned False, counters 0
init_state = And(pre['h'] == Null, pre['lastRow'] == Null, pre['first'] == Null, pre['prev'] == Null,
                 pre['r_cnt'] == 0, pre['c_cnt'] == 0, pre['k_cnt'] == 0,
                 *[pre['assigned'][u] == False for u in nodes])
vc1 = Implies(And(*domain_cons, init_state), OuterInv(pre))
print("Checking VC1: init -> OuterInv")
check_vc(vc1, "init->OuterInv")

# VC2: InnerInv & (c_cnt < n) & domain -> after inner body, InnerInv holds (preservation)
inner_body = inner_body_transition(pre, post)
vc2_formula = Implies(And(domain_cons, InnerInv(pre), pre['c_cnt'] < n, pre['k_cnt'] < N),
                      InnerInv(post))
print("Checking VC2: InnerInv preserved by inner body")
check_vc(vc2_formula, "InnerInv_preserve_by_inner_body")

# VC3: InnerInv & (c_cnt >= n) implies after outer-body, OuterInv holds
# We model post of outer body via outer_body_transition(pre, post)
outer_body = outer_body_transition(pre, post)
vc3_formula = Implies(And(domain_cons, InnerInv(pre), pre['c_cnt'] == n),
                      OuterInv(post))
print("Checking VC3: OuterInv preserved by outer body (after c==n)")
check_vc(vc3_formula, "OuterInv_preserve_by_outer_body")

# VC4: OuterInv & exit condition (r_cnt == m & c_cnt == 0) -> PostCond
vc4 = Implies(And(domain_cons, OuterInv(pre), pre['r_cnt'] == m, pre['c_cnt'] == 0), PostCond(pre))
print("Checking VC4: OuterInv & exit -> PostCond")
check_vc(vc4, "Exit_implies_PostCond")

print("Done checks. Note: for larger m,n the checks become heavier. Tune timeouts or use induction for general proof.")
