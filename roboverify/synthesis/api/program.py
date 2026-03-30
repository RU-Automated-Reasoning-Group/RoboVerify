from copy import deepcopy
from typing import List, Union

from z3 import (
    Z3_OP_UNINTERPRETED,
    And,
    Const,
    Consts,
    Exists,
    ForAll,
    Implies,
    Not,
    Or,
    is_app,
    is_implies,
    is_quantifier,
    substitute,
)

import synthesis.inference_lib.inference
import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
import synthesis.verification_lib.lowlevel_verification_lib as lowlevel_verification_lib
from synthesis.api.instructions import (
    Assign,
    Instruction,
    PickPlace,
    PickPlaceByName,
    Put,
    Seq,
    Skip,
    While,
)


def rewrite_for_put_for_ON_star(expr, b_prime, b, context):
    """Rewrite every possible occurrence of alpha<on*> beta to
    Or(alpha<on*>beta, And(alpha<on*>b_prime, b<on*>beta))
    """
    # Case 1: Quantifier
    if is_quantifier(expr):
        # Extract info about the quantifier
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]

        # Extract and rewrite the body
        body = expr.body()
        rewritten_body = rewrite_for_put_for_ON_star(body, b_prime, b, context)

        # Rebuild the quantifier (keep same type)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        else:
            return Exists(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )

    # Case 2: Function application (And, Or, ON_star, etc.)
    elif is_app(expr):
        decl = expr.decl()

        # Match ON_star(a,b)
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "ON_star":
            alpha, beta = expr.children()
            return Or(
                context.ON_star(alpha, beta),
                And(
                    context.ON_star(alpha, b_prime),
                    context.ON_star(b, beta),
                ),
            )

        # Otherwise rebuild recursively
        new_children = [
            rewrite_for_put_for_ON_star(c, b_prime, b, context) for c in expr.children()
        ]
        return decl(*new_children)

    # Case 3: Constants, bound variables, etc.
    else:
        return expr


def rewrite_for_put_for_higher(expr, b_prime, b, context):
    """Rewrite every possible occurrence of alpha<higher> beta to
    Or(alpha<higher>beta, And(alpha<higher>b_prime, b<higher>beta))
    """
    # Case 1: Quantifier
    if is_quantifier(expr):
        # Extract info about the quantifier
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]

        # Extract and rewrite the body
        body = expr.body()
        rewritten_body = rewrite_for_put_for_higher(body, b_prime, b, context)

        # Rebuild the quantifier (keep same type)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        else:
            return Exists(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )

    # Case 2: Function application (And, Or, ON_star, etc.)
    elif is_app(expr):
        decl = expr.decl()

        # Match ON_star(a,b)
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "Higher":
            m, n = expr.children()
            t = Const("t", context.BoxSort)
            return Or(
                And(m != b_prime, m != b, n != b_prime, n != b, context.Higher(m, n)),
                And(
                    m != b_prime,
                    m != b,
                    n == b_prime,
                    Exists(
                        [t], And(t != n, context.Higher(n, t), context.Higher(t, b))
                    ),
                ),
                And(m != b_prime, m != b, n == b, context.Higher(m, n)),
                And(m == b, n != b_prime, n != b, context.Higher(m, n)),
                And(m == b, n == b),
                And(
                    m == b_prime,
                    n != b_prime,
                    n != b,
                    Or(
                        context.Higher(b, n),
                        ForAll(
                            [t],
                            And(
                                context.Higher(n, b),
                                Implies(
                                    And(t != n, context.Higher(n, t)),
                                    context.Higher(b, t),
                                ),
                            ),
                        ),
                    ),
                ),
                And(m == b_prime, n == b),
                And(m == b_prime, n == b_prime),
            )

        # Otherwise rebuild recursively
        new_children = [
            rewrite_for_put_for_higher(c, b_prime, b, context) for c in expr.children()
        ]
        return decl(*new_children)

    # Case 3: Constants, bound variables, etc.
    else:
        return expr


def rewrite_for_put_for_scattered(expr, b_prime, b, context):
    """Weakest-precondition rewrite for Scattered(m, n) after put(b', b).

    Verbose DNF (user specification): only pairs involving b' or b change;
    b' on b redirects scattered-with-b' to scattered-with-b.
    """
    if is_quantifier(expr):
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]
        body = expr.body()
        rewritten_body = rewrite_for_put_for_scattered(body, b_prime, b, context)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        return Exists(
            list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
            rewritten_body,
        )

    if is_app(expr):
        decl = expr.decl()
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "Scattered":
            m, n = expr.children()
            return Or(
                And(
                    m != b_prime,
                    m != b,
                    n != b_prime,
                    n != b,
                    context.Scattered(m, n),
                ),
                And(
                    m != b_prime,
                    m != b,
                    n == b,
                    context.Scattered(m, n),
                ),
                And(
                    m != b_prime,
                    m != b,
                    n == b_prime,
                    context.Scattered(m, b),
                ),
                And(
                    m == b_prime,
                    n != b_prime,
                    n != b,
                    context.Scattered(b, n),
                ),
                And(
                    m == b,
                    n != b_prime,
                    n != b,
                    context.Scattered(m, n),
                ),
            )
        new_children = [
            rewrite_for_put_for_scattered(c, b_prime, b, context)
            for c in expr.children()
        ]
        return decl(*new_children)

    return expr


Stmt = Union[Instruction, While]


class Program:
    length: int
    instructions: List[Stmt]

    def __init__(self, length: int, instructions: Union[List[Stmt], None] = None):
        self.length = length
        if instructions:
            assert len(instructions) == length
            self.instructions = deepcopy(instructions)
        else:
            self.instructions = [Skip() for _ in range(self.length)]

    def eval(
        self,
        env,
        return_img: bool = False,
        symbolic_name_to_box_id: dict[str, int] | None = None,
    ):
        """evaluate the program in the environment and return the trajectories"""
        old_mapping = getattr(env, "symbolic_name_to_box_id", None)
        if symbolic_name_to_box_id is not None:
            env.symbolic_name_to_box_id = symbolic_name_to_box_id
        traj = [env.reset()[0]]
        if return_img:
            imgs = [env.render()]
        for line in self.instructions:
            line_imgs = line.eval(env, traj, return_img)
            if return_img:
                imgs.extend(line_imgs)
        if symbolic_name_to_box_id is not None:
            if old_mapping is None:
                delattr(env, "symbolic_name_to_box_id")
            else:
                env.symbolic_name_to_box_id = old_mapping
        if return_img:
            return traj, imgs
        return traj

    def register_trainable_parameter(self):
        parameters = []
        for line in self.instructions:
            line.register_trainable_parameter(parameters)
        return parameters

    def update_trainable_parameter(self, new_parameter):
        for line in self.instructions:
            line.update_trainable_parameter(new_parameter)

    def __str__(self):
        instruction_str = [f"\t{inst}" for inst in self.instructions]
        return "\n".join(["begin", *instruction_str, "end"])

    def VC_gen(self, P, Q, context):
        # P, Q are z3 formula
        seq_instruction = to_seq(self.instructions)
        return [Implies(P, self.wp(Q, context))] + VC_aux(seq_instruction, Q, context)

    def wp(self, Q, context):
        seq_instruction = to_seq(self.instructions)
        return wp(seq_instruction, Q, context)

    def highlevel_verification(
        self,
        P,
        Q,
        context: Union[highlevel_verification_lib.HighLevelContext, None] = None,
        use_tbl: bool = False,
        box_sort_mode: str = "declare",
        num_blocks: Union[int, None] = None,
        enum_names: Union[List[str], None] = None,
        visualize_enum_scene: bool = False,
        visualization_prefix: str = "highlevel_scene",
    ):
        solver = (
            context
            if context is not None
            else highlevel_verification_lib.HighLevelContext(
                mode=box_sort_mode,
                num_blocks=num_blocks,
                enum_names=enum_names,
                use_tbl=use_tbl,
                visualize_enum_scene=visualize_enum_scene,
                visualization_prefix=visualization_prefix,
            )
        )
        vcs = self.VC_gen(P, Q, solver)
        print("testing axioms")
        solver.start_verification()
        print("=====================")

        print("total number of VCs:", len(vcs))
        for idx, vc in enumerate(vcs):
            print(f"verifying VC {idx}", vc)
            if is_implies(vc):
                premise = vc.arg(0)
                conclusion = vc.arg(1)
                print("check 1: axioms + premise")
                solver.check_satisfiable(
                    premise,
                    visualize_model=True,
                    viz_tag=f"vc_{idx}_check1",
                )
                print("---------------------")
                print("check 2: axioms + premise + not(conclusion)")
                solver.check_satisfiable(
                    And(premise, Not(conclusion)),
                    visualize_model=True,
                    viz_tag=f"vc_{idx}_check2",
                )
            else:
                print("non-implication VC; fallback to original check axioms + not(VC)")
                solver.start_verification(vc, viz_tag=f"vc_{idx}")
            print("=====================")

    def lowlevel_verification(
        self,
        context: Union[lowlevel_verification_lib.LowLevelContext, None] = None,
        sort_name: str = "Box",
    ):
        solver = (
            context
            if context is not None
            else lowlevel_verification_lib.LowLevelContext(sort_name=sort_name)
        )
        for idx, inst in enumerate(self.instructions):
            if isinstance(inst, While):
                print(
                    f"starting low-level verification for while loop with index {idx}"
                )
                print(f"invariant: {inst.invariant}")
                print(f"body: {inst.body}")
                solver.start_verification(
                    inst.invariant, inst.body, constants=["b0", "b", "b_prime"]
                )


def to_seq(instructions):
    """convert a list of instructions to Seq connected expression (Seq is left associate)
    That is x;y;z is equivalent to ((x;y);z)
    """
    if len(instructions) == 1:
        return instructions[0]
    elif len(instructions) > 1:
        return Seq(to_seq(instructions[:-1]), instructions[-1])
    else:
        assert False, "unable to convert to seq for empty instructions"


def wp(seq_instruction, Q, context):
    """calculate weakest precondition"""
    if isinstance(seq_instruction, Skip):
        return Q
    elif isinstance(seq_instruction, Seq):
        return wp(seq_instruction.s1, wp(seq_instruction.s2, Q, context), context)
    elif isinstance(seq_instruction, While):
        return seq_instruction.invariant
    elif isinstance(seq_instruction, Assign):
        return substitute(
            Q,
            (
                context.get_consts(seq_instruction.left),
                context.get_consts(seq_instruction.right),
            ),
        )
    elif isinstance(seq_instruction, Put):
        b_prime = context.get_consts(seq_instruction.upper_block)
        b = context.get_consts(seq_instruction.base_block)
        Q = And(
            Not(context.ON_star(b, b_prime)),
            rewrite_for_put_for_ON_star(Q, b_prime, b, context),
        )
        Q = rewrite_for_put_for_higher(Q, b_prime, b, context)
        return rewrite_for_put_for_scattered(Q, b_prime, b, context)
    assert (
        False
    ), f"Unrecognized seq instruction {type(seq_instruction)} to calculate wp"


def VC_aux(seq_instruction, Q, context) -> List:
    """generate auxiliary verification conditions"""
    if isinstance(seq_instruction, Seq):
        return VC_aux(
            seq_instruction.s1, wp(seq_instruction.s2, Q, context), context
        ) + VC_aux(seq_instruction.s2, Q, context)
    elif isinstance(seq_instruction, Instruction):
        return []
    elif isinstance(seq_instruction, While):
        return VC_aux(
            to_seq(seq_instruction.body), seq_instruction.invariant, context
        ) + [
            Implies(
                And(seq_instruction.instantiated_cond, seq_instruction.invariant),
                wp(to_seq(seq_instruction.body), seq_instruction.invariant, context),
            ),
            Implies(And(Not(seq_instruction.cond), seq_instruction.invariant), Q),
        ]
    assert False, "Unrecognized seq instruction for VC_aux"


def run_stack_example_with_only_ON_star():
    context = highlevel_verification_lib.HighLevelContext(mode="declare")
    inferred_invariant, candidate_lists = (
        synthesis.inference_lib.inference.run_proposal_example(context=context)
    )
    b_prime, b, n, b0, a = Consts("b_prime b n b0 a", context.BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            instantiated_cond=And(
                ForAll([n], Or(b_prime == n, Not(context.ON_star(n, b_prime)))),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=inferred_invariant,
            # invariant=And(*candidate_lists)
        ),
    ]
    p = Program(2, instructions=instructions)

    m, n = Consts("m n", context.BoxSort)
    precondition = ForAll(
        [m],
        ForAll([n], Or(m == n, Not(context.ON_star(n, m)))),
    )

    m, n, b0 = Consts("m n b0", context.BoxSort)
    postcondition = ForAll([m], context.ON_star(m, b0))

    # print(p.VC_gen(precondition, postcondition))
    p.highlevel_verification(precondition, postcondition, context=context)


def run_unstack_example():
    pass
