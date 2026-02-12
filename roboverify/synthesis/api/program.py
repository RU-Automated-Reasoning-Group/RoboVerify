import pdb
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union

import numpy as np
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
    is_quantifier,
    substitute,
)

import synthesis.inference_lib.inference
import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.verification_lib.highlevel_verification_lib import (
    BoxSort,
    ON_star,
    higher,
)


class Parameter:
    def __init__(self, val: float = 0):
        self.pos: int | None = None
        self.val: float = val

    def register(self, parameters: List):
        self.pos = len(parameters)
        parameters.append(self.val)

    def update(self, new_parameter: List[float]):
        if self.pos is not None:
            self.val = new_parameter[self.pos]
        else:
            pdb.set_trace()
            raise ValueError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return False
        return self.pos == other.pos and self.val == other.val

    def __str__(self):
        return f"{self.val:.3}"


class Instruction(ABC):
    @abstractmethod
    def eval(self, env, traj, return_image=False) -> List:
        pass

    def register_trainable_parameter(self, parameters: List):
        pass

    def update_trainable_parameter(self, new_parameter: List):
        pass

    def get_operand(self):
        return []

    def set_operand(self, new_operands):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Skip(Instruction):
    def __init__(self, skip_steps: int = 20):
        self.skip_steps = skip_steps

    def eval(self, env, traj, return_img=False):
        imgs = []
        for _ in range(self.skip_steps):
            obs = env.flatten_observation(env.env._get_obs())
            if return_img:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def __str__(self):
        return "Skip"


class PickPlace(Instruction):
    def __init__(self, grab_box_id: int = 0, target_box_id: int = 0, limit: int = 50):
        self.limit = limit
        self.grab_box_id = grab_box_id
        self.target_box_id = target_box_id
        self.types = ["Box", "Box"]
        self.target_offset = [
            Parameter(0.0) for _ in range(3)
        ]  # target offset is applied to goal_box
        # self.target_offset = [Parameter(-0.0), Parameter(-0.2), Parameter(0.08)]  # target offset is applied to goal_box

    def get_box_pos(self, box_id, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        else:
            assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_img=False):
        # call the neural controller here
        from synthesis.environment.data.pickplace_naive import get_pick_control_naive

        imgs = []
        success = False
        initial_goal_box = self.get_box_pos(self.target_box_id, traj[-1])
        step = 0
        while not success and step < self.limit:
            obs = env.flatten_observation(env.env._get_obs())
            # import pdb; pdb.set_trace()
            action, success = get_pick_control_naive(
                obs,
                initial_goal_box
                + np.array([offset.val for offset in self.target_offset]),
                block_id=self.grab_box_id,
                last_block=True,
            )
            env.step(action)
            step += 1
            if return_img:
                imgs.append(env.render())
            # print(step, success)
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        for p in self.target_offset:
            p.update(new_parameter)

    def get_operand(self):
        return [
            {"type": self.types[0], "val": self.grab_box_id},
            {"type": self.types[1], "val": self.target_box_id},
        ]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "Box"
        assert new_operands[1]["type"] == "Box"
        self.grab_box_id = new_operands[0]["val"]
        self.target_box_id = new_operands[1]["val"]

    def __eq__(self, other):
        if not isinstance(other, PickPlace):
            return False
        cond1 = self.get_operand() == other.get_operand()
        cond2 = self.target_offset == other.target_offset
        return cond1 and cond2

    def __str__(self):
        return f"PickPlace({self.grab_box_id}, {self.target_box_id}, {[str(x) for x in self.target_offset]})"


class While:
    def __init__(self, cond, instantiated_cond, body, invariant):
        self.cond = cond
        self.instantiated_cond = instantiated_cond
        self.body = body
        self.invariant = invariant


class Put(Instruction):
    def __init__(self, upper_block, base_block):
        self.base_block = base_block
        self.upper_block = upper_block

    def __str__(self):
        return f"put({self.base_block}, {self.upper_block})"

    def eval(self, env, traj, return_image=False) -> List:
        pass


def rewrite_for_put_for_ON_star(expr, b_prime, b):
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
        rewritten_body = rewrite_for_put_for_ON_star(body, b_prime, b)

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
                ON_star(alpha, beta), And(ON_star(alpha, b_prime), ON_star(b, beta))
            )

        # Otherwise rebuild recursively
        new_children = [
            rewrite_for_put_for_ON_star(c, b_prime, b) for c in expr.children()
        ]
        return decl(*new_children)

    # Case 3: Constants, bound variables, etc.
    else:
        return expr


def rewrite_for_put_for_higher(expr, b_prime, b):
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
        rewritten_body = rewrite_for_put_for_higher(body, b_prime, b)

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
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "higher":
            alpha, beta = expr.children()
            return Or(higher(alpha, beta), And(higher(alpha, b_prime), higher(b, beta)))

        # Otherwise rebuild recursively
        new_children = [
            rewrite_for_put_for_higher(c, b_prime, b) for c in expr.children()
        ]
        return decl(*new_children)

    # Case 3: Constants, bound variables, etc.
    else:
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

    def eval(self, env, return_img=False):
        """evaluate the program in the environment and return the trajectories"""
        traj = [env.reset()[0]]
        if return_img:
            imgs = [env.render()]
        for line in self.instructions:
            line_imgs = line.eval(env, traj, return_img)
            if return_img:
                imgs.extend(line_imgs)
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

    def VC_gen(self, P, Q):
        # P, Q are z3 formula
        seq_instruction = to_seq(self.instructions)
        return [Implies(P, self.wp(Q))] + VC_aux(seq_instruction, Q)

    def wp(self, Q):
        seq_instruction = to_seq(self.instructions)
        return wp(seq_instruction, Q)

    def highlevel_verification(self, P, Q):
        vcs = self.VC_gen(P, Q)
        print("testing axioms")
        solver = highlevel_verification_lib.highlevel_z3_solver()
        solver.start_verification()
        print("=====================")

        for idx, vc in enumerate(vcs):
            print(f"verifying VC {idx}", vc)
            solver = highlevel_verification_lib.highlevel_z3_solver()
            solver.start_verification(vc)
            print("=====================")

    def lowlevel_verification(self):
        pass


class Assign(Instruction):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} <- {self.right}"

    def eval(self, env, traj, return_image=False) -> List:
        pass


class Seq:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2


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


def wp(seq_instruction, Q):
    """calculate weakest precondition"""
    if isinstance(seq_instruction, Skip):
        return Q
    elif isinstance(seq_instruction, Seq):
        return wp(seq_instruction.s1, wp(seq_instruction.s2, Q))
    elif isinstance(seq_instruction, While):
        return seq_instruction.invariant
    elif isinstance(seq_instruction, Assign):
        return substitute(
            Q,
            (
                highlevel_verification_lib.get_consts(seq_instruction.left),
                highlevel_verification_lib.get_consts(seq_instruction.right),
            ),
        )
    elif isinstance(seq_instruction, Put):
        b_prime = highlevel_verification_lib.get_consts(seq_instruction.upper_block)
        b = highlevel_verification_lib.get_consts(seq_instruction.base_block)
        Q = And(Not(ON_star(b, b_prime)), rewrite_for_put_for_ON_star(Q, b_prime, b))
        # return Q
        return rewrite_for_put_for_higher(Q, b_prime, b)
    assert (
        False
    ), f"Unrecognized seq instruction {type(seq_instruction)} to calculate wp"


def VC_aux(seq_instruction, Q) -> List:
    """generate auxiliary verification conditions"""
    if isinstance(seq_instruction, Seq):
        return VC_aux(seq_instruction.s1, wp(seq_instruction.s2, Q)) + VC_aux(
            seq_instruction.s2, Q
        )
    elif isinstance(seq_instruction, Instruction):
        return []
    elif isinstance(seq_instruction, While):
        return VC_aux(to_seq(seq_instruction.body), seq_instruction.invariant) + [
            Implies(
                And(seq_instruction.instantiated_cond, seq_instruction.invariant),
                wp(to_seq(seq_instruction.body), seq_instruction.invariant),
            ),
            Implies(And(Not(seq_instruction.cond), seq_instruction.invariant), Q),
        ]
    assert False, "Unrecognized seq instruction for VC_aux"


def run_stack_example_with_only_ON_star():
    inferred_invariant, candidate_lists = (
        synthesis.inference_lib.inference.run_proposal_example()
    )
    b_prime, b, n, b0, a = Consts("b_prime b n b0 a", BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            cond=(
                Exists(
                    [b_prime],
                    And(
                        ForAll([n], Or(b_prime == n, Not(ON_star(n, b_prime)))),
                        b_prime != b,
                    ),
                )
            ),
            instantiated_cond=And(
                ForAll([n], Or(b_prime == n, Not(ON_star(n, b_prime)))),
                b_prime != b,
            ),
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=inferred_invariant,
            # invariant=And(*candidate_lists)
        ),
    ]
    p = Program(2, instructions=instructions)

    m, n = Consts("m n", BoxSort)
    precondition = ForAll([m], ForAll([n], Or(m == n, Not(ON_star(n, m)))))

    m, n, b0 = Consts("m n b0", BoxSort)
    postcondition = ForAll([m], ON_star(m, b0))

    # print(p.VC_gen(precondition, postcondition))
    p.highlevel_verification(precondition, postcondition)


def run_unstack_example():
    pass
