import pdb
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from z3 import Implies, And, Or, Not

class Parameter:
    def __init__(self, val: float = 0):
        self.pos: int | None = None
        self.val: float = val

    def register(self, parameters: list):
        self.pos = len(parameters)
        parameters.append(self.val)

    def update(self, new_parameter: list[float]):
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
    def eval(self, env, traj, return_image=False) -> list:
        pass

    def register_trainable_parameter(self, parameters: list):
        pass

    def update_trainable_parameter(self, new_parameter: list):
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
        if box_id == 0:
            return obs[10:13]
        elif box_id == 1:
            return obs[22:25]
        else:
            assert False, "unknown box id"

    def eval(self, env, traj, return_img=False):
        # call the neural controller here
        from environment.data.pickplace_naive import get_pick_control_naive

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

    def register_trainable_parameter(self, parameter: list[float]):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter: list[float]):
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
    def __init__(self, cond, body: list[Instruction], invariant):
        self.cond = cond
        self.body = body
        self.invariant = invariant


class Program:
    def __init__(self, length: int = 5):
        self.length = length
        self.instructions: list[Instruction] = [Skip() for _ in range(self.length)]

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

class Assign(Instruction):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"{self.left} <- {self.right}"


class Seq:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

def to_seq(instructions):
    return []

def wp(seq_instruction, Q):
    if isinstance(seq_instruction, Skip):
        return Q
    elif isinstance(seq_instruction, Seq):
        return wp(seq_instruction.s1, wp(seq_instruction.s2, Q))
    elif isinstance(seq_instruction, While):
        return seq_instruction.invariant
    elif isinstance(seq_instruction, Assign)
    assert False, "Unrecognized seq instruction to calculate wp"

def VC_aux(seq_instruction, Q) -> list:
    if isinstance(seq_instruction, Seq):
        return VC_aux(seq_instruction.s1, wp(seq_instruction.s2, Q)) + VC_aux(seq_instruction.s2, Q)
    elif isinstance(seq_instruction, Instruction):
        return []
    elif isinstance(seq_instruction, While):
        return VC_aux(seq_instruction.body, seq_instruction.invariant) + [Implies(And(seq_instruction.cond, seq_instruction.invariant), wp(seq_instruction.body, seq_instruction.invariant)), 
                                                                          Implies(And(Not(seq_instruction.cond), seq_instruction.invariant), Q)]
    assert False, "Unrecognized seq instruction for VC_aux"

if __name__ == "__main__":
    program = Program(3)
    program.instructions = [PickPlace(0), PickPlace(0), PickPlace(0)]
    trainable_parameters = program.register_trainable_parameter()
    print(trainable_parameters)
    updated_paremters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    program.update_trainable_parameter(updated_paremters)
    print(program.register_trainable_parameter())
