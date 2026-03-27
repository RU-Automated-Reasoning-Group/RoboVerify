import pdb
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from z3 import Exists


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
        self.target_offset = [Parameter(0.0) for _ in range(3)]

    def get_box_pos(self, box_id, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_img=False):
        from synthesis.environment.data.pickplace_naive import get_pick_control_naive

        imgs = []
        success = False
        initial_goal_box = self.get_box_pos(self.target_box_id, traj[-1])
        step = 0
        while not success and step < self.limit:
            obs = env.flatten_observation(env.env._get_obs())
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
    def __init__(
        self,
        instantiated_cond,
        guard_exists_vars,
        body,
        invariant,
    ):
        if guard_exists_vars is None:
            raise ValueError("guard_exists_vars must be provided for While.")
        self.body = body
        self.invariant = invariant
        self.guard_exists_vars = guard_exists_vars
        self.instantiated_cond = instantiated_cond
        self.cond = Exists(guard_exists_vars, instantiated_cond)


class Put(Instruction):
    def __init__(self, upper_block, base_block):
        self.base_block = base_block
        self.upper_block = upper_block

    def __str__(self):
        return f"put({self.base_block}, {self.upper_block})"

    def eval(self, env, traj, return_image=False) -> List:
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
