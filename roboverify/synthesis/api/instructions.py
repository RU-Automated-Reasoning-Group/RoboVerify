import pdb
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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


class PickPlaceByName(Instruction):
    def __init__(
        self,
        *,
        grab_box_name: str,
        target_box_name_x: str,
        target_box_name_y: str,
        target_box_name_z: str,
        limit: int = 50,
        target_offset: Optional[List[float]] = None,
    ):
        self.limit = limit
        self.grab_box_name = grab_box_name
        self.target_box_name_x = target_box_name_x
        self.target_box_name_y = target_box_name_y
        self.target_box_name_z = target_box_name_z
        self.types = ["BoxName", "BoxName", "BoxName", "BoxName"]
        if target_offset is None:
            self.target_offset = [Parameter(0.0) for _ in range(3)]
        else:
            if len(target_offset) != 3:
                raise ValueError("target_offset must be a length-3 list of floats.")
            self.target_offset = [Parameter(float(v)) for v in target_offset]

    def _resolve(self, env) -> Dict[str, int]:
        mapping = getattr(env, "symbolic_name_to_box_id", None)
        if mapping is None:
            raise ValueError(
                "PickPlaceByName requires env.symbolic_name_to_box_id (e.g. {'b0': 1})."
            )
        if not isinstance(mapping, dict):
            raise TypeError("env.symbolic_name_to_box_id must be a dict[str, int].")
        return mapping

    @property
    def target_box_names(self) -> tuple[str, str, str]:
        return (self.target_box_name_x, self.target_box_name_y, self.target_box_name_z)

    def get_box_pos(self, box_id: int, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_img=False):
        from synthesis.environment.data.pickplace_naive import get_pick_control_naive

        mapping = self._resolve(env)
        grab_box_id = mapping[self.grab_box_name]
        target_x_id = mapping[self.target_box_name_x]
        target_y_id = mapping[self.target_box_name_y]
        target_z_id = mapping[self.target_box_name_z]

        imgs = []
        success = False
        obs0 = traj[-1]
        gx, gy, gz = (
            self.get_box_pos(target_x_id, obs0)[0],
            self.get_box_pos(target_y_id, obs0)[1],
            self.get_box_pos(target_z_id, obs0)[2],
        )
        initial_goal = np.array([gx, gy, gz], dtype=float)

        step = 0
        while not success and step < self.limit:
            obs = env.flatten_observation(env.env._get_obs())
            action, success = get_pick_control_naive(
                obs,
                initial_goal + np.array([offset.val for offset in self.target_offset]),
                block_id=grab_box_id,
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
            {"type": self.types[0], "val": self.grab_box_name},
            {"type": self.types[1], "val": self.target_box_name_x},
            {"type": self.types[2], "val": self.target_box_name_y},
            {"type": self.types[3], "val": self.target_box_name_z},
        ]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "BoxName"
        assert new_operands[1]["type"] == "BoxName"
        assert new_operands[2]["type"] == "BoxName"
        assert new_operands[3]["type"] == "BoxName"
        self.grab_box_name = new_operands[0]["val"]
        self.target_box_name_x = new_operands[1]["val"]
        self.target_box_name_y = new_operands[2]["val"]
        self.target_box_name_z = new_operands[3]["val"]

    def __eq__(self, other):
        if not isinstance(other, PickPlaceByName):
            return False
        cond1 = self.get_operand() == other.get_operand()
        cond2 = self.target_offset == other.target_offset
        return cond1 and cond2

    def __str__(self):
        tx, ty, tz = self.target_box_names
        return (
            f"PickPlaceByName({self.grab_box_name}, ({tx}, {ty}, {tz}), "
            f"{[str(x) for x in self.target_offset]})"
        )


class While:
    def __init__(
        self,
        instantiated_cond,
        guard_exists_vars,
        body: List[Instruction],
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
        # `Put` is a logical/table-level operation in the stack DSL.
        # For physical execution, the program should typically replace
        # `Put`/`Assign`/`While` with concrete pick-and-place instructions.
        #
        # We intentionally do not mutate `env.symbolic_name_to_box_id` here:
        # symbolic aliases are established explicitly via `Assign`.
        raise RuntimeError(
            "Put.eval() was called, but `Put` is a verification-only logical "
            "operation in the stack DSL. It cannot be evaluated at runtime. "
            "Replace `Put`/`Assign`/`While` with concrete pick-and-place instructions "
            "before execution."
        )


class Assign(Instruction):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} <- {self.right}"

    def eval(self, env, traj, return_image=False) -> List:
        mapping = getattr(env, "symbolic_name_to_box_id", None)
        if mapping is None:
            raise ValueError(
                "Assign.eval requires env.symbolic_name_to_box_id to exist."
            )
        if not isinstance(mapping, dict):
            raise TypeError("env.symbolic_name_to_box_id must be a dict[str, int].")

        if self.right not in mapping:
            raise KeyError(
                f"Assign.eval could not resolve RHS symbolic name {self.right!r}; "
                f"available: {sorted(mapping.keys())}"
            )

        # Create/update the alias: env[left] = env[right].
        mapping[self.left] = mapping[self.right]
        return []


class Seq:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
