import pdb
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


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
                block_id=0,
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

    def __str__(self):
        return f"PickPlace({self.grab_box_id}, {self.target_box_id}, {[str(x) for x in self.target_offset]})"


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


if __name__ == "__main__":
    program = Program(3)
    program.instructions = [PickPlace(0), PickPlace(0), PickPlace(0)]
    trainable_parameters = program.register_trainable_parameter()
    print(trainable_parameters)
    updated_paremters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    program.update_trainable_parameter(updated_paremters)
    print(program.register_trainable_parameter())
