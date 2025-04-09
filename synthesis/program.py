import pdb
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import numpy as np


class Parameter:
    def __init__(self, val: int = 0):
        self.pos = None
        self.val = val

    def register(self, parameters: List):
        self.pos = len(parameters)
        parameters.append(self.val)

    def update(self, new_parameter: List):
        self.val = new_parameter[self.pos]


class Instruction(ABC):
    @abstractmethod
    def eval(self, env):
        pass

    def register_trainable_parameter(self, parameters: List):
        pass

    def update_trainable_parameter(self, new_parameter: List):
        pass


class Skip(Instruction):
    def __init__(self, skip_steps: int = 10):
        self.skip_steps = skip_steps

    def eval(self, env):
        # TODO: wee need to consider what we should do for skip
        pass


class PickPlace(Instruction):
    def __init__(self, box_id: int):
        self.box_id = box_id
        # self.target_offset = [Parameter(0) for _ in range(3)]
        self.target_offset = [Parameter(-0.0), Parameter(-0.2), Parameter(0.08)]

    def eval(self, env, traj, limit: int = 50):
        # call the neural controller here
        from environment.data.pickplace_dest import get_pickdest_control

        success = False
        initial_box = traj[-1][10:13]
        step = 0
        while not success and step < limit:
            obs = env.flatten_observation(env._get_obs())
            # import pdb; pdb.set_trace()
            action, success = get_pickdest_control(
                obs,
                initial_box + np.array([offset.val for offset in self.target_offset]),
                block_id=0,
                last_block=True,
            )
            env.step(action)
            step += 1
            # env.render()
            # print(step, success)
            traj.append(obs)

    def register_trainable_parameter(self, parameter: List):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter):
        for p in self.target_offset:
            p.update(new_parameter)


class Program:
    def __init__(self, length: int = 5):
        self.length = length
        self.instructions = [Skip() for _ in range(self.length)]

    def eval(self, env):
        """evaluate the program in the environment and return the trajectories"""
        traj = [env.reset()[0]]
        for line in self.instructions:
            line.eval(env, traj)
        return traj

    def register_trainable_parameter(self):
        parameters = []
        for line in self.instructions:
            line.register_trainable_parameter(parameters)
        return parameters

    def update_trainable_parameter(self, new_parameter):
        for line in self.instructions:
            line.update_trainable_parameter(new_parameter)


def evaluate_program(p: Program, parameter: List):
    p = deepcopy(p)
    p.update_trainable_parameter(parameter)
    return p.eval()


if __name__ == "__main__":

    program = Program(3)
    program.instructions = [PickPlace(0), PickPlace(0), PickPlace(0)]
    trainable_parameters = program.register_trainable_parameter()
    print(trainable_parameters)
    updated_paremters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    program.update_trainable_parameter(updated_paremters)
    print(program.register_trainable_parameter())
