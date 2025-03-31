from abc import ABC, abstractmethod


class Instruction(ABC):
    @abstractmethod
    def eval(self, env):
        pass


class Skip(Instruction):
    def __init__(self, skip_steps: int = 10):
        self.skip_steps = skip_steps

    def eval(self, env):
        # TODO: wee need to consider what we should do for skip
        pass


class PickPlace(Instruction):
    def __init__(self):
        pass

    def eval(self, env):
        # call the neural controller here
        pass


class Program:
    def __init__(self, length: int = 5):
        self.length = length
        self.instructions = [Skip() for _ in range(self.length)]

    def eval(self, env):
        """evaluate the program in the environment and return the trajectories"""
        traj = [env.reset()]
        for line in self.instructions:
            segment = line.eval(env)
            traj.extend(segment)

        return traj
