import numpy as np

import cem
from cost_func import *
from environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
from environment.data.collect_demos import CollectDemos
from environment.general_env import GymToGymnasium
from program import *

num_seeds = 10


def set_np_seed(seed: int):
    np.random.seed(seed)


def collect_trajectories(env_name: str, n: int):
    states = []
    for i in range(n):
        set_np_seed(i)
        collector = CollectDemos(
            "demo",
            traj_len=50,
            num_trajectories=1,
            task="pickmulti",
            img_path="img",
            env_name=env_name,
            block_num=int(1),
            debug=False,
        )
        obs_seq, obs_imgs = collector.collect(store=False)
        del collector
        for state in obs_seq[0]["obs"]:
            states.append(state)
    print("number of expert states", len(states))
    return states


def evaluate_program(p: Program, n: int):
    states = []
    for i in range(n):
        set_np_seed(i)
        env_name = "pickmulti1"
        env = GymToGymnasium(
            FetchPickAndPlaceConstruction(
                name=env_name,
                sparse=False,
                shaped_reward=False,
                num_blocks=int(env_name[-1]),
                reward_type="sparse",
                case="PickAndPlace",
                visualize_mocap=False,
                simple=True,
            ),
            render_mode="human",
        )
        traj = p.eval(env)
        for state in traj:
            states.append(state)
    print("number of policy states", len(states))
    return states


# set_np_seed(10)
# env_name = "pickmulti1"
# env1 = GymToGymnasium(
#     FetchPickAndPlaceConstruction(
#         name=env_name,
#         sparse=False,
#         shaped_reward=False,
#         num_blocks=int(env_name[-1]),
#         reward_type="sparse",
#         case="PickAndPlace",
#         visualize_mocap=False,
#         simple=True,
#     ),
#     render_mode="human"
# )
# obs1, _ = env1.reset()

# set_np_seed(10)
# env2 = GymToGymnasium(
#     FetchPickAndPlaceConstruction(
#         name=env_name,
#         sparse=False,
#         shaped_reward=False,
#         num_blocks=int(env_name[-1]),
#         reward_type="sparse",
#         case="PickAndPlace",
#         visualize_mocap=False,
#         simple=True,
#     ),
#     render_mode="human"
# )
# obs2, _ = env2.reset()
# pdb.set_trace()


class Runner:
    def __init__(self, program: Program, expert_states, num_seeds: int = 10):
        self.p = program
        self.expert_states = np.array(expert_states)
        self.slices = [slice(None)] * self.expert_states.ndim
        self.slices[1] = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
        ]
        self.slices = tuple(self.slices)
        self.expert_states = self.expert_states[self.slices]
        self.num_seeds = num_seeds

    def __call__(self, new_parameters):
        p = deepcopy(self.p)
        p.update_trainable_parameter(new_parameters)
        policy_states = evaluate_program(p, self.num_seeds)
        policy_states = np.array(policy_states)[self.slices]
        kl_value = kl_divergence_kde(policy_states, self.expert_states)
        return -kl_value


expert_states = collect_trajectories("pickmulti1", num_seeds)

p = Program(3)
p.instructions = [PickPlace(0)]

f = Runner(p, expert_states, num_seeds)
initial_parameters = p.register_trainable_parameter()
cem.cem_optimize(f, len(initial_parameters), N=16, K=5, init_mu=initial_parameters)

# expert_states = np.array(expert_states)[:,:3]
# pdb.set_trace()
# policy_states = np.array(policy_states)[:,:3]

# kl_value = kl_divergence_kde(policy_states, expert_states)
# print(kl_value)

# kl_value = kl_divergence_kde(expert_states, expert_states)
# print(kl_value)
