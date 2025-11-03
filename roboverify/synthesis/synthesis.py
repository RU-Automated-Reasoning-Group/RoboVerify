import math
import os
import random

import cem
import ffmpeg
import imageio
import numpy as np
from cost_func import *
from environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
from environment.data.collect_demos import CollectDemos
from environment.general_env import GymToGymnasium
from program import *

from typing import Any
from copy import deepcopy
import random
import decision_tree
import pickle
from PIL import Image  # for saving images as PNG


# def learn(
#     demos: list[list[Any]],
#     predicates: list[Callable[[Any], bool]],
#     success_threshold: float,
#     mcmc_iters: int,
#     num_seeds: int,
#     cem_N: int = 16,
#     cem_K: int = 4,
# ) -> Tuple[Any, Callable[[Any], bool], int]:
#     """
#     Learn a program from demonstrations using MCMC. If the program
#     fails to achieve a success rate above `success_threshold`, extract
#     positive/negative examples and learn a decision tree labeling function.

#     Parameters
#     ----------
#     demos : list[list[Any]]
#         Expert demonstration sequences (list of states per demo).
#     predicates : list[Callable[[Any], bool]]
#         Predicates used to build the decision tree labeling function.
#     success_threshold : float
#         Minimum acceptable success rate for the best program.
#     mcmc_iters : int
#         Number of MCMC iterations.
#     num_seeds : int
#         Number of seeds for program evaluation or CEM optimization.
#     cem_N : int
#         CEM parameter N (population size).
#     cem_K : int
#         CEM parameter K (top-K selection).

#     Returns
#     -------
#     best_model_or_program : Either the best decision tree (if threshold not met) or the best program.
#     label_fn : Callable[[Any], bool]
#         Either the decision tree labeling function or a wrapper around the program.
#     best_idx : int
#         Index of best tree if tree learned, otherwise 0.
#     """
#     # 1️⃣ Run MCMC to propose programs
#     initial_program = (
#         Program()
#     )  # assuming Program class exists and has a default constructor
#     samples, costs = MCMC(
#         current_program=initial_program,
#         iters=mcmc_iters,
#         expert_states=demos,
#         num_seeds=num_seeds,
#         cem_N=cem_N,
#         cem_K=cem_K,
#     )

#     # 2️⃣ Pick the best program (lowest cost)
#     best_idx_prog = costs.index(min(costs))
#     best_program = samples[best_idx_prog]

#     # 3️⃣ Evaluate best program
#     success_rate, states = evaluate(
#         best_program
#     )  # should return success rate and states visited

#     if success_rate >= success_threshold:
#         # Program is good enough: return it directly
#         def program_label_fn(state: Any) -> bool:
#             # Wrap the program's decision as a labeling function
#             return best_program.apply(state)  # assuming apply(state) returns True/False

#         return best_program, program_label_fn, 0

#     # 4️⃣ Collect positive/negative examples from evaluation
#     neg_examples: list[Any] = states  # all visited states
#     pos_examples: list[Any] = []
#     for state_seq in states:
#         if len(state_seq) >= 2:
#             pos_examples.append(state_seq[-2])  # state immediately before last
#         else:
#             pos_examples.append(state_seq[0])

#     # 5️⃣ Learn decision tree labeling function
#     best_tree, label_fn, tree_idx = decision_tree.select_best_tree(
#         pos_examples=pos_examples,
#         neg_examples=neg_examples,
#         predicates=predicates,
#         demos=demos,
#         n_trees=10,  # fixed number of trees
#         base_tree_config={"max_depth": 3, "criterion": "entropy"},
#         visualize_dir=None,  # can optionally provide a directory
#     )

#     return best_tree, label_fn, tree_idx


def sample_proportional(values):
    """
    Given a list of positive integers, sample one index with probability
    proportional to its value.

    Parameters:
    - values: list of positive integers

    Returns:
    - An index sampled according to the proportional distribution
    """
    total = sum(values)
    probs = [v / total for v in values]
    return random.choices(range(len(values)), weights=probs, k=1)[0]


def swap_two_elements_maybe_same(lst):
    """
    Randomly selects two indices (which may be the same) and swaps their elements.

    Parameters:
    - lst: List of elements to mutate (in-place)

    Returns:
    - A new list with two elements swapped (or not, if indices are the same)
    """
    if len(lst) == 0:
        raise ValueError("List must contain at least one element.")

    i = random.randint(0, len(lst) - 1)
    j = random.randint(0, len(lst) - 1)
    print(f"swap instructions {i} and {j}")
    lst[i], lst[j] = lst[j], lst[i]
    return lst


def mutate_program(
    current_program: Program,
    available_operands: dict,
    available_instructions: list,
) -> tuple[Program, bool]:
    pc = 0  # opcode
    po = 1  # operand
    ps = 1  # swap
    pi = 1  # instruction
    sampled_mutation = sample_proportional([pc, po, ps, pi])
    mutate_type_name = {
        0: "opcode",
        1: "operand",
        2: "swap",
        3: "instruction",
    }[sampled_mutation]
    print("mutation type", mutate_type_name)

    new_program = deepcopy(current_program)
    if sampled_mutation == 0:
        pass
    elif sampled_mutation == 1:
        index = random.randint(0, len(new_program.instructions) - 1)
        old_operands = new_program.instructions[index].get_operand()
        if old_operands:
            operand_index = random.randint(0, len(old_operands) - 1)
            proposed_operand = random.choice(
                available_operands[old_operands[operand_index]["type"]]
            )
            print(
                "old operand",
                old_operands[operand_index],
                "proposed operand",
                proposed_operand,
            )
            if old_operands[operand_index] == proposed_operand:
                return new_program, False
            old_operands[operand_index]["val"] = proposed_operand
            new_program.instructions[index].set_operand(old_operands)
        else:
            return new_program, False
    elif sampled_mutation == 2:
        swap_two_elements_maybe_same(new_program.instructions)
    elif sampled_mutation == 3:
        index = random.randint(0, len(new_program.instructions) - 1)
        pu = 0.25  # probability the SKIP token is proposed
        if random.random() < pu:
            if type(new_program.instructions[index]) == Skip:
                print("chaning skip to skip")
                return new_program, False
            new_program.instructions[index] = Skip()
        else:
            new_instruction = random.choice(available_instructions)()
            operands = new_instruction.get_operand()
            for i in range(len(operands)):
                operands[i]["val"] = random.choice(
                    available_operands[operands[i]["type"]]
                )
            new_instruction.set_operand(operands)
            new_program.instructions[index] = new_instruction
    else:
        assert False, "unknown mutation"

    return new_program, True


import os
import pickle
import math
import random
from copy import deepcopy


def MCMC(
    current_program: Program,
    available_operands: dict,
    available_instructions: list,
    iters: int,
    expert_states,
    num_seeds: int,
    cem_N: int = 16,
    cem_K: int = 4,
    cem_iterations: int = 10,
    save_dir: str = "MCMC_results",
):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    print("=== initial program ===\n", current_program)
    current_cost, current_program = optimize_program(
        current_program, expert_states, num_seeds, cem_N, cem_K, cem_iterations
    )
    print("evaluated with cost", current_cost)

    samples = [deepcopy(current_program)]
    costs = [current_cost]

    for i in range(iters):
        iter_folder = os.path.join(save_dir, f"iter{i}")
        os.makedirs(iter_folder, exist_ok=True)

        new_program, changed = mutate_program(
            current_program, available_operands, available_instructions
        )
        print(f"=== iter {i} program ===\n", new_program)

        ins1 = [x for x in current_program.instructions if type(x) != Skip]
        ins2 = [x for x in new_program.instructions if type(x) != Skip]
        equivalence = ins1 == ins2
        print("program equivalence", equivalence)

        # Save a copy of current_program BEFORE acceptance update
        saved_current_program = deepcopy(current_program)
        saved_current_cost = current_cost

        if changed and not equivalence:
            # Optimize new_program, which may modify it
            new_cost, new_program = optimize_program(
                new_program, expert_states, num_seeds, cem_N, cem_K, cem_iterations
            )
            print("evaluated with cost", new_cost)
        else:
            new_cost = current_cost

        # Save a copy of new_program AFTER optimization
        saved_new_program = deepcopy(new_program)

        # Decide acceptance
        if changed and not equivalence:
            acceptance_ratio = math.exp(new_cost - current_cost)
            print("acceptance_ratio is", acceptance_ratio)
            if random.random() < acceptance_ratio:
                print("new program accepted")
                current_program = new_program
                current_cost = new_cost
            else:
                print("new program NOT accepted")

        # Save pickle files
        with open(os.path.join(iter_folder, "current_program.pkl"), "wb") as f:
            pickle.dump(saved_current_program, f)
        with open(os.path.join(iter_folder, "new_program.pkl"), "wb") as f:
            pickle.dump(saved_new_program, f)

        # Save text summary
        with open(os.path.join(iter_folder, "summary.txt"), "w") as f:
            f.write("=== Current Program ===\n")
            f.write(str(saved_current_program) + "\n")
            f.write(f"Current Cost: {saved_current_cost}\n\n")
            f.write("=== New Program ===\n")
            f.write(str(saved_new_program) + "\n")
            f.write(f"New Cost: {new_cost}\n")
            f.write(f"Program Equivalence: {equivalence}\n")
            f.write(f"Accepted: {current_program == new_program}\n")

        # --- Evaluate program and save images/video ---
        frames_dir = os.path.join(iter_folder, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Evaluate new_program and get images
        _, imgs = evaluate_program(saved_new_program, n=num_seeds, return_img=True)

        # Save each image as PNG
        for idx, img in enumerate(imgs):
            img_path = os.path.join(frames_dir, f"img{idx:04d}.png")
            if isinstance(img, Image.Image):
                img.save(img_path)
            else:  # assume numpy array
                Image.fromarray(img).save(img_path)

        # Generate video
        output_video_path = os.path.join(iter_folder, "program_video.mp4")
        images_to_video(frames_dir, output_video_path=output_video_path)

        samples.append(deepcopy(new_program))
        costs.append(new_cost)

    return samples, costs


def optimize_program(
    p: Program,
    expert_states,
    num_seeds: int,
    cem_N: int,
    cem_K: int,
    cem_iterations: int,
) -> tuple[float, Program]:
    f = Runner(p, expert_states, num_seeds)
    initial_parameters = p.register_trainable_parameter()
    iterations = cem_iterations if initial_parameters else 0
    best_cost, best_parameter = cem.cem_optimize(
        f,
        len(initial_parameters),
        iterations,
        N=cem_N,
        K=cem_K,
        init_mu=initial_parameters,
    )
    p.update_trainable_parameter(best_parameter)
    return best_cost, p


def set_np_seed(seed: int):
    np.random.seed(seed)


def images_to_video(input_dir, output_video_path="output.mp4", framerate=30):
    """
    Convert PNG images in a directory to a video using ffmpeg.
    Assumes images are named as img0000.png, img0001.png, ...

    Parameters:
    - input_dir: directory containing the PNG images
    - output_video_path: output video file path
    - framerate: frames per second for the video
    """
    input_pattern = os.path.join(input_dir, "img%04d.png")

    try:
        (
            ffmpeg.input(input_pattern, framerate=framerate)
            .output(output_video_path, vcodec="libx264", pix_fmt="yuv420p")
            .overwrite_output()
            .run()
        )
        print(f"Video saved to {output_video_path}")
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())


def save_numpy_arrays_as_images(arrays, output_dir="images"):
    """
    Save a list of numpy arrays as PNG images with filenames like img0000.png, img0001.png, ...

    Parameters:
    - arrays: list of numpy arrays (each should represent an image)
    - output_dir: directory where images will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, arr in enumerate(arrays):
        filename = f"img{idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        imageio.imwrite(filepath, arr)


def collect_trajectories(env_name: str, n: int, save_imgs=False):
    states = []
    imgs = []
    for i in range(n):
        set_np_seed(i)
        collector = CollectDemos(
            "demo",
            traj_len=50,
            num_trajectories=1,
            task="tower",
            img_path="img",
            env_name=env_name,
            block_num=int(2),
            debug=False,
            render=False,
        )
        obs_seq, obs_imgs = collector.collect(store=False)
        collector.env.close()
        del collector
        for state in obs_seq[0]["obs"]:
            states.append(state)
        for image in obs_imgs[0]:
            imgs.append(image)
    if save_imgs:
        save_numpy_arrays_as_images(imgs)
    print("number of expert states", len(states))
    return states


def evaluate_program(p: Program, n: int, return_img=False):
    states = []
    imgs = []
    for i in range(n):
        set_np_seed(i)
        env_name = "pickmulti1"
        env = GymToGymnasium(
            FetchPickAndPlaceConstruction(
                name=env_name,
                sparse=False,
                shaped_reward=False,
                num_blocks=int(2),
                reward_type="sparse",
                case="Singletower",
                visualize_mocap=False,
                stack_only=True,
                simple=True,
            ),
        )
        traj = p.eval(env, return_img)
        if return_img:
            traj, traj_imgs = traj
            for image in traj_imgs:
                imgs.append(image)

        for state in traj:
            states.append(state)

    print("number of policy states", len(states))
    if return_img:
        return states, imgs
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
        self.slices: list[Any] = [slice(None)] * self.expert_states.ndim
        self.slices[1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24]
        self.tuple_slices = tuple(self.slices)
        self.expert_states = self.expert_states[self.tuple_slices]
        self.num_seeds = num_seeds

    def __call__(self, new_parameters):
        p = deepcopy(self.p)
        p.update_trainable_parameter(new_parameters)
        policy_states = evaluate_program(p, self.num_seeds)
        policy_states = np.array(policy_states)[self.tuple_slices]
        kl_value = kl_divergence_kde(policy_states, self.expert_states)
        return -kl_value


if __name__ == "__main__":
    available_operands = {
        "Box": [0, 1],
    }
    available_instructions = [PickPlace]
    num_seeds = 15
    expert_states = collect_trajectories("pickmulti1", num_seeds, save_imgs=True)
    images_to_video("images", "groundtruth.mp4")

    MCMC(
        Program(3),
        available_operands,
        available_instructions,
        200,
        expert_states=expert_states,
        num_seeds=num_seeds,
    )

    exit()

    # num_seeds = 15

    # expert_states = collect_trajectories("pickmulti1", num_seeds, save_imgs=True)
    # exit()
    p = Program(3)
    p.instructions = [
        PickPlace(grab_box_id=1, target_box_id=1),  # move up with respect to box 0
        PickPlace(
            grab_box_id=1, target_box_id=0
        ),  # move horizontally to the top of box 1
        PickPlace(grab_box_id=1, target_box_id=0),  # move down to place on box 1
    ]
    p.register_trainable_parameter()
    p.update_trainable_parameter(
        [
            0.0279608,
            0.04278932,
            0.13399421,
            0.02289095,
            -0.04434892,
            0.13288633,
            -0.01862492,
            0.00964001,
            0.05832452,
        ]
    )
    # # p.instructions[0].target_offset = [Parameter(0.), Parameter(0.), Parameter(0.2)]
    # # p.instructions[1].target_offset = [Parameter(0), Parameter(0), Parameter(0.2)]
    # # p.instructions[2].target_offset = [Parameter(0), Parameter(0), Parameter(0.05)]
    # # for _ in range(50):
    # #     print(mutate_program(p))
    # pdb.set_trace()
    iter_folder = "tmp_testing1"
    frames_dir = os.path.join(iter_folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Evaluate new_program and get images
    _, imgs = evaluate_program(p, n=num_seeds, return_img=True)

    # Save each image as PNG
    for idx, img in enumerate(imgs):
        img_path = os.path.join(frames_dir, f"img{idx:04d}.png")
        if isinstance(img, Image.Image):
            img.save(img_path)
        else:  # assume numpy array
            Image.fromarray(img).save(img_path)

    # Generate video
    output_video_path = os.path.join(iter_folder, "program_video.mp4")
    images_to_video(frames_dir, output_video_path=output_video_path)
    # states, imgs = evaluate_program(p, 15, True)
    # pdb.set_trace()

    # f = Runner(p, expert_states, num_seeds)
    # initial_parameters = p.register_trainable_parameter()
    # cem.cem_optimize(f, len(initial_parameters), N=16, K=4, init_mu=initial_parameters)

# expert_states = np.array(expert_states)[:,:3]
# pdb.set_trace()
# policy_states = np.array(policy_states)[:,:3]

# kl_value = kl_divergence_kde(policy_states, expert_states)
# print(kl_value)

# kl_value = kl_divergence_kde(expert_states, expert_states)
# print(kl_value)
