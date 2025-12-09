import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn import tree

BLOCK_LENGTH = (
    0.025 * 2
)  # (0.025, 0.025, 0.025) in the xml file of the gym env is half length


class Feature:
    pass


def on(block1, block2):
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2 < 1.5 * BLOCK_LENGTH
    )


def get_block_pos(obs, block_id):
    start_idx = 10 + 12 * block_id
    end_idx = start_idx + 3
    return np.array(obs[start_idx:end_idx])


class ON_feature(Feature):
    """Represent the on(b1, b2) feature"""

    def __init__(self, b1: int, b2: int):
        self.b1 = b1
        self.b2 = b2

    def __call__(self, obs):
        return on(get_block_pos(obs, self.b1), get_block_pos(obs, self.b2))

    def __str__(self) -> str:
        return f"ON({self.b1}, {self.b2})"


def compute_ON_features(num_block: int):
    features = []
    for b1 in range(num_block):
        for b2 in range(num_block):
            features.append(ON_feature(b1, b2))
    return features


def learn_features(sample_trajs, demos, demo_goal_idxs, features, num_trees: int):
    """learn the feature that could seperate the positive and negative samples"""
    positive_samples = compute_positive_set(demos, demo_goal_idxs)
    negative_samples = compute_negative_set(sample_trajs)
    all_samples, all_features, all_labels = compute_features(
        positive_samples, negative_samples, features
    )
    all_trees = []
    for _ in range(num_trees):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(all_features, all_labels)
        all_trees.append(clf)
    
    best_tree, best_goal_idx = None, None
    for cur_tree in all_trees:
        # for each classifying feature, check the first index that this feature becomes true after
        avg_goal_idx, _ = check_feature(cur_tree, demos, demo_goal_idxs)
        if best_goal_idx is None or best_goal_idx > avg_goal_idx:
            best_goal_idx, best_tree = avg_goal_idx, cur_tree
    return best_tree


def check_feature(feature, demos, demo_goal_idxs):
    """return the average index that the feature becomes true afterwards"""
    first_idxs = []
    for demo, goal_idx in zip(demos, demo_goal_idxs):
        assert goal_idx > 0
        assert feature(demo[goal_idx - 1])
        cur_idx = goal_idx
        while cur_idx > 0:
            if not feature(demo[cur_idx - 1]):
                first_idxs.append(cur_idx)
                break
            cur_idx -= 1;
    return sum(first_idxs) / len(first_idxs), first_idxs


def compute_features(positive_samples, negative_samples, features):
    all_samples = positive_samples + negative_samples
    all_labels = [1 for _ in range(positive_samples)] + [0 for _ in range(negative_samples)]
    all_features = []
    for sample in all_samples:
        all_features.append(
            [f(sample) for f in features]
        )
    return all_samples, all_features, all_labels


def compute_negative_set(sample_trajs):
    """sample_trajs is a 2D list of sampled trajectories using current program"""
    negative_samples = []
    for traj in sample_trajs:
        for state in traj:
            negative_samples.append(deepcopy(state))
    return negative_samples


def compute_positive_set(demos, demo_goal_idxs):
    """demos is 2D array of demo trajectories, demo_gaol_idxs is a list containing current idxs"""
    positive_samples = []
    assert len(demos) == len(demo_goal_idxs)
    for demo, goal_idx in zip(demos, demo_goal_idxs):
        if goal_idx == 0:
            print("current goal idx is already 0, aborting")
            exit()
        positive_samples.append(deepcopy(demo[goal_idx - 1]))
    return positive_samples
