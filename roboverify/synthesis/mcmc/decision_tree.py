from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

from synthesis.util import on


class Feature:
    pass


class ON_feature(Feature):
    """Represent the on(b1, b2) feature"""

    def __init__(self, b1: int, b2: int):
        self.b1 = b1
        self.b2 = b2

    def __call__(self, obs):
        return on.on(on.get_block_pos(obs, self.b1), on.get_block_pos(obs, self.b2))

    def __str__(self) -> str:
        return f"ON({self.b1}, {self.b2})"

    def __repr__(self) -> str:
        return self.__str__()


def compute_ON_features(num_block: int):
    features = []
    for b1 in range(num_block):
        for b2 in range(num_block):
            features.append(ON_feature(b1, b2))
    return features


def learn_features(
    num_block: int, sample_trajs, demos, demo_goal_idxs, features, num_trees: int
):
    """learn the feature that could seperate the positive and negative samples"""
    positive_samples = compute_positive_set(demos, demo_goal_idxs)
    negative_samples = compute_negative_set(sample_trajs)
    all_samples, all_features, all_labels = compute_features(
        positive_samples, negative_samples, features
    )
    all_trees = []
    for i in range(num_trees):
        clf = tree.DecisionTreeClassifier(max_depth=1, random_state=i)
        clf = clf.fit(all_features, all_labels)
        training_score = clf.score(all_features, all_labels)
        all_trees.append((clf, training_score))

    best_feature_idx, best_tree, best_goal_idx = None, None, None
    for idx, (cur_tree, training_score) in enumerate(all_trees):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        tree.plot_tree(cur_tree, filled=True, rounded=True, ax=axes)
        plt.savefig(f"decision_tree{idx}.png", dpi=300, bbox_inches="tight")

        if training_score < 1.0:
            continue
        # for each classifying feature, check the first index that this feature becomes true after
        feature_id = cur_tree.tree_.feature[0]
        avg_goal_idx, _ = check_feature(features[feature_id], demos, demo_goal_idxs)
        print(
            f"learned tree {idx} is using feature {features[feature_id]} with average goal index {avg_goal_idx}"
        )
        if best_goal_idx is None or best_goal_idx > avg_goal_idx:
            best_feature_id, best_goal_idx, best_tree = (
                feature_id,
                avg_goal_idx,
                cur_tree,
            )

    print(
        f"selecting tree with feature {features[best_feature_id]} with average goal idx {best_goal_idx}"
    )
    return best_tree, features[best_feature_id]


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
            cur_idx -= 1
    return sum(first_idxs) / len(first_idxs), first_idxs


def compute_features(positive_samples, negative_samples, features):
    all_samples = positive_samples + negative_samples
    all_labels = [1 for _ in range(len(positive_samples))] + [
        0 for _ in range(len(negative_samples))
    ]
    all_features = []
    for sample in all_samples:
        all_features.append([f(sample) for f in features])
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
