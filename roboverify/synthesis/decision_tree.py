from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

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


def learn_features(sample_trajs, demos, demo_goal_idxs, features):
    positive_samples = compute_positive_set(demos, demo_goal_idxs)
    negative_samples = sample_trajs
    all_samples, all_features = compute_features(
        positive_samples, negative_samples, features
    )


def compute_positive_set(demos, demo_goal_idxs):
    pass


def compute_features(positive_samples, negative_samples, features):
    pass
