from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

def train_labeling_function(pos_examples, neg_examples, predicates, tree_config=None, tree_path=None):
    """
    Train a decision tree labeling function using given predicate features,
    and optionally save a visualization of the trained tree.
    
    Parameters
    ----------
    pos_examples : list
        List of raw positive examples.
    neg_examples : list
        List of raw negative examples.
    predicates : list of callables
        Each predicate is a function f(x) -> bool that extracts one binary feature from x.
    tree_config : dict, optional
        Parameters for sklearn.tree.DecisionTreeClassifier (e.g. {'max_depth': 3})
    tree_path : str or None
        If not None, save a PNG image of the trained decision tree to this path.
    
    Returns
    -------
    model : DecisionTreeClassifier
        Trained classifier.
    label_function : callable
        Function that applies the trained model and the predicates to new examples.
    """
    # Combine examples and labels
    all_examples = pos_examples + neg_examples
    y = np.array([1] * len(pos_examples) + [0] * len(neg_examples))
    
    # Compute binary features for each example
    X = np.array([
        [int(pred(x)) for pred in predicates]
        for x in all_examples
    ])
    
    # Default config if not provided
    if tree_config is None:
        tree_config = {}
    
    # Train the decision tree
    clf = DecisionTreeClassifier(**tree_config)
    clf.fit(X, y)
    
    # Optionally save a visualization of the tree
    if tree_path is not None:
        plt.figure(figsize=(10, 6))
        plot_tree(
            clf,
            filled=True,
            feature_names=[f"pred_{i}" for i in range(len(predicates))],
            class_names=["False", "True"]
        )
        plt.title("Decision Tree Labeling Function")
        plt.savefig(tree_path, bbox_inches="tight")
        plt.close()
    
    # Define the labeling function
    def label_function(x):
        feats = np.array([[int(pred(x)) for pred in predicates]])
        return bool(clf.predict(feats)[0])
    
    return clf, label_function


import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from typing import Callable, Optional, Any


def select_best_tree(
    pos_examples: list[Any],
    neg_examples: list[Any],
    predicates: list[Callable[[Any], bool]],
    demos: list[list[Any]],
    n_trees: int = 5,
    base_tree_config: Optional[dict[str, Any]] = None,
    visualize_dir: Optional[str] = None
) -> tuple[DecisionTreeClassifier, Callable[[Any], bool], int]:
    """
    Train multiple trees (using `train_labeling_function`) and select the best one based on:
        (1) Perfect classification accuracy on labeled examples
        (2) Lowest average index of first True activation across demonstrations.

    Parameters
    ----------
    pos_examples : list[Any]
        Raw positive examples (True class).
    neg_examples : list[Any]
        Raw negative examples (False class).
    predicates : list[Callable[[Any], bool]]
        List of feature predicates. Each predicate f(x) -> bool produces one binary feature.
    demos : list[list[Any]]
        List of demonstration sequences. Each sequence is a list of raw states.
    n_trees : int, default=5
        Number of trees to train.
    base_tree_config : Optional[dict[str, Any]], default=None
        Configuration dictionary for DecisionTreeClassifier (e.g. {'max_depth': 3}).
        If None, defaults are used and random_state is varied automatically.
    visualize_dir : Optional[str], default=None
        Directory to save visualizations. If provided, creates directory and
        saves each tree visualization as PNG.

    Returns
    -------
    best_model : DecisionTreeClassifier
        The best classifier.
    best_label_fn : Callable[[Any], bool]
        Function that applies the best classifier to raw examples.
    best_idx : int
        Index (0-based) of the best tree among the trained models.
    """
    all_examples = pos_examples + neg_examples
    y_true = np.array([1] * len(pos_examples) + [0] * len(neg_examples))
    valid_trees: list[tuple[int, dict[str, Any], DecisionTreeClassifier, Callable[[Any], bool]]] = []

    # --- Prepare visualization directory ---
    if visualize_dir is not None:
        os.makedirs(visualize_dir, exist_ok=True)

    # --- Train multiple trees ---
    for i in range(n_trees):
        # Build config for this tree
        if base_tree_config is not None:
            config = dict(base_tree_config)  # shallow copy
            if "random_state" not in config:
                config["random_state"] = i
        else:
            config = {"random_state": i}

        # Visualization path for this tree
        tree_path: Optional[str] = None
        if visualize_dir is not None:
            tree_path = os.path.join(visualize_dir, f"tree_{i}.png")

        # Train (and visualize if requested)
        clf, label_fn = train_labeling_function(
            pos_examples,
            neg_examples,
            predicates,
            tree_config=config,
            tree_path=tree_path
        )

        # Evaluate accuracy
        preds = [label_fn(x) for x in all_examples]
        acc = accuracy_score(y_true, preds)

        if acc == 1.0:
            valid_trees.append((i, config, clf, label_fn))

    if not valid_trees:
        raise ValueError("❌ No decision tree achieved perfect classification on training examples.")

    # --- Evaluate demo behavior ---
    def avg_first_true(label_fn: Callable[[Any], bool]) -> float:
        indices: list[int] = []
        for demo in demos:
            first_true: Optional[int] = None
            for step_idx, state in enumerate(demo):
                if label_fn(state):
                    first_true = step_idx
                    break
            if first_true is not None:
                indices.append(first_true)
            else:
                indices.append(len(demo))  # penalize if never true
        return float(np.mean(indices))

    # --- Score all valid trees ---
    results: list[tuple[float, int, dict[str, Any], DecisionTreeClassifier, Callable[[Any], bool]]] = []
    for idx, config, clf, label_fn in valid_trees:
        avg_idx = avg_first_true(label_fn)
        results.append((avg_idx, idx, config, clf, label_fn))

    # --- Pick the best one ---
    results.sort(key=lambda x: x[0])
    best_avg, best_idx, best_config, best_model, best_fn = results[0]

    print(f"✅ Selected tree #{best_idx} (avg first-True index = {best_avg:.2f})")
    return best_model, best_fn, best_idx

