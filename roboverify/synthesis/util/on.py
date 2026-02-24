import numpy as np

BLOCK_LENGTH = (
    0.025 * 2
)  # (0.025, 0.025, 0.025) in the xml file of the gym env is half length


def on_star_eval(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def on(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2 < 1.5 * BLOCK_LENGTH
    )


def on_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def top_implementation(block, all_blocks) -> bool:
    """Check if a block is on top"""
    top_flag = True
    for other_block in all_blocks:
        if other_block != block and on_star_implementation(other_block, block):
            top_flag = False
            break
    return top_flag


def get_block_pos(obs, block_id):
    start_idx = 10 + 12 * block_id
    end_idx = start_idx + 3
    return np.array(obs[start_idx:end_idx])


def print_block_layout(obs, num_block):
    for i in range(0, num_block):
        for j in range(0, num_block):
            print(f"on({i}, {j})", on(get_block_pos(obs, i), get_block_pos(obs, j)))
