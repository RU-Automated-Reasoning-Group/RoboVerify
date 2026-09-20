"""Record loop-head traces from the existing physical Stack program."""

import argparse

from synthesis.entry.verify_stack_with_learned_invariant import build_stack_programs
from synthesis.inference_lib.demo_store import DemoStore
from synthesis.mcmc.synthesis import (
    make_roboverify_stack_env,
    preserved_global_rng,
    set_np_seed,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def collect_stack_loop_traces(*, num_blocks=4, seeds=(0, 1), max_iters=10):
    """Return observed states and rollout outcomes; no verification is implied."""
    context = HighLevelContext(mode="declare")
    _, program = build_stack_programs(context, max_iters=max_iters)
    store = DemoStore()
    outcomes = []
    with preserved_global_rng():
        for seed in seeds:
            set_np_seed(int(seed))
            env = make_roboverify_stack_env(num_blocks=num_blocks)
            try:
                before = len(store)
                traj = program.eval(env, on_loop_head=store.add)
                outcomes.append(
                    {
                        "seed": int(seed),
                        "loop_heads": len(store) - before,
                        "success": bool(env.env._is_success(traj[-1])),
                    }
                )
            finally:
                env.close()
    return store, outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="Destination DemoStore JSON file."
    )
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-iters", type=int, default=10)
    args = parser.parse_args()
    store, outcomes = collect_stack_loop_traces(
        num_blocks=args.num_blocks, seeds=args.seeds, max_iters=args.max_iters
    )
    if not len(store):
        parser.error("No loop bodies were entered; no inference dataset was recorded")
    store.save(args.output)
    print(f"Saved {len(store)} loop-head states (loop ID '1') to {args.output}")
    print(outcomes)
