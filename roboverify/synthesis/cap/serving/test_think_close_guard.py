import importlib.util
import unittest
from types import SimpleNamespace

HAVE_VLLM = importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(HAVE_VLLM, "runs in the vLLM serving environment")
class ThinkCloseGuardTest(unittest.TestCase):
    def test_reasoning_open_detection(self):
        from .think_close_guard import reasoning_open

        self.assertTrue(reasoning_open([12, 5, 10, 3], 10, 11, 12))
        self.assertFalse(reasoning_open([12, 5, 10, 3, 11, 4], 10, 11, 12))
        self.assertFalse(reasoning_open([10, 3, 12, 5], 10, 11, 12))
        self.assertFalse(reasoning_open([], 10, 11, 12))

    def test_stop_mass_moves_to_think_end_until_closed(self):
        import numpy as np
        import torch

        from .think_close_guard import ThinkCloseGuard

        tokens = torch.zeros((3, 16), dtype=torch.long)
        tokens[0, :3] = torch.tensor([12, 10, 3])  # slot 0: reasoning open
        tokens[1, :3] = torch.tensor([12, 5, 3])  # slot 1: no reasoning block
        tokens[2, :5] = torch.tensor([12, 10, 3, 11, 4])  # slot 2: resumed, closed
        states = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=3,
            prompt_len=SimpleNamespace(np=np.array([3, 3, 3])),
            total_len=SimpleNamespace(gpu=torch.tensor([3, 3, 5])),
            all_token_ids=SimpleNamespace(gpu=tokens),
        )
        guard = ThinkCloseGuard.__new__(ThinkCloseGuard)
        guard.think_start, guard.think_end, guard.turn_start = 10, 11, 12
        guard.req_states = states
        guard.stops = torch.tensor([1, 2])
        guard.phase = torch.zeros(3, dtype=torch.int8)
        guard.blank = torch.zeros(16, dtype=torch.bool)
        guard.blank[[11, 13]] = True  # </think> and a newline token
        guard.tracked = [False] * 3
        guard.pending = []
        for slot in range(3):
            guard.add_request(slot, None)
        guard.apply_staged_writes()
        self.assertEqual(guard.phase.tolist(), [1, 0, 0])

        probs = torch.tensor(
            [0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0, 0, 0]
        )
        base = torch.log(probs + 1e-30).repeat(3, 1)
        ctx = SimpleNamespace(
            expanded_idx_mapping=torch.tensor([0, 1, 2]),
            idx_mapping_np=np.array([0, 1, 2]),
            input_ids=torch.tensor([3, 3, 4]),
        )
        out = guard.apply(base.clone(), ctx)
        first = torch.softmax(out[0], 0)
        self.assertAlmostEqual(float(first[11]), 0.6, places=5)
        self.assertEqual(float(first[1]) + float(first[2]), 0.0)
        self.assertAlmostEqual(float(first[12]), 0.4, places=5)
        self.assertTrue(torch.equal(out[1:], base[1:]))
        # Slot 0 samples </think>: stop tokens stay blocked, nothing is moved.
        ctx.input_ids = torch.tensor([11, 3, 4])
        out = guard.apply(base.clone(), ctx)
        self.assertEqual(float(torch.softmax(out[0], 0)[[1, 2]].sum()), 0.0)
        self.assertEqual(float(out[0, 11]), float(base[0, 11]))
        # A blank token keeps the block; the first visible token lifts it.
        ctx.input_ids = torch.tensor([13, 3, 4])
        self.assertEqual(
            float(torch.softmax(guard.apply(base.clone(), ctx)[0], 0)[1]), 0.0
        )
        ctx.input_ids = torch.tensor([5, 3, 4])
        self.assertTrue(torch.equal(guard.apply(base.clone(), ctx), base))
        self.assertEqual(guard.phase.tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
