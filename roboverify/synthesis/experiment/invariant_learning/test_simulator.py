"""Generated scenes only: never depend on saved demonstration archives."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthesis.cfg.collection import record_execution
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import restore
from synthesis.experiment.invariant_learning.tasks import StackExperiment
from synthesis.experiment.invariant_learning.witness import WitnessSearch
from synthesis.mcmc.synthesis import make_roboverify_stack_env


class SimulatorTests(unittest.TestCase):
    def test_generated_snapshot_preparation_archive_and_replay(self):
        with contextlib.redirect_stdout(io.StringIO()):
            task = StackExperiment("synthesis.examples.stack:build_program")
            query = task.search_query(2, 10000)
            query.solver.add(query.coverage)
            import z3

            self.assertEqual(query.solver.check(), z3.sat)
            search = WitnessSearch(
                "found", size=2, query=query, witness=query.decode(query.solver.model())
            )
            trace = task.execute(search, seed=0, timeout_seconds=60, video_path=None)
            self.assertTrue(task.validate(trace), trace.metadata)
            self.assertEqual(
                trace.metadata["counterexample_initialization"]["settling_steps"], 50
            )
            self.assertEqual(
                trace.metadata["initialization"],
                {"source": "snapshot", "settling_steps": 0},
            )
            self.assertEqual(len(task.learning_states(trace)), 2)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.npz"
                save_traces(path, [trace])
                restored = load_traces(path, require_valid=True)[0]
            env = make_roboverify_stack_env(num_blocks=2)
            try:
                observation = restore(env, restored.snapshots[0])
                np.testing.assert_allclose(
                    observation, trace.states[0], atol=1e-8, rtol=0
                )
            finally:
                env.close()
            replay = record_execution(
                task.definition,
                seed=999,
                num_blocks=2,
                initial_snapshot=restored.snapshots[0],
                max_loop_iterations=1,
            )
            self.assertTrue(task.validate(replay), replay.metadata)
            np.testing.assert_allclose(replay.states, trace.states, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                replay.actions[0], trace.actions[0], atol=1e-8, rtol=0
            )
            for state in task.learning_states(replay):
                self.assertEqual(
                    state.entry_positions,
                    task.learning_states(trace)[0].entry_positions,
                )


if __name__ == "__main__":
    unittest.main()
