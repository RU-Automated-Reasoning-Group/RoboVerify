"""Unthrottled failure events, text mirroring, and immutable checkpoints."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest

from .checkpoints import Checkpoints
from .logger import CapLogger


class LoggerTests(unittest.TestCase):
    def test_repeated_failures_and_all_console_output_are_saved(self):
        with tempfile.TemporaryDirectory() as root:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                logger = CapLogger(root)
                for i in range(55):
                    logger.emit("validation_failed", attempt=i)
                logger.close()
            from pathlib import Path

            saved = (Path(root) / "stdout.log").read_text()
            self.assertEqual(saved, output.getvalue())
            events = [
                json.loads(line)
                for line in (logger.root / "timeline.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(e["kind"] == "validation_failed" for e in events), 55)

    def test_checkpoints_reconstruct_sources_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            logger = CapLogger(root, console=False)
            checkpoints = Checkpoints(logger, method="cap_method_1")
            a = checkpoints.save("perform(b0)", reason="initial_source")
            b = checkpoints.save(
                "perform(b0)",
                {"perform": "def perform(b):\n    Pick(b)"},
                reason="execution_bundle",
                validation="valid",
            )
            manifest = json.loads(
                (logger.root / "checkpoints" / b / "manifest.json").read_text()
            )
            self.assertEqual(manifest["parent_checkpoint_id"], a)
            for item in manifest["sources"].values():
                self.assertEqual(
                    hashlib.sha256(
                        (logger.run_dir / item["path"]).read_bytes()
                    ).hexdigest(),
                    item["sha256"],
                )
            with self.assertRaises(FileExistsError):
                logger.artifact(f"checkpoints/{a}/policy.py", "changed")
            logger.close()


if __name__ == "__main__":
    unittest.main()
