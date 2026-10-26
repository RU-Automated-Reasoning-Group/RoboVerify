# CaP module docs

Short descriptions of the modules behind the CaP framework (Method 2 with failure recovery), with links to the code. The revision documents record the design history; these pages describe what the code does now.

| Pseudocode name | Module doc | Main code |
| --- | --- | --- |
| `cap_trial`, `config` | [Entry points and configuration](entry_and_config.md) | [runner.py](../../runner.py), [config.py](../../config.py), [configs/](../../configs/) |
| `compact_demonstrations` | [Demonstration compactor](demonstration_compactor.md) | [demonstrations.py](../../demonstrations.py) |
| `build_prompt` | [Prompt builder](prompt_builder.md) | [prompt_package.py](../../prompt_package.py), [prompts/](../../prompts/) |
| `llm` | [LLM client](llm_client.md) | [llm.py](../../llm.py) |
| `synthesize` | [Synthesizer: Method 2](synthesizer_method2.md) | [cap_method_2/method.py](../../cap_method_2/method.py) |
| `checker` | [DSL checker](dsl_checker.md) | [checker.py](../../checker.py), [validation.py](../../validation.py) |
| `rollout` | [Rollout](rollout.md) | [adapter.py](../../adapter.py), [runtime.py](../../runtime.py), [recording.py](../../recording.py) |
| `evaluate` | [Evaluator](evaluator.md) | [tasks.py](../../tasks.py) |
| `build_feedback` | [Feedback builder](feedback_builder.md) | [recovery.py](../../recovery.py), [primitive_feedback.py](../../primitive_feedback.py) |
| recovery loop, `budget` | [Recovery loop and budgets](recovery_loop.md) | [recovery.py](../../recovery.py), [budget.py](../../budget.py) |
| `logger` | [Logger and artifacts](logger.md) | [logger.py](../../logger.py), [checkpoints.py](../../checkpoints.py) |

Line links point to the code at commit `3ef0fc7` (2026-10-26). Update them when the code moves.
