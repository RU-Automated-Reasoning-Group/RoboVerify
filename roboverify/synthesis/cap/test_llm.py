"""Provider wire compatibility without external model requests."""

import io
import json
import unittest

from .config import ModelConfig
from .llm import HTTPModel, ModelError


class ProviderTests(unittest.TestCase):
    def test_three_chat_providers_share_normalized_output(self):
        for provider in ("openrouter", "openai", "vllm"):
            with self.subTest(provider=provider):
                requests = []

                def opener(request, timeout):
                    requests.append(request)
                    return io.BytesIO(
                        json.dumps(
                            {
                                "choices": [
                                    {
                                        "message": {"content": "Pick(b0)"},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {"total_tokens": 9},
                            }
                        ).encode()
                    )

                client = HTTPModel(
                    ModelConfig(
                        provider=provider,
                        base_url="http://localhost/v1",
                        api_key_env="",
                    ),
                    opener,
                )
                result = client.generate("Generate code", 5)
                self.assertEqual(result.text, "Pick(b0)")
                self.assertEqual(result.usage["total_tokens"], 9)
                self.assertEqual(
                    requests[0].full_url, "http://localhost/v1/chat/completions"
                )
                self.assertEqual(
                    json.loads(requests[0].data)["messages"],
                    [{"role": "user", "content": "Generate code"}],
                )

    def test_responses_adapter_uses_text_message_output_only(self):
        def opener(request, timeout):
            payload = json.loads(request.data)
            self.assertIn("input", payload)
            self.assertIn("max_output_tokens", payload)
            return io.BytesIO(
                json.dumps(
                    {
                        "status": "completed",
                        "output": [
                            {"type": "reasoning", "summary": []},
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "pass"}],
                            },
                        ],
                    }
                ).encode()
            )

        client = HTTPModel(ModelConfig(api_style="responses", api_key_env=""), opener)
        self.assertEqual(client.generate("Generate code", 5).text, "pass")

    def test_unsupported_field_overrides_fail_explicitly(self):
        with self.assertRaises(ModelError):
            HTTPModel(ModelConfig(extra_body={"model": "another"})).payload("Generate")

    def test_credential_value_cannot_be_used_as_variable_name(self):
        with self.assertRaisesRegex(ValueError, "environment-variable name"):
            ModelConfig(api_key_env="not-an-environment-variable")


if __name__ == "__main__":
    unittest.main()
