"""vLLM logits processor: close the reasoning block instead of ending the turn.

Served Qwen3.8 27B sometimes assigns high probability to the end-of-turn token
while it is still inside ``<think>``, so a request ends with all text in the
reasoning field and no answer (Revision 4.3.1, Section 10). For requests whose
reasoning block is still open, this processor moves the probability of the
stop tokens onto ``</think>``:

    logit[</think>] <- logsumexp(logit[</think>], logit[stop tokens])
    logit[stop tokens] <- -inf

The model's decision to stop reasoning is kept. After ``</think>``, stop
tokens stay blocked until the first non-whitespace answer token, because a
model that wrote its program inside the reasoning may otherwise end the turn
with an empty answer. All other probabilities are unchanged, and nothing
changes once the answer has started or for requests whose prompt does not end
inside an open reasoning block.

Written for the vLLM 0.32 model runner V2 interface. Load with
``vllm serve ... --logits-processors
synthesis.cap.serving.think_close_guard:ThinkCloseGuard`` and ``roboverify/`` on
``PYTHONPATH``. The CaP runtime does not import this module.
"""

from __future__ import annotations

from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

THINK_START = "<think>"
THINK_END = "</think>"
TURN_START = "<|im_start|>"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def reasoning_open(prompt_ids, think_start, think_end, turn_start):
    """True if the prompt's last turn has an unclosed reasoning block."""
    for token in reversed(list(prompt_ids or ())):
        if token == think_end or token == turn_start:
            return False
        if token == think_start:
            return True
    return False


REASONING, AWAITING_ANSWER = 1, 2


def redirect_stop_mass(logits, open_rows, think_end, stops):
    """Move stop-token probability onto ``think_end`` for rows in ``open_rows``."""
    import torch

    current = logits[:, think_end]
    stop_logits = logits[:, stops]
    merged = torch.logsumexp(torch.cat([current.unsqueeze(1), stop_logits], 1), 1)
    logits[:, think_end] = torch.where(open_rows, merged, current)
    logits[:, stops] = torch.where(
        open_rows.unsqueeze(1), torch.full_like(stop_logits, -float("inf")), stop_logits
    )
    return logits


def block_stops(logits, rows, stops):
    """Forbid stop tokens for ``rows``; other tokens keep their logits."""
    import torch

    stop_logits = logits[:, stops]
    logits[:, stops] = torch.where(
        rows.unsqueeze(1), torch.full_like(stop_logits, -float("inf")), stop_logits
    )
    return logits


def next_phase(phase, token, think_end, blank):
    """Phase after committing ``token``: reasoning -> awaiting answer -> free."""
    import torch

    closing = (phase == REASONING) & (token == think_end)
    answered = (phase == AWAITING_ANSWER) & ~blank[token]
    phase = torch.where(closing, torch.full_like(phase, AWAITING_ANSWER), phase)
    return torch.where(answered, torch.zeros_like(phase), phase)


class ThinkCloseGuard(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        import torch
        from transformers import AutoTokenizer

        model = vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model.tokenizer, revision=model.tokenizer_revision
        )
        ids = tokenizer.convert_tokens_to_ids
        self.think_start, self.think_end, self.turn_start = (
            ids(THINK_START),
            ids(THINK_END),
            ids(TURN_START),
        )
        stops = [ids(t) for t in STOP_TOKENS]
        if None in (self.think_start, self.think_end, self.turn_start, *stops):
            raise ValueError("Tokenizer lacks the reasoning or stop delimiters")
        self.req_states = req_states
        device = req_states.device
        self.stops = torch.tensor(stops, dtype=torch.long, device=device)
        # Tokens that decode to whitespace only do not start an answer.
        blank = [
            not text.strip()
            for text in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
        ]
        blank += [False] * max(0, req_states.vocab_size - len(blank))
        self.blank = torch.tensor(blank, dtype=torch.bool, device=device)
        self.blank[self.think_end] = True
        # Per request slot: phase (device), and whether the slot started inside
        # reasoning (host), to skip batches without such requests.
        self.phase = torch.zeros(
            req_states.max_num_reqs, dtype=torch.int8, device=device
        )
        self.tracked = [False] * req_states.max_num_reqs
        self.pending: list[int] = []

    def add_request(self, req_idx, sampling_params) -> bool:
        self.pending.append(req_idx)
        return True

    def apply_staged_writes(self) -> None:
        # Runs once per admission batch, after the prompt is on the device.
        states = self.req_states
        for slot in self.pending:
            prompt_len = int(states.prompt_len.np[slot])
            # Scan the prompt back to its last turn start (one read per request).
            prompt = states.all_token_ids.gpu[slot, :prompt_len].tolist()
            opened = reasoning_open(
                prompt, self.think_start, self.think_end, self.turn_start
            )
            phase = REASONING if opened else 0
            if opened:
                # A preempted request resumes with its output so far.
                total = int(states.total_len.gpu[slot])
                for token in states.all_token_ids.gpu[slot, prompt_len:total].tolist():
                    if phase == REASONING and token == self.think_end:
                        phase = AWAITING_ANSWER
                    elif phase == AWAITING_ANSWER and not bool(self.blank[token]):
                        phase = 0
                        break
            self.phase[slot] = phase
            self.tracked[slot] = opened
        self.pending.clear()

    def apply(self, logits, ctx):
        if not any(self.tracked[s] for s in ctx.idx_mapping_np):
            return logits
        slots = ctx.expanded_idx_mapping
        # The input token of a decode row is the token sampled in the last step.
        phase = next_phase(self.phase[slots], ctx.input_ids, self.think_end, self.blank)
        self.phase[slots] = phase
        logits = redirect_stop_mass(
            logits, phase == REASONING, self.think_end, self.stops
        )
        return block_stops(logits, phase == AWAITING_ANSWER, self.stops)


__all__ = (
    "ThinkCloseGuard",
    "block_stops",
    "next_phase",
    "reasoning_open",
    "redirect_stop_mass",
)
