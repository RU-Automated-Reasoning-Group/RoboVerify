"""Reuse successful block searches only while their exact inputs stay unchanged."""

import hashlib
import pickle

from synthesis.api.program import Program
from synthesis.cfg.program_source import program_fingerprint
from synthesis.cfg.region import BlockRegion


def _bindings(values):
    return None if values is None else tuple(sorted(values.items()))


def _trace_content(segment):
    trace = segment.trace
    # DemoTrace is mutable. Hash the actual imitation/frozen-entry/reset inputs
    # as well as holding/table metadata, so in-place edits invalidate reuse.
    # Pickle is used only to encode local values for hashing, never to load data.
    values = (
        segment.states,
        trace.states[segment.entry_index],
        (
            (trace.snapshots[0], trace.snapshots[segment.t_start])
            if len(trace.snapshots)
            else ()
        ),
        trace.actions,
        trace.metadata,
        trace.seed,
        trace.task,
        trace.num_blocks,
    )
    return hashlib.blake2b(pickle.dumps(values, protocol=5), digest_size=16).digest()


def _signature(node, demos, post):
    return (
        node.name,
        node.synthesis_approach,
        tuple(sorted(node.available_scope)),
        post.sexpr() if hasattr(post, "sexpr") else repr(post),
        tuple(
            (
                id(segment.trace),
                _trace_content(segment),
                segment.demo_idx,
                segment.t_start,
                segment.t_end,
                segment.entry_index,
                _bindings(segment.bindings),
                _bindings(segment.final_bindings),
                _bindings(segment.goal_bindings),
            )
            for segment in demos
        ),
    )


def _fingerprint(region):
    return program_fingerprint(Program(len(region.physical), list(region.physical)))


class SuccessfulRealizationCache:
    """Per-run cache for deterministic successful physical BlockRegion search.

    The owning node is retained, so an unrelated later object cannot reuse its
    Python ID. Loop realizations and penalty-guided repairs are never cached.
    """

    def __init__(self):
        self._entries = {}

    def get(self, node, demos, post, *, penalty=None):
        saved = self._entries.get(id(node))
        if saved is None:
            return None
        owner, signature, fingerprint = saved
        if (
            penalty is not None
            or owner is not node
            or not isinstance(node.region, BlockRegion)
            or signature != _signature(node, demos, post)
            or fingerprint != _fingerprint(node.region)
        ):
            self._entries.pop(id(node), None)
            return None
        return node.region

    def remember(self, node, demos, post, region, *, accepted, penalty=None):
        self._entries.pop(id(node), None)
        if accepted and penalty is None and isinstance(region, BlockRegion):
            self._entries[id(node)] = (
                node,
                _signature(node, demos, post),
                _fingerprint(region),
            )
