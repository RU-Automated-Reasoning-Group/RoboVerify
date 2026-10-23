"""Observation-relative coordinate operands, resolved once per instruction."""

from enum import Enum


class ObservationReference(Enum):
    MIN_Y = "Y_min"
    MAX_Z = "Z_m"
    GRIPPER = "gripper"


OBSERVATION_REFERENCE_NAMES = frozenset(r.value for r in ObservationReference)
OBSERVATION_REFERENCE_MODE = "physical-extrema-and-gripper-at-instruction-entry-v1"
