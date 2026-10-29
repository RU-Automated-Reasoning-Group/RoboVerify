"""Column reachability within one primitive's step budget (Revision 9, §3.3).

From the arm's start posture, the gripper rises quickly to about 1.0 m and
then only creeps (about 0.2 mm per step, with forearm/wrist self-contact
observed): hundreds of servo steps later it is higher, and from that
reconfigured posture it moves freely again. A Move has 50 steps, so a column
is reachable when, with an empty gripper in an otherwise empty relaxed scene:

- one Move from the start posture reaches the hover point, `CLEARANCE` above
  the column's top centre;
- one Move descends from there to the top centre, where a block is grasped or
  released;
- one Move lifts from the top centre back to the hover point, which a carried
  block needs to clear the column below.

All Moves use Move's own controller, budget and 2 mm tolerance. A long chain
of millimetre Moves could creep higher; the criterion deliberately excludes
that. The empty gripper is a lower bound on the effort of carrying a block.
"""

from dataclasses import dataclass

import numpy as np

CLEARANCE = 0.05
PRIMITIVE_LIMIT = 50
# Parked blocks stay this far from every probed column.
PARK_CANDIDATES = ((0.35, -0.75), (0.35, 0.75), (0.0, -0.75), (0.0, 0.75))
PARK_DISTANCE = 0.15


@dataclass(frozen=True)
class ColumnReach:
    xy: tuple
    top_z: float
    hover: bool
    down: bool
    up: bool
    hover_error: float
    down_error: float
    up_error: float

    @property
    def reachable(self):
        return self.hover and self.down and self.up


class PrimitiveReach:
    """One empty relaxed environment reused for many column checks."""

    def __init__(self, limit=PRIMITIVE_LIMIT, clearance=CLEARANCE):
        from synthesis.cfg.reset import inner_env
        from synthesis.mcmc.synthesis import make_roboverify_env

        self.limit, self.clearance = limit, clearance
        self.env = make_roboverify_env(
            "stack", num_blocks=1, relax_boundaries_for_generalization_test=True
        )
        self.env.reset(seed=0)
        self.inner = inner_env(self.env)
        self.base = np.asarray(self.inner.robot_base_xy, dtype=float)
        self.start = self.inner.sim.get_state()

    def _reset(self, xy):
        self.inner.sim.set_state(self.start)
        park = max(
            (self.base + c for c in PARK_CANDIDATES),
            key=lambda p: np.linalg.norm(p - xy),
        )
        if np.linalg.norm(park - xy) < PARK_DISTANCE:
            raise ValueError("No parking place clear of the probed column")
        qpos = self.inner.sim.data.get_joint_qpos("object0:joint").copy()
        qpos[:3] = [park[0], park[1], self.inner.height_offset]
        self.inner.sim.data.set_joint_qpos("object0:joint", qpos)
        self.inner.sim.forward()

    def _move(self, target, limit):
        from synthesis.api.control import PrimitiveController

        controller = PrimitiveController(self.env, [], limit=limit)
        converged = controller.move(np.asarray(target, dtype=float))
        return converged, float(controller.result.position_error)

    def column(self, xy, top_z):
        xy = np.asarray(xy, dtype=float)
        top = np.r_[xy, top_z]
        hover = top + [0.0, 0.0, self.clearance]
        self._reset(xy)
        hover_ok, hover_error = self._move(hover, self.limit)
        down_ok = up_ok = False
        down_error = up_error = float("nan")
        if hover_ok:
            down_ok, down_error = self._move(top, self.limit)
        if down_ok:
            up_ok, up_error = self._move(hover, self.limit)
        return ColumnReach(
            tuple(float(v) for v in xy),
            float(top_z),
            hover_ok,
            down_ok,
            up_ok,
            hover_error,
            down_error,
            up_error,
        )

    def close(self):
        self.env.close()
