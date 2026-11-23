"""Maintained observation-independent MuJoCo contact/frame audit for Stack/Unstack transfer candidates.

No teacher controls, phase labels or inverse target reconstruction are used.
Contact geometry is audited at every 2 ms MuJoCo physics substep; callbacks only
read simulator state. Existing fixed robot/environment overlap is reported as a
baseline exception, not advertised as globally collision-free scene geometry.
"""

from __future__ import annotations

import ctypes
import time
from collections import Counter

import numpy as np

from synthesis.api.instructions import (
    Assign,
    MoveByName,
    PickByName,
    ReleaseByName,
    Skip,
)
from synthesis.api.program import Program
from synthesis.cfg.reset import collect_recording, inner_env, restore
from synthesis.util.on import on, scattered_implementation

from .data import positions


class ContactObserver:
    def __init__(
        self,
        env,
        example,
        *,
        contact_distance_tolerance_m=1e-6,
        geometry_tolerance_m=0.002,
        drift_tolerance_m=0.001,
    ):
        self.env = env
        self.inner = inner_env(env)
        self.sim = self.inner.sim
        self.model = self.sim.model
        self.data = self.sim.data
        self.example = example
        self.task = example["task"]
        self.mover = int(example["bindings"]["b_prime"])
        self.target = int(example["bindings"].get("b", example["bindings"]["b0"]))
        self.n = example["num_blocks"]
        self.distance_tol = float(contact_distance_tolerance_m)
        self.geometry_tol = float(geometry_tolerance_m)
        self.drift_tol = float(drift_tolerance_m)
        self.names = [
            self.model.geom_id2name(i) or f"unnamed_geom_{i}"
            for i in range(self.model.ngeom)
        ]
        self.objects = {self.model.geom_name2id(f"object{i}"): i for i in range(self.n)}
        self.object_geoms = [
            self.model.geom_name2id(f"object{i}") for i in range(self.n)
        ]
        self.fingers = {
            i for i, n in enumerate(self.names) if n.endswith("gripper_finger_link")
        }
        self.table = self.model.geom_name2id("table0")
        self.table_top = float(
            self.data.geom_xpos[self.table, 2]
            + (
                0.0
                if int(self.model.geom_type[self.table]) == 0
                else np.sum(
                    np.abs(self.data.geom_xmat[self.table].reshape(3, 3)[2])
                    * self.model.geom_size[self.table]
                )
            )
        )
        self.initial = np.array(self.data.geom_xpos[self.object_geoms], copy=True)
        self.max_drift = np.zeros(self.n)
        self.max_height_gain = 0.0
        self.lifted = False
        self.saw_finger_source = False
        self.baseline_background = {}
        self.initial_support_pairs = set()
        self.initial_table_objects = set()
        self.initial_mover_support_pairs = set()
        for c in self.data.contact[: self.data.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            pair = tuple(sorted((a, b)))
            if a == self.table and b in self.objects and abs(c.frame[2]) >= 0.9:
                self.initial_table_objects.add(self.objects[b])
            if b == self.table and a in self.objects and abs(c.frame[2]) >= 0.9:
                self.initial_table_objects.add(self.objects[a])
            if a not in self.objects and b not in self.objects:
                self.baseline_background[pair] = min(
                    self.baseline_background.get(pair, 0.0), float(c.dist)
                )
            if a in self.objects and b in self.objects and abs(c.frame[2]) >= 0.9:
                oa, ob = self.objects[a], self.objects[b]
                if self.mover not in (oa, ob):
                    self.initial_support_pairs.add(pair)
                else:
                    other = ob if oa == self.mover else oa
                    if self.initial[self.mover, 2] > self.initial[other, 2] + 0.025:
                        self.initial_mover_support_pairs.add(pair)
        # Exact-touch reset pairs can be absent from MuJoCo's current contact list.
        # Freeze only near-aligned entry support pairs; never learn permissions later.
        for upper in range(self.n):
            for lower in range(self.n):
                delta = self.initial[upper] - self.initial[lower]
                if (
                    upper != lower
                    and abs(delta[2] - 0.05) <= 0.001
                    and np.max(np.abs(delta[:2])) <= 0.002
                ):
                    pair = tuple(
                        sorted((self.object_geoms[upper], self.object_geoms[lower]))
                    )
                    if self.mover not in (upper, lower):
                        self.initial_support_pairs.add(pair)
                    elif upper == self.mover:
                        self.initial_mover_support_pairs.add(pair)
        self.counts = Counter()
        self.pairs = {}
        self.samples = 0
        self.unintended_samples = 0
        self.unintended_contact_events = 0
        self.examples = []
        self.callback_error = None
        self.old_callback = None
        self.begin_time = float(self.data.time)

    def _top_support(self, upper, lower, c):
        u = self.data.geom_xpos[self.object_geoms[upper]]
        l = self.data.geom_xpos[self.object_geoms[lower]]
        t = self.geometry_tol
        return bool(
            abs(c.frame[2]) >= 0.9
            and abs(u[2] - l[2] - 0.05) <= 2 * t
            and abs(c.pos[2] - (l[2] + 0.025)) <= t
            and abs(c.pos[2] - (u[2] - 0.025)) <= t
            and np.max(np.abs(u[:2] - l[:2])) <= 0.025 + t
        )

    def classify(self, c):
        a, b = int(c.geom1), int(c.geom2)
        pair = tuple(sorted((a, b)))
        oa = self.objects.get(a)
        ob = self.objects.get(b)
        if oa is None and ob is None:
            if (
                pair in self.baseline_background
                and c.dist >= self.baseline_background[pair] - self.geometry_tol
            ):
                return "baseline_robot_environment_contact"
            return "unintended_robot_environment_contact"
        if oa is not None and ob is not None:
            if (
                self.task == "stack"
                and {oa, ob} == {self.mover, self.target}
                and self._top_support(self.mover, self.target, c)
            ):
                return "intended_mover_target_top_support"
            if (
                self.task == "unstack"
                and not self.lifted
                and pair in self.initial_mover_support_pairs
            ):
                other = ob if oa == self.mover else oa
                if self._top_support(self.mover, other, c):
                    return "initial_mover_block_support"
            if pair in self.initial_support_pairs:
                upper, lower = (
                    (oa, ob) if self.initial[oa, 2] >= self.initial[ob, 2] else (ob, oa)
                )
                if self._top_support(upper, lower, c):
                    return "existing_nonmover_top_support"
            return "unintended_block_contact"
        obj, other = (oa, b) if oa is not None else (ob, a)
        p = self.data.geom_xpos[self.object_geoms[obj]]
        if other in self.fingers and obj == self.mover:
            if (
                abs(c.frame[2]) <= 0.35
                and abs(c.pos[2] - p[2]) <= 0.025 + self.geometry_tol
            ):
                self.saw_finger_source = True
                return "intended_mover_finger_grasp"
            return "unintended_mover_finger_geometry"
        if other == self.table:
            top = (
                abs(c.frame[2]) >= 0.9
                and abs(c.pos[2] - self.table_top) <= self.geometry_tol
                and abs(p[2] - 0.025 - self.table_top) <= 2 * self.geometry_tol
            )
            if top and obj != self.mover and obj in self.initial_table_objects:
                return "existing_nonmover_table_support"
            if top and obj == self.mover and not self.lifted:
                return "initial_mover_table_support"
            if top and obj == self.mover and self.task == "unstack" and self.lifted:
                if all(
                    scattered_implementation(
                        p, self.data.geom_xpos[self.object_geoms[i]]
                    )
                    for i in range(self.n)
                    if i != self.mover
                ):
                    return "intended_mover_table_support"
            return "unintended_block_table_contact"
        return "unintended_robot_block_contact"

    def sample(self):
        self.samples += 1
        current = self.data.geom_xpos[self.object_geoms]
        self.max_drift = np.maximum(
            self.max_drift, np.linalg.norm(current - self.initial, axis=1)
        )
        gain = float(current[self.mover, 2] - self.initial[self.mover, 2])
        self.max_height_gain = max(self.max_height_gain, gain)
        self.lifted |= gain > 0.03
        bad = False
        for c in self.data.contact[: self.data.ncon]:
            if c.dist > self.distance_tol:
                continue
            category = self.classify(c)
            self.counts[category] += 1
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            key = (category, pair)
            if key not in self.pairs:
                self.pairs[key] = {
                    "category": category,
                    "geoms": [self.names[i] for i in pair],
                    "events": 0,
                    "max_penetration_m": 0.0,
                    "first_time_s": float(self.data.time) - self.begin_time,
                    "last_time_s": 0.0,
                }
            row = self.pairs[key]
            row["events"] += 1
            row["max_penetration_m"] = max(row["max_penetration_m"], -float(c.dist))
            row["last_time_s"] = float(self.data.time) - self.begin_time
            if category.startswith("unintended"):
                bad = True
                self.unintended_contact_events += 1
                if sum(x["category"] == category for x in self.examples) < 8:
                    self.examples.append(
                        {
                            "category": category,
                            "geoms": row["geoms"],
                            "time_s": float(self.data.time) - self.begin_time,
                            "distance_m": float(c.dist),
                            "contact_position": c.pos.tolist(),
                            "normal": c.frame[:3].tolist(),
                        }
                    )
        self.unintended_samples += int(bad)

    def __enter__(self):
        self.old_callback = int(self.sim.substep_callback_ptr)
        if self.old_callback:
            raise RuntimeError(
                "Refusing to replace an existing simulator substep callback"
            )

        def callback(m, d):
            if self.callback_error:
                return
            try:
                self.sample()
            except BaseException as exc:
                self.callback_error = f"{type(exc).__name__}: {exc}"

        self._callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
            callback
        )
        self.sim.set_substep_callback(
            int(ctypes.cast(self._callback, ctypes.c_void_p).value)
        )
        self.sample()
        return self

    def __exit__(self, *args):
        try:
            self.sample()
        finally:
            self.sim.set_substep_callback(self.old_callback or None)

    def report(self):
        nonmovers = {
            str(i): {
                "max_displacement_m": float(self.max_drift[i]),
                "final_displacement_m": float(
                    np.linalg.norm(
                        self.data.geom_xpos[self.object_geoms[i]] - self.initial[i]
                    )
                ),
            }
            for i in range(self.n)
            if i != self.mover
        }
        max_drift = max(
            (x["max_displacement_m"] for x in nonmovers.values()), default=0.0
        )
        return {
            "physics_samples": self.samples,
            "physics_timestep_seconds": float(self.model.opt.timestep),
            "contact_counts": dict(self.counts),
            "contact_pairs": list(self.pairs.values()),
            "unintended_contact_events": self.unintended_contact_events,
            "unintended_contact_samples": self.unintended_samples,
            "no_unintended_contacts": self.unintended_contact_events == 0
            and self.callback_error is None,
            "unintended_examples": self.examples,
            "nonmovers": nonmovers,
            "max_nonmover_displacement_m": max_drift,
            "frame_within_tolerance": max_drift <= self.drift_tol,
            "saw_mover_finger_contact": self.saw_finger_source,
            "max_mover_height_gain_m": self.max_height_gain,
            "lift_observed": self.lifted,
            "callback_error": self.callback_error,
            "tolerances": {
                "contact_distance_m": self.distance_tol,
                "contact_geometry_m": self.geometry_tol,
                "nonmover_drift_m": self.drift_tol,
                "support_normal_abs_z_min": 0.9,
                "finger_normal_abs_z_max": 0.35,
            },
            "scope": "Every MuJoCo physics substep, active contact distance <=1 micrometre. Only geometrically appropriate finger/source and designated placement contacts are intended. Unstack permits initial mover/support contact before lift and table deposition after lift only when scattered from all nonmovers. Existing block support and static robot/environment overlaps are recorded exceptions; new/deeper background contact is rejected. No dynamics or all-scene proof.",
        }


def decode(rows):
    out = []
    for row in rows:
        kind = row["kind"]
        a = row.get("operands", [])
        v = row.get("offsets", [])
        if kind == "Pick":
            out.append(PickByName(*a))
        elif kind == "Move":
            out.append(MoveByName(*a, target_offset=v))
        elif kind == "Release":
            out.append(ReleaseByName(*a, target_z=v[0]))
        elif kind == "Assign":
            out.append(Assign(*(a or [row["left"], row["right"]])))
        elif kind == "Skip":
            out.append(Skip(skip_steps=int(row.get("skip_steps", 20))))
        else:
            raise ValueError(f"Unsupported candidate instruction {kind}")
    return out


def settle(env, initial, steps=50):
    states = []
    s = np.asarray(initial)
    anchor = s[:3].copy()
    for _ in range(steps):
        v = 14 * (anchor - s[:3])
        norm = np.linalg.norm(v)
        if norm > 0.55:
            v *= 0.55 / norm
        s = np.asarray(env.step(np.r_[v, 0.2])[0])
        states.append(s.copy())
    return states


def run_rows(env, rows, example, *, settle_steps=50, drift_tolerance_m=0.005):
    began = time.monotonic()
    a = int(example["bindings"]["b_prime"])
    first = restore(env, example["initial_snapshot"])
    inner_env(env).symbolic_name_to_box_id.update(example["bindings"])
    observer = ContactObserver(env, example, drift_tolerance_m=drift_tolerance_m)
    states = [first]
    recording = None
    error = None
    execution_end_index = 0
    released = False
    aperture = None
    end_finger_contacts = None
    try:
        with observer:
            states, recording = collect_recording(
                Program(len(rows), decode(rows)), env, initial_observation=first
            )
            states = list(states)
            execution_end_index = len(states) - 1
            aperture = float(np.sum(states[-1][3:5]))
            end_finger_contacts = sum(
                1
                for c in observer.data.contact[: observer.data.ncon]
                if c.dist <= observer.distance_tol
                and (
                    (
                        int(c.geom1) in observer.fingers
                        and observer.objects.get(int(c.geom2)) == a
                    )
                    or (
                        int(c.geom2) in observer.fingers
                        and observer.objects.get(int(c.geom1)) == a
                    )
                )
            )
            released = aperture > 0.053 and end_finger_contacts == 0
            states.extend(settle(env, states[-1], settle_steps))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    diagnostics = observer.report()
    a = int(example["bindings"]["b_prime"])

    def goal(s):
        p = positions(s, example["num_blocks"])
        if example["task"] == "stack":
            return bool(on(p[a], p[int(example["bindings"]["b"])]))
        return bool(
            abs(p[a][2] - (observer.table_top + 0.025)) <= 0.001
            and all(scattered_implementation(p[a], q) for i, q in p.items() if i != a)
        )

    on_values = [goal(s) for s in states[execution_end_index:]]
    controls = (
        []
        if recording is None
        else [
            e.get("control", {})
            for e in recording.events
            if e["kind"] == "instruction_end" and "control" in e
        ]
    )
    diagnostics.update(
        error=error,
        released_at_execution_end=released,
        execution_end_gripper_aperture_m=aperture,
        finger_source_contacts_at_execution_end=end_finger_contacts,
        controls_converged=bool(controls)
        and all(c.get("converged", False) for c in controls),
        controller_results=controls,
        goal_name=(
            "ON(source,target)"
            if example["task"] == "stack"
            else "source_on_physical_table_and_Scattered_all"
        ),
        table_surface_height_m=observer.table_top,
        settled_goal=bool(on_values[-1]),
        stable_goal_last_10=bool(
            len(on_values) >= min(10, settle_steps + 1) and all(on_values[-10:])
        ),
        settling_goal_count=sum(on_values),
        settling_observations=len(on_values),
        execution_end_index=execution_end_index,
        final_settle_steps=settle_steps,
        elapsed_seconds=time.monotonic() - began,
    )
    diagnostics["eligible"] = bool(
        error is None
        and diagnostics["controls_converged"]
        and diagnostics["released_at_execution_end"]
        and diagnostics["stable_goal_last_10"]
        and diagnostics["no_unintended_contacts"]
        and diagnostics["frame_within_tolerance"]
        and diagnostics["saw_mover_finger_contact"]
        and diagnostics["lift_observed"]
    )
    diagnostics["strict_1mm_eligible"] = bool(
        diagnostics["eligible"] and diagnostics["max_nonmover_displacement_m"] <= 0.001
    )
    return np.asarray(states), diagnostics
