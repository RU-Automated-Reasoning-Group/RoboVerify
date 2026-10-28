"""Demonstration input for CaP (Revision 6): loading, compaction and accounting.

Only observed behaviour reaches the model. The loader reads each trajectory's
``states`` and ``actions``, its task, block count, initial input bindings and,
for goal tasks, its goal layout. It never reads the generating ``program``,
runtime ``events``, ``task_spec`` text, fingerprints or simulator snapshots of
a demonstration archive (docs/revision_6_add_demonstration.md, Section 3.1).

Compaction (``demo-keyframes-v1``, Section 3.3) keeps two kinds of frames:

- event frames: the first and last state and every state at which the gripper
  command switches between open and closed, with all block positions and the
  DSL predicates that changed since the previous event frame;
- motion waypoints: corners of the gripper's 3D path found by
  Ramer-Douglas-Peucker simplification, with the gripper position and the held
  block only.
"""

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .contracts import GOAL_TASKS, TABLE_TASKS, BudgetExceeded, CapError

SERIALIZATION_VERSION = "demo-keyframes-v1"
TOKEN_COUNTERS = ("server_tokenize", "utf8_bytes")
# A closed gripper holds the nearest block whose centre lies within this
# distance (3D) of the gripper; grasped blocks stay within about 3 mm.
HOLD_DISTANCE = 0.03
# Two scenes are the same when every block centre agrees within this distance.
SAME_SCENE_DISTANCE = 0.001
PREDICATE_ORDER = ("ON", "ON_star", "Higher", "Scattered", "ON_star_zero")
TRACE_ID = re.compile(r"d(0|[1-9][0-9]*)")


class DemonstrationError(CapError):
    """Incompatible or malformed demonstrations, or an evaluation-scene overlap."""


class DemonstrationBudgetExceeded(BudgetExceeded):
    """The rendered demonstration block exceeds demonstrations.max_tokens."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class Demonstration:
    """The fields of one archived trajectory that CaP may use."""

    trace_id: str
    seed: int | None
    task: str
    num_blocks: int
    states: object = field(repr=False)  # (T, D) float array
    actions: object = field(repr=False)  # (T - 1, 4) float array
    initial_bindings: dict = field(default_factory=dict)
    goal_layout: dict | None = None

    @property
    def source_sha256(self):
        """Hash of exactly the arrays and fields the compaction reads."""
        import numpy as np

        digest = hashlib.sha256()
        for array in (self.states, self.actions):
            array = np.ascontiguousarray(array)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        digest.update(
            json.dumps(
                {
                    "task": self.task,
                    "num_blocks": self.num_blocks,
                    "initial_bindings": self.initial_bindings,
                    "goal_layout": self.goal_layout,
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()

    def block_positions(self, t):
        import numpy as np

        return np.array(
            [self.states[t][10 + 12 * i : 13 + 12 * i] for i in range(self.num_blocks)]
        )


# ---------------------------------------------------------------- loading


def _expected_environment(config):
    from .tasks import environment_options

    return environment_options(config)


def load_demonstrations(config):
    """Validated demonstrations selected by ``config.demonstrations``.

    Returns ``(demonstrations, archive_record)``. Raises DemonstrationError for a
    missing, malformed or incompatible archive or selection.
    """
    import numpy as np

    from synthesis.cfg.recordings import FORMAT, VERSION

    settings = config.demonstrations
    path = Path(settings.archive)
    if path.suffix != ".npz" or not path.is_file():
        raise DemonstrationError(f"Demonstration archive not found: {path}")
    archive_sha256 = sha256_bytes(path.read_bytes())
    if settings.archive_sha256 and settings.archive_sha256 != archive_sha256:
        raise DemonstrationError("Demonstration archive hash differs from the pin")
    expected_environment = _expected_environment(config)
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("format") != FORMAT or meta.get("version") != VERSION:
                raise DemonstrationError("Unsupported demonstration archive format")
            rows = meta["traces"]
            available = [f"d{i}" for i in range(len(rows))]
            selected = (
                list(settings.trace_ids) if settings.trace_ids else list(available)
            )
            missing = [t for t in selected if t not in available]
            if missing:
                raise DemonstrationError(f"Unknown demonstration trace ids: {missing}")
            demonstrations = []
            for trace_id in selected:
                row = rows[int(trace_id[1:])]
                demonstrations.append(
                    _load_trace(
                        data,
                        trace_id,
                        row,
                        config,
                        expected_environment,
                    )
                )
    except DemonstrationError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        raise DemonstrationError(
            f"Malformed demonstration archive: {type(exc).__name__}: {exc}"
        ) from exc
    seeds = [d.seed for d in demonstrations if d.seed is not None]
    if len(set(seeds)) != len(seeds):
        raise DemonstrationError("Selected demonstrations repeat a scene seed")
    return demonstrations, {
        "path": str(path.resolve()),
        "sha256": archive_sha256,
        "format": FORMAT,
        "format_version": VERSION,
        "available_trace_ids": available,
    }


def _load_trace(data, trace_id, row, config, expected_environment):
    import numpy as np

    from synthesis.environment.goal_layout import GoalLayout

    task, count = row["task"], row["num_blocks"]
    metadata = row["metadata"]
    if task != config.task or count != config.num_blocks:
        raise DemonstrationError(
            f"{trace_id} is {task} with {count} blocks; the trial is"
            f" {config.task} with {config.num_blocks}"
        )
    # Completeness: collection accepted the trajectory (normal completion,
    # converged primitives, shared pre/postcondition).
    if metadata.get("status") != "valid":
        raise DemonstrationError(f"{trace_id} is not a validated demonstration")
    tolerance = metadata.get("higher_tolerance")
    if tolerance is None or not math.isclose(
        tolerance, config.higher_tolerance, rel_tol=0, abs_tol=1e-12
    ):
        raise DemonstrationError(
            f"{trace_id} was validated with Higher tolerance {tolerance}, not"
            f" {config.higher_tolerance}"
        )
    environment = metadata.get("environment", {})
    if expected_environment and environment != expected_environment:
        raise DemonstrationError(
            f"{trace_id} layout {environment} differs from {expected_environment}"
        )
    states = np.asarray(data[trace_id + "_states"], dtype=float)
    actions = np.asarray(data[trace_id + "_actions"], dtype=float)
    indices = np.asarray(data[trace_id + "_indices"])
    steps = len(states)
    if (
        states.ndim != 2
        or steps < 2
        or states.shape[1] < 10 + 12 * count
        or not np.isfinite(states).all()
    ):
        raise DemonstrationError(f"{trace_id} has malformed states")
    if (
        actions.shape != (steps - 1, 4)
        or not np.isfinite(actions).all()
        or np.abs(actions).max() > 1 + 1e-9
    ):
        raise DemonstrationError(f"{trace_id} has malformed actions")
    if indices.shape != (steps,) or (indices != np.arange(steps)).any():
        raise DemonstrationError(
            f"{trace_id} is incomplete: every action must be followed by its state"
        )
    bindings = metadata.get("initial_bindings")
    if not isinstance(bindings, dict) or row["bindings"][0] != bindings:
        raise DemonstrationError(f"{trace_id} has inconsistent initial bindings")
    layout = None
    if task in GOAL_TASKS:
        if bindings or "goal_layout" not in row:
            raise DemonstrationError(f"{trace_id} lacks its goal layout")
        layout = GoalLayout.from_dict(row["goal_layout"]).as_dict()
        if len(layout["positions"]) != count:
            raise DemonstrationError("Goal and block counts differ")
        if row["goal_bindings"][0] != {"h0": layout["head"], "null": None}:
            raise DemonstrationError(f"{trace_id} has unexpected goal bindings")
        bindings = {"h0": layout["head"]}
    else:
        expected = {"b0", "tbl"} if task in TABLE_TASKS else {"b0"}
        b0 = bindings.get("b0")
        if (
            set(bindings) != expected
            or type(b0) is not int
            or not 0 <= b0 < count
            or bindings.get("tbl", "tbl") != "tbl"
        ):
            raise DemonstrationError(f"{trace_id} has unexpected input bindings")
        bindings = {"b0": b0}
    return Demonstration(
        trace_id,
        row.get("seed"),
        task,
        count,
        states,
        actions,
        bindings,
        layout,
    )


# ------------------------------------------------------------- compaction


def gripper_commands(actions):
    """Gripper command in force at each state: the latest nonzero command.

    A positive gripper action opens, a negative one closes, and zero keeps the
    current command. Before any command the gripper counts as open. State t
    follows action t - 1, so the command at state t reflects actions[:t].
    """
    commands, current = ["open"], "open"
    for action in actions:
        if action[3] > 0:
            current = "open"
        elif action[3] < 0:
            current = "closed"
        commands.append(current)
    return commands


def event_frames(commands):
    """First and last state, and every state whose command differs from before."""
    last = len(commands) - 1
    return sorted(
        {0, last}
        | {t for t in range(1, len(commands)) if commands[t] != commands[t - 1]}
    )


def _segment_distances(points, start, end):
    import numpy as np

    direction = end - start
    length = float(direction @ direction)
    if length == 0.0:
        return np.linalg.norm(points - start, axis=1)
    fraction = np.clip((points - start) @ direction / length, 0.0, 1.0)
    return np.linalg.norm(points - (start + fraction[:, None] * direction), axis=1)


def rdp(points, tolerance):
    """Indices kept by Ramer-Douglas-Peucker; distances are to the chord segment."""
    import numpy as np

    points = np.asarray(points, dtype=float)
    last = len(points) - 1
    if last < 1:
        return list(range(len(points)))
    keep, stack = {0, last}, [(0, last)]
    while stack:
        first, final = stack.pop()
        if final - first < 2:
            continue
        distances = _segment_distances(
            points[first + 1 : final], points[first], points[final]
        )
        index = int(distances.argmax())
        if distances[index] > tolerance:
            middle = first + 1 + index
            keep.add(middle)
            stack += [(first, middle), (middle, final)]
    return sorted(keep)


def motion_waypoints(path, anchors, tolerance):
    """RDP corners of the gripper path between consecutive anchor (event) frames.

    Event frames are always kept, so each polyline between two of them is
    simplified on its own; the result excludes the anchors themselves.
    """
    corners = set()
    for first, final in zip(anchors, anchors[1:]):
        corners.update(first + i for i in rdp(path[first : final + 1], tolerance))
    return sorted(corners - set(anchors))


def held_block(demonstration, t, command):
    import numpy as np

    if command != "closed":
        return None
    gripper = demonstration.states[t][:3]
    distances = np.linalg.norm(demonstration.block_positions(t) - gripper, axis=1)
    index = int(distances.argmin())
    return index if distances[index] <= HOLD_DISTANCE else None


def block_name(index):
    return f"o{index}"


def goal_name(index):
    return f"g{index}"


def relations(demonstration, t):
    """True DSL atoms of the task API on state t, via the shared scene evaluator.

    Atoms with tbl are omitted: in the shared semantics tbl relates only to
    itself, so no atom between tbl and a block holds. Scattered is symmetric
    and listed once per pair. ON_star_zero uses the first recorded state as the
    frozen entry geometry. Goal tasks add AtGoal and HasBlock (20 mm, 3D).
    """
    import numpy as np

    from synthesis.predicates.scene import evaluate, scene_from_obs
    from synthesis.predicates.term import atom, block_id

    from .contracts import task_api_names

    names = task_api_names(demonstration.task)
    count = demonstration.num_blocks
    scene = scene_from_obs(
        demonstration.states[t],
        count,
        include_table=demonstration.task in TABLE_TASKS,
        entry_obs=demonstration.states[0],
    )
    atoms = []
    for name in PREDICATE_ORDER:
        if name not in names:
            continue
        for a in range(count):
            for b in range(count):
                if a == b or (name == "Scattered" and b < a):
                    continue
                if evaluate(atom(name, block_id(a), block_id(b)), scene):
                    atoms.append(f"{name}({block_name(a)},{block_name(b)})")
    if demonstration.goal_layout is not None:
        from synthesis.environment.goal_layout import occupancy

        occupied = occupancy(
            demonstration.block_positions(t),
            np.asarray(demonstration.goal_layout["positions"]),
        )
        atoms += [f"AtGoal({block_name(i)})" for i in range(count) if occupied[i].any()]
        atoms += [
            f"HasBlock({goal_name(j)})"
            for j in range(occupied.shape[1])
            if occupied[:, j].any()
        ]
    return atoms


def number(value):
    text = f"{float(value):.3f}"
    return "0.000" if text == "-0.000" else text


def point(values):
    return "(" + ",".join(number(v) for v in values) + ")"


@dataclass(frozen=True)
class Frame:
    t: int
    kind: str  # "event" or "waypoint"
    command: str
    gripper: tuple
    holding: int | None
    blocks: tuple | None = None
    relations: tuple | None = None
    added: tuple | None = None
    removed: tuple | None = None


def compact(demonstration, waypoint_tolerance):
    """Event frames and motion waypoints of one demonstration, in time order."""
    commands = gripper_commands(demonstration.actions)
    events = event_frames(commands)
    path = demonstration.states[:, :3]
    waypoints = set(motion_waypoints(path, events, waypoint_tolerance))
    frames, previous = [], None
    for t in sorted(set(events) | waypoints):
        common = dict(
            t=t,
            command=commands[t],
            gripper=tuple(float(v) for v in path[t]),
            holding=held_block(demonstration, t, commands[t]),
        )
        if t in waypoints:
            frames.append(Frame(kind="waypoint", **common))
            continue
        current = relations(demonstration, t)
        frames.append(
            Frame(
                kind="event",
                blocks=tuple(
                    tuple(float(v) for v in p) for p in demonstration.block_positions(t)
                ),
                relations=tuple(current) if previous is None else None,
                added=(
                    None
                    if previous is None
                    else tuple(a for a in current if a not in previous)
                ),
                removed=(
                    None
                    if previous is None
                    else tuple(a for a in previous if a not in current)
                ),
                **common,
            )
        )
        previous = current
    return frames


def render_frame(frame):
    holding = "none" if frame.holding is None else block_name(frame.holding)
    line = (
        f"t={frame.t} {frame.kind} {frame.command} gripper={point(frame.gripper)}"
        f" holding={holding}"
    )
    if frame.kind == "waypoint":
        return line
    lines = [
        line,
        "  blocks: "
        + " ".join(f"{block_name(i)}={point(p)}" for i, p in enumerate(frame.blocks)),
    ]
    if frame.relations is not None:
        lines.append("  holds: " + (" ".join(frame.relations) or "none"))
    else:
        changes = [f"+{a}" for a in frame.added] + [f"-{a}" for a in frame.removed]
        lines.append("  changes: " + (" ".join(changes) or "none"))
    return "\n".join(lines)


def render_inputs(demonstration):
    """The inputs line and, for goal tasks, the goal layout line."""
    lines = []
    if demonstration.goal_layout is not None:
        layout = demonstration.goal_layout
        lines.append(f"inputs: h0={goal_name(layout['head'])}")
        goals = []
        for j, position_ in enumerate(layout["positions"]):
            right = layout["r"][j]
            down = layout["d"][j]
            goals.append(
                f"{goal_name(j)}={point(position_)}"
                f" right={'null' if right is None else goal_name(right)}"
                f" down={'null' if down is None else goal_name(down)}"
            )
        lines.append("goals: " + "; ".join(goals))
    else:
        lines.append(f"inputs: b0={block_name(demonstration.initial_bindings['b0'])}")
    return lines


def summary_line(demonstration, frames):
    """Task, blocks, recorded states and frame counts of one compacted trace."""
    count = demonstration.num_blocks
    steps = len(demonstration.states)
    return (
        f"{demonstration.task}, {count} blocks"
        f" {block_name(0)}..{block_name(count - 1)}, {steps} states"
        f" (t=0..{steps - 1}), {sum(f.kind == 'event' for f in frames)} events,"
        f" {sum(f.kind == 'waypoint' for f in frames)} waypoints"
    )


def render_demonstration(demonstration, frames, position, total):
    lines = [
        f"DEMONSTRATION {position}/{total}: " + summary_line(demonstration, frames)
    ]
    lines += render_inputs(demonstration)
    lines += [render_frame(frame) for frame in frames]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- context


@dataclass
class DemonstrationContext:
    """One rendered demonstration block, shared by every request of a trial."""

    text: str
    tokens: int
    manifest: dict
    initial_positions: list = field(repr=False, default_factory=list)

    @property
    def sha256(self):
        return sha256_text(self.text)

    @property
    def seeds(self):
        return [d["seed"] for d in self.manifest["demonstrations"]]

    def check_budget(self, max_tokens):
        if self.tokens > max_tokens:
            raise DemonstrationBudgetExceeded(
                f"Demonstration block has {self.tokens} tokens; the limit"
                f" demonstrations.max_tokens is {max_tokens}"
            )


def render_context(demonstrations, config, archive):
    """Rendered block and manifest; numbers and order are deterministic."""
    from synthesis.util.on import using_higher_tolerance

    from .prompt_package import package, prompt_provenance

    settings = config.demonstrations
    templates = package(config.prompt_version).get("demonstrations")
    if templates is None:
        raise DemonstrationError(
            f"Prompt package {config.prompt_version} has no demonstration templates"
        )
    if templates.get("serialization_version") != SERIALIZATION_VERSION:
        raise DemonstrationError("Prompt package serialization version differs")
    total = len(demonstrations)
    parts, records = [], []
    with using_higher_tolerance(config.higher_tolerance):
        for position, demonstration in enumerate(demonstrations, 1):
            frames = compact(demonstration, settings.waypoint_tolerance)
            text = render_demonstration(demonstration, frames, position, total)
            parts.append(text)
            records.append(
                {
                    "position": position,
                    "trace_id": demonstration.trace_id,
                    "seed": demonstration.seed,
                    "states": len(demonstration.states),
                    "events": sum(f.kind == "event" for f in frames),
                    "waypoints": sum(f.kind == "waypoint" for f in frames),
                    "frame_steps": [f.t for f in frames],
                    "source_sha256": demonstration.source_sha256,
                    "text_sha256": sha256_text(text),
                }
            )
    header = templates["header"].format(
        version=SERIALIZATION_VERSION,
        count=total,
        task=config.task,
        waypoint_tolerance=number(settings.waypoint_tolerance),
        hold_distance=number(HOLD_DISTANCE),
        higher_tolerance=repr(float(config.higher_tolerance)),
    )
    text = header + "\n".join(parts) + templates["footer"]
    manifest = {
        "serialization_version": SERIALIZATION_VERSION,
        "settings": asdict(settings),
        "archive": archive,
        "ordering": "configured trace_ids order (archive order when unset)",
        "trace_ids": [d.trace_id for d in demonstrations],
        "demonstrations": records,
        "higher_tolerance": config.higher_tolerance,
        **prompt_provenance(config.prompt_version),
        "context_sha256": sha256_text(text),
        "context_bytes": len(text.encode("utf-8")),
        "fields_read": [
            "states",
            "actions",
            "indices (completeness check only)",
            "task",
            "num_blocks",
            "seed (scene disjointness only)",
            "metadata.status, metadata.higher_tolerance, metadata.environment"
            " (validation only)",
            "metadata.initial_bindings / goal_layout / goal_bindings[0]",
        ],
    }
    return text, manifest


def count_tokens(text, settings, model_config, opener=None, timeout=60):
    """Token count of the rendered block and the method used to measure it."""
    measured = settings.measured_tokens
    if measured is not None:
        # Revision 7.5: measured once for the batch; valid only for this block.
        if measured["sha256"] != sha256_text(text):
            raise DemonstrationError(
                "demonstrations.measured_tokens was measured for a different block"
            )
        return measured["tokens"], dict(measured.get("measurement") or {}) | {
            "measured_once": True
        }
    if settings.token_counter == "utf8_bytes":
        return len(text.encode("utf-8")), {"method": "utf8_bytes_upper_bound"}
    if settings.token_counter == "provider_usage":
        return provider_usage_tokens(text, model_config, opener, max(timeout, 300))
    import os

    url = settings.tokenizer_url
    if url is None:
        base = model_config.base_url.rstrip("/")
        url = (base[: -len("/v1")] if base.endswith("/v1") else base) + "/tokenize"
    key = (
        os.environ.get(model_config.api_key_env, "") if model_config.api_key_env else ""
    )
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {"model": model_config.model, "prompt": text, "add_special_tokens": False}
        ).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as handle:
            reply = json.load(handle)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise DemonstrationError(
            f"Demonstration token count unavailable: {type(exc).__name__}"
        ) from exc
    count = reply.get("count")
    if type(count) is not int or count <= 0:
        raise DemonstrationError("Tokenizer returned no token count")
    return count, {
        "method": "server_tokenize",
        "url": url,
        "model": model_config.model,
        "add_special_tokens": False,
        "server_max_model_len": reply.get("max_model_len"),
    }


PROVIDER_USAGE_SUFFIX = "Reply with OK."


def provider_usage_tokens(text, model_config, opener=None, timeout=300):
    """Block tokens as the provider counts them, for models without a local
    tokenizer: prompt tokens of ``block + suffix`` minus those of ``suffix``
    alone, each from a one-token request. The chat template's overhead cancels.
    """
    from .llm import HTTPModel, ModelError

    model = HTTPModel(model_config, opener)
    counts = []
    for prompt in (text + PROVIDER_USAGE_SUFFIX, PROVIDER_USAGE_SUFFIX):
        body = {
            "model": model_config.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            model_config.token_field: 16,
        }
        try:
            reply = model.post("chat/completions", body, timeout)
        except ModelError as exc:
            raise DemonstrationError(
                f"Demonstration token count unavailable: {exc}"
            ) from exc
        count = (reply.get("usage") or {}).get("prompt_tokens")
        if type(count) is not int or count <= 0:
            raise DemonstrationError("Provider reported no prompt token count")
        counts.append(count)
    if counts[0] <= counts[1]:
        raise DemonstrationError("Provider prompt counts do not include the block")
    return counts[0] - counts[1], {
        "method": "provider_usage",
        "model": model_config.model,
        "base_url": model_config.base_url,
        "prompt_tokens_with_block": counts[0],
        "prompt_tokens_without_block": counts[1],
        "suffix": PROVIDER_USAGE_SUFFIX,
        "provider": reply.get("provider"),
    }


def measure_once(config, model_config):
    """``demonstrations.measured_tokens`` for ``config``: one measurement of its
    rendered block, reused by every trial of a batch (Revision 7.5)."""
    from dataclasses import replace

    settings = replace(config.demonstrations, measured_tokens=None)
    demonstrations, archive = load_demonstrations(config)
    text, _ = render_context(demonstrations, config, archive)
    tokens, measurement = count_tokens(text, settings, model_config)
    return {"sha256": sha256_text(text), "tokens": tokens, "measurement": measurement}


def prepare(config, model_config, *, token_counter=None):
    """Load, compact, render and measure the configured demonstrations once."""
    demonstrations, archive = load_demonstrations(config)
    text, manifest = render_context(demonstrations, config, archive)
    if token_counter is not None:
        tokens, measurement = token_counter(text)
    else:
        tokens, measurement = count_tokens(text, config.demonstrations, model_config)
    manifest |= {"tokens": tokens, "token_measurement": measurement}
    return DemonstrationContext(
        text,
        tokens,
        manifest,
        [d.block_positions(0) for d in demonstrations],
    )


# --------------------------------------------------------- scene overlap


def check_evaluation_seed(context, task, seed, initial_state=None, trace_index=0):
    """Reject an evaluation scene generated from a demonstration's seed or trace."""
    archive = context.manifest["archive"]
    if initial_state is not None:
        same_file = Path(initial_state).resolve() == Path(archive["path"])
        if same_file and f"d{trace_index}" in context.manifest["trace_ids"]:
            raise DemonstrationError(
                "The evaluation snapshot is one of the selected demonstrations"
            )
    if seed is not None and seed in context.seeds:
        raise DemonstrationError(
            f"Evaluation seed {seed} generated a selected {task} demonstration"
        )


def check_evaluation_scene(context, entry_observation, num_blocks):
    """Reject an evaluation start whose block centres match a demonstration's."""
    import numpy as np

    positions = np.array(
        [entry_observation[10 + 12 * i : 13 + 12 * i] for i in range(num_blocks)]
    )
    for record, initial in zip(
        context.manifest["demonstrations"], context.initial_positions
    ):
        if np.abs(positions - initial).max() < SAME_SCENE_DISTANCE:
            raise DemonstrationError(
                f"The evaluation scene matches demonstration {record['trace_id']}"
            )
