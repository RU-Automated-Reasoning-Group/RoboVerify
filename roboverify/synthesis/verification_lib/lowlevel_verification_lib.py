from itertools import product
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from z3 import (
    Z3_OP_DISTINCT,
    Z3_OP_EQ,
    Z3_OP_UNINTERPRETED,
    Abs,
    And,
    BoolVal,
    Consts,
    DeclareSort,
    Function,
    If,
    Not,
    Or,
    Real,
    Reals,
    RealSort,
    RealVal,
    Solver,
    get_var_index,
    is_app,
    is_quantifier,
    is_true,
    is_var,
    sat,
    unsat,
)

import synthesis.api.instructions as instructions


def abs_diff(u, v):
    return If(u - v >= 0, u - v, v - u)


def draw_cube(ax, cx, cy, cz, L, color="blue"):
    """Draw a cube centered at (cx, cy, cz) with side length L."""
    r = L / 2.0
    v = [
        (cx - r, cy - r, cz - r),
        (cx + r, cy - r, cz - r),
        (cx + r, cy + r, cz - r),
        (cx - r, cy + r, cz - r),
        (cx - r, cy - r, cz + r),
        (cx + r, cy - r, cz + r),
        (cx + r, cy + r, cz + r),
        (cx - r, cy + r, cz + r),
    ]
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.3))


def _box_corners(center, halfwidth):
    cx, cy, cz = center
    h = float(halfwidth)
    return np.array(
        [
            [cx - h, cy - h, cz - h],
            [cx - h, cy - h, cz + h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy + h, cz + h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz - h],
            [cx + h, cy + h, cz + h],
        ],
        dtype=float,
    )


def draw_encoded_tube(ax, p0, p1, halfwidth, color="purple", alpha=0.12):
    """Visualize the swept AABB 'tube' used by `encode_collision` as ONE object.

    The swept volume is the convex hull of the two endpoint AABBs centered at p0/p1
    with halfwidth `halfwidth` (since the constraint is Abs(coord - c(t)) < halfwidth).
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    # Draw the centerline.
    ax.plot([x0, x1], [y0, y1], [z0, z1], color=color, linewidth=2, alpha=0.6)

    pts = np.vstack([_box_corners(p0, halfwidth), _box_corners(p1, halfwidth)])
    hull = ConvexHull(pts)

    faces = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        faces.append(tri.tolist())

    poly = Poly3DCollection(faces, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(poly)


class LowLevelContext:
    def __init__(self, sort_name: str = "Box", default_L: float = 1.0):
        self.sort_name = sort_name
        self.default_L = default_L
        self._build_symbols()

    def _build_symbols(self):
        self.BoxSort = DeclareSort(self.sort_name)
        self.X = Function("X", self.BoxSort, RealSort())
        self.Y = Function("Y", self.BoxSort, RealSort())
        self.Z = Function("Z", self.BoxSort, RealSort())
        (self.L,) = Reals("L")

    def lowlevel_box_equal(self, b1, b2):
        return And(
            self.X(b1) == self.X(b2), self.Y(b1) == self.Y(b2), self.Z(b1) == self.Z(b2)
        )

    def lowlevel_on(self, b1, b2):
        return Or(
            And(
                Abs(self.X(b1) - self.X(b2)) < self.L / 2,
                Abs(self.Y(b1) - self.Y(b2)) < self.L / 2,
                self.Z(b1) >= self.Z(b2),
            ),
        )

    def lowlevel_higher(self, b1, b2):
        # b1 is higher than b2
        return self.Z(b1) >= self.Z(b2)

    def lowlevel_scattered(self, t1, t2):
        return Or(
            Abs(self.X(t1) - self.X(t2)) > 2 * self.L,
            Abs(self.Y(t1) - self.Y(t2)) > 2 * self.L,
        )

    def get_consts(self, symbol: str):
        (c,) = Consts(symbol, self.BoxSort)
        return c

    def encode_collision(self, a, p0, p1):
        """Return a Z3 formula asserting block *a* collides with the swept
        volume of a cube moving linearly from *p0* to *p1*.

        p0, p1 are (x, y, z) triples of Z3 Real expressions.
        """
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        t = Real("t")
        cx = (1 - t) * x0 + t * x1
        cy = (1 - t) * y0 + t * y1
        cz = (1 - t) * z0 + t * z1
        return And(
            t >= 0,
            t <= 1,
            Abs(self.X(a) - cx) < self.L,
            Abs(self.Y(a) - cy) < self.L,
            Abs(self.Z(a) - cz) < self.L,
        )

    def check_solver(
        self, s, blocks, save_path=None, encoded_tube=None, extra_blocks=None
    ):
        """Check satisfiability; visualize the scene if SAT, print core if UNSAT."""
        result = s.check()
        print(f"Satisfiability: {result}")
        if result == unsat:
            print("UNSAT - constraints are unsatisfiable")
            core = s.unsat_core()
            if core:
                print("Unsat core:", core)
        elif result == sat:
            m = s.model()
            print(m)
            self.visualize_scene(
                m,
                blocks,
                save_path=save_path,
                encoded_tube=encoded_tube,
                extra_blocks=extra_blocks,
            )
        return result

    def visualize_scene(
        self, model, blocks, save_path=None, encoded_tube=None, extra_blocks=None
    ):
        """Plot each block as a 3-D cube using coordinates from the Z3 model."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        colors = ["red", "blue", "yellow", "green", "gray", "cyan", "magenta", "orange"]

        def to_float(z3val):
            s = str(model.eval(z3val).as_decimal(20))
            return float(s) if s[-1] != "?" else float(s[:-1])

        L_val = to_float(self.L)
        half_side = L_val / 2.0

        # Track bounds so we can enforce equal aspect ratio.
        min_x = float("inf")
        min_y = float("inf")
        min_z = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        max_z = float("-inf")

        def include_aabb(cx, cy, cz, hx, hy, hz):
            nonlocal min_x, min_y, min_z, max_x, max_y, max_z
            min_x = min(min_x, cx - hx)
            max_x = max(max_x, cx + hx)
            min_y = min(min_y, cy - hy)
            max_y = max(max_y, cy + hy)
            min_z = min(min_z, cz - hz)
            max_z = max(max_z, cz + hz)

        def print_relation_tables(objects, labels):
            def eval_bool(expr):
                return bool(is_true(model.eval(expr, model_completion=True)))

            def print_table(title, rel_fn):
                width = max(5, max(len(s) for s in labels) + 1)
                print("\n" + title)
                print("".ljust(width) + "".join(s.ljust(width) for s in labels))
                for i, (oi, li) in enumerate(zip(objects, labels)):
                    row = [li.ljust(width)]
                    for j, oj in enumerate(objects):
                        cell = "T" if eval_bool(rel_fn(oi, oj)) else "F"
                        row.append(cell.ljust(width))
                    print("".join(row))

            # `ON_star`/`ON_star_zero` are translated to `lowlevel_on`.
            print_table(
                "Relation: ON_star (lowlevel_on)  [row ON col]", self.lowlevel_on
            )
            print_table(
                "Relation: Higher (lowlevel_higher)  [row >= col]", self.lowlevel_higher
            )
            print_table(
                "Relation: Scattered (lowlevel_scattered)  [row scattered-from col]",
                self.lowlevel_scattered,
            )

        for block, color in zip(blocks, colors[: len(blocks)]):
            cx = to_float(self.X(block))
            cy = to_float(self.Y(block))
            cz = to_float(self.Z(block))
            draw_cube(ax, cx, cy, cz, L_val, color=color)
            include_aabb(cx, cy, cz, half_side, half_side, half_side)
            print(str(block), cx, cy, cz)
            ax.text(cx, cy, cz, str(block), fontsize=12)

        # Print relationships among all objects we are visualizing.
        rel_objects = list(blocks)
        rel_labels = [str(b) for b in blocks]

        if extra_blocks:
            for item in extra_blocks:
                if len(item) == 3:
                    block, color, label = item
                elif len(item) == 2:
                    block, label = item
                    color = "black"
                else:
                    block = item[0]
                    color = "black"
                    label = str(block)
                cx = to_float(self.X(block))
                cy = to_float(self.Y(block))
                cz = to_float(self.Z(block))
                draw_cube(ax, cx, cy, cz, L_val, color=color)
                include_aabb(cx, cy, cz, half_side, half_side, half_side)
                print(str(label), cx, cy, cz)
                ax.text(cx, cy, cz, str(label), fontsize=12, color=color)
                rel_objects.append(block)
                rel_labels.append(str(label))

        if rel_objects:
            print_relation_tables(rel_objects, rel_labels)

        if encoded_tube is not None:
            p0_expr, p1_expr, label = encoded_tube
            p0 = tuple(to_float(e) for e in p0_expr)
            p1 = tuple(to_float(e) for e in p1_expr)
            # In encode_collision we use Abs(coord - c(t)) < L, so L is halfwidth.
            draw_encoded_tube(ax, p0, p1, halfwidth=L_val)
            # Tube halfwidth in each dimension is L_val (not L_val/2).
            include_aabb(p0[0], p0[1], p0[2], L_val, L_val, L_val)
            include_aabb(p1[0], p1[1], p1[2], L_val, L_val, L_val)
            mx = 0.5 * (p0[0] + p1[0])
            my = 0.5 * (p0[1] + p1[1])
            mz = 0.5 * (p0[2] + p1[2])
            ax.text(mx, my, mz, str(label), fontsize=12, color="purple")

        # Enforce equal scaling so cubes render as cubes even if X/Y/Z ranges differ.
        if min_x != float("inf"):
            cx = 0.5 * (min_x + max_x)
            cy = 0.5 * (min_y + max_y)
            cz = 0.5 * (min_z + max_z)
            max_range = max(max_x - min_x, max_y - min_y, max_z - min_z)
            if max_range == 0:
                max_range = L_val
            half = 0.5 * max_range
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_zlim(cz - half, cz + half)

        # Newer matplotlib supports this and improves cube appearance further.
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if save_path:
            plt.savefig(save_path)
            print(f"Scene saved to {save_path}")
        else:
            plt.show()

        return ax

    def start_verification(
        self, initial_condition: List, pickplace_instructions: List, constants: List
    ) -> bool:
        s = Solver()
        s.set(unsat_core=True)
        # Default block side length.
        s.add(self.L == RealVal(str(self.default_L)))
        const_map = self.translate_condition(s, constants, initial_condition)
        blocks = list(const_map.values())
        sym = self.get_consts("sym")

        print("checking condition satisfiability")
        ok = True
        cond_result = self.check_solver(s, blocks, extra_blocks=[(sym, "black", "sym")])
        if cond_result != sat:
            ok = False
            print(
                f"[FAIL] low-level axioms/initial-condition consistency returned {cond_result}; expected sat"
            )

        grab_const = None
        handled_block_id = None
        current_pos = None

        for idx, instruction in enumerate(pickplace_instructions):
            if not isinstance(instruction, instructions.PickPlaceByName):
                print(f"instruction {idx} is not an PickPlaceByName")
                continue
            print(f"\n=== verifying pickplace instruction {idx}: {instruction} ===")

            tx, ty, tz = instruction.target_box_names
            ll_target_x = const_map[str(tx)]
            ll_target_y = const_map[str(ty)]
            ll_target_z = const_map[str(tz)]

            if idx == 0:
                handled_block_id = instruction.grab_box_name
                grab_key = instruction.grab_box_name
                grab_const = const_map[str(grab_key)]
                current_pos = (
                    self.X(grab_const),
                    self.Y(grab_const),
                    self.Z(grab_const),
                )
            else:
                assert instruction.grab_box_name == handled_block_id, (
                    f"Expected same block {handled_block_id}, "
                    f"got {instruction.grab_box_name}"
                )

            end_pos = (
                self.X(ll_target_x) + RealVal(instruction.target_offset[0].val),
                self.Y(ll_target_y) + RealVal(instruction.target_offset[1].val),
                self.Z(ll_target_z) + RealVal(instruction.target_offset[2].val),
            )

            print(f"  checking tube_{idx}")
            s.push()
            s.assert_and_track(
                self.encode_collision(sym, current_pos, end_pos),
                f"tube_{idx}",
            )
            s.assert_and_track(
                Not(self.lowlevel_box_equal(sym, grab_const)),
                f"sym_neq_grab_{idx}",
            )
            tube_result = self.check_solver(
                s,
                blocks,
                encoded_tube=(current_pos, end_pos, f"tube_{idx}"),
                extra_blocks=[(sym, "black", "sym")],
            )
            if tube_result != unsat:
                ok = False
                print(
                    f"[FAIL] tube_{idx} satisfiability returned {tube_result}; expected unsat (tube should be false)"
                )
            s.pop()

            current_pos = end_pos
        return ok

    def _translate_expr(self, expr, lowlevel_constants, const_map, bindings):
        """Recursively translate a high-level z3 expression to low-level.

        ForAll is eliminated by enumerating all combinations of lowlevel_constants.
        ON_star / ON_star_zero -> lowlevel_on, Higher -> lowlevel_higher,
        Scattered -> lowlevel_scattered. Boolean structure is preserved.

        bindings: list where bindings[de_bruijn_index] = concrete lowlevel constant.
        """
        if is_var(expr):
            return bindings[get_var_index(expr)]

        if is_quantifier(expr) and expr.is_forall():
            num_vars = expr.num_vars()
            body = expr.body()
            conjuncts = []
            for assignment in product(lowlevel_constants, repeat=num_vars):
                # de Bruijn: var_name(i) has index (num_vars - 1 - i) in body
                new_prefix = [None] * num_vars
                for i in range(num_vars):
                    new_prefix[num_vars - 1 - i] = assignment[i]
                new_bindings = new_prefix + bindings
                conjuncts.append(
                    self._translate_expr(
                        body, lowlevel_constants, const_map, new_bindings
                    )
                )
            return And(*conjuncts)

        if is_app(expr):
            decl = expr.decl()

            if decl.kind() == Z3_OP_UNINTERPRETED:
                name = decl.name()
                if decl.arity() == 0:
                    if name in const_map:
                        return const_map[name]
                    raise ValueError(f"Unknown constant in translation: {name}")

                children = [
                    self._translate_expr(c, lowlevel_constants, const_map, bindings)
                    for c in expr.children()
                ]
                if name in ("ON_star", "ON_star_zero"):
                    return self.lowlevel_on(children[0], children[1])
                if name == "Higher":
                    return self.lowlevel_higher(children[0], children[1])
                if name == "Scattered":
                    return self.lowlevel_scattered(children[0], children[1])
                raise NotImplementedError(f"Unhandled high-level predicate: {name}")

            if decl.kind() == Z3_OP_EQ:
                children = [
                    self._translate_expr(c, lowlevel_constants, const_map, bindings)
                    for c in expr.children()
                ]
                return self.lowlevel_box_equal(children[0], children[1])

            if decl.kind() == Z3_OP_DISTINCT:
                children = [
                    self._translate_expr(c, lowlevel_constants, const_map, bindings)
                    for c in expr.children()
                ]
                # Mirror the `==` translation: prefer "distinct position" for box terms.
                # If the children are not box terms, constructing lowlevel_box_equal will
                # raise a sort/type error; in that case, preserve the original Distinct.
                try:
                    if len(children) == 2:
                        return Not(self.lowlevel_box_equal(children[0], children[1]))
                    pairwise = []
                    for i in range(len(children)):
                        for j in range(i + 1, len(children)):
                            pairwise.append(
                                Not(self.lowlevel_box_equal(children[i], children[j]))
                            )
                    return And(*pairwise) if pairwise else BoolVal(True)
                except Exception:
                    return decl(*children)

            children = [
                self._translate_expr(c, lowlevel_constants, const_map, bindings)
                for c in expr.children()
            ]
            return decl(*children)

        return expr

    def translate_condition(self, s: Solver, constants: List, conditions: List):
        const_map = {str(c): self.get_consts(str(c)) for c in constants}
        lowlevel_constants = list(const_map.values())
        sym = self.get_consts("sym")
        lowlevel_constants.append(sym)

        for idx, condition in enumerate(conditions):
            print(f"translating condition {idx}, {condition}")
            translated = self._translate_expr(
                condition, lowlevel_constants, const_map, []
            )
            s.assert_and_track(translated, f"condition_{idx}")

        return const_map
