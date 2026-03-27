from itertools import product
from typing import List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from z3 import (
    Abs,
    And,
    BoolVal,
    Consts,
    DeclareSort,
    Function,
    If,
    Or,
    Real,
    Reals,
    RealSort,
    RealVal,
    Solver,
    Z3_OP_EQ,
    Z3_OP_UNINTERPRETED,
    get_var_index,
    is_app,
    is_quantifier,
    is_var,
    sat,
    unsat,
)


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

class LowLevelContext:
    def __init__(self, sort_name: str = "Box"):
        self.sort_name = sort_name
        self._build_symbols()

    def _build_symbols(self):
        self.BoxSort = DeclareSort(self.sort_name)
        self.X = Function("X", self.BoxSort, RealSort())
        self.Y = Function("Y", self.BoxSort, RealSort())
        self.Z = Function("Z", self.BoxSort, RealSort())
        (self.L,) = Reals("L")

    def lowlevel_box_equal(self, b1, b2):
        return And(self.X(b1) == self.X(b2), self.Y(b1) == self.Y(b2), self.Z(b1) == self.Z(b2))

    def lowlevel_on(self, b1, b2):
        return Or(
            And(Abs(self.X(b1) - self.X(b2)) < self.L / 2, Abs(self.Y(b1) - self.Y(b2)) < self.L / 2, self.Z(b1) >= self.Z(b2)),
        )

    def lowlevel_higher(self, b1, b2):
        # b1 is higher than b2
        return self.Z(b1) >= self.Z(b2)

    def lowlevel_scattered(self, t1, t2):
        return Or(Abs(self.X(t1) - self.X(t2)) > 2 * self.L, Abs(self.Y(t1) - self.Y(t2)) > 2 * self.L)

    def get_consts(self, symbol: str):
        (c,) = Consts(symbol, self.BoxSort)
        return c

    def encode_collision(self, a, p0, p1):
        """Return a Z3 formula asserting block *a* collides with the swept
        volume of a cube moving linearly from *p0* to *p1*.

        p0, p1 are (x, y, z) triples of Z3 Real expressions.
        """
        (x0, y0, z0) = p0
        (x1, y1, z1) = p1
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

    def check_solver(self, s, blocks, save_path=None):
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
            self.visualize_scene(m, blocks, save_path=save_path)
        return result

    def visualize_scene(self, model, blocks, save_path=None):
        """Plot each block as a 3-D cube using coordinates from the Z3 model."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        colors = ["red", "blue", "yellow", "green", "gray",
                  "cyan", "magenta", "orange"]

        def to_float(z3val):
            s = str(model.eval(z3val).as_decimal(20))
            return float(s) if s[-1] != "?" else float(s[:-1])

        L_val = to_float(self.L)

        for block, color in zip(blocks, colors[: len(blocks)]):
            cx = to_float(self.X(block))
            cy = to_float(self.Y(block))
            cz = to_float(self.Z(block))
            draw_cube(ax, cx, cy, cz, L_val, color=color)
            print(str(block), cx, cy, cz)
            ax.text(cx, cy, cz, str(block), fontsize=12)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)

        if save_path:
            plt.savefig(save_path)
            print(f"Scene saved to {save_path}")
        else:
            plt.show()

        return ax

    def start_verification(
        self, initial_condition: List, pickplace_instructions: List, constants: List
    ) -> None:
        s = Solver()
        s.set(unsat_core=True)
        const_map = self.translate_condition(s, constants, initial_condition)
        blocks = list(const_map.values())
        sym = self.get_consts("sym")

        print("checking condition satisfiability")
        self.check_solver(s, blocks)

        grab_const = None
        handled_block_id = None
        current_pos = None

        for idx, instruction in enumerate(pickplace_instructions):
            print(f"\n=== verifying pickplace instruction {idx}: {instruction} ===")

            target_key = str(constants[instruction.target_box_id])
            ll_target = const_map[target_key]

            if idx == 0:
                handled_block_id = instruction.grab_box_id
                grab_key = str(constants[instruction.grab_box_id])
                grab_const = const_map[grab_key]
                current_pos = (
                    self.X(grab_const),
                    self.Y(grab_const),
                    self.Z(grab_const),
                )
            else:
                assert instruction.grab_box_id == handled_block_id, (
                    f"Expected same block {handled_block_id}, "
                    f"got {instruction.grab_box_id}"
                )

            end_pos = (
                self.X(ll_target) + RealVal(instruction.target_offset[0].val),
                self.Y(ll_target) + RealVal(instruction.target_offset[1].val),
                self.Z(ll_target) + RealVal(instruction.target_offset[2].val),
            )

            print(f"  checking tube_{idx}")
            s.push()
            s.assert_and_track(
                self.encode_collision(sym, current_pos, end_pos),
                f"tube_{idx}",
            )
            s.assert_and_track(sym != grab_const, f"sym_neq_grab_{idx}")
            self.check_solver(s, blocks)
            s.pop()

            current_pos = end_pos

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
                    self._translate_expr(body, lowlevel_constants, const_map, new_bindings)
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
            s.add(translated)

        return const_map


class lowlevel_z3_solver:
    def __init__(
        self, sort_name: str = "Box", context: Optional[LowLevelContext] = None
    ):
        self.context = (
            context if context is not None else LowLevelContext(sort_name=sort_name)
        )

    def start_verification(self, initial_condition, pickplace_instructions) -> None:
        return self.context.start_verification(
            initial_condition, pickplace_instructions
        )
