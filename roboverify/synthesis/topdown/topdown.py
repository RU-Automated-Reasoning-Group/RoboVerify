# from dsl import Exists, Program
from collections import deque
import time

import copy
import string
from typing import List
from collections import deque
import time

MAX_QUANTIFIERS = 2


BLOCK_LENGTH = (
    0.025 * 2
)  # (0.025, 0.025, 0.025) in the xml file of the gym env is half length


def on_star_eval(block1, block2):
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def fresh_var(ctx):
    """Generate a new variable name based on current ctx length"""
    return string.ascii_lowercase[len(ctx)]


# --------------------------------------------------
# Attribute accessors
# --------------------------------------------------


class z:
    def __call__(self, obj):
        return obj.z

    def __str__(self):
        return "z"


# --------------------------------------------------
# Environment objects
# --------------------------------------------------


class EnvObj:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.id


class Box(EnvObj):
    """Box object with x,y,z"""

    def __init__(self, id):
        super().__init__(id)
        self.set = False

    def set_attribute(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.set = True


# --------------------------------------------------
# Base Program
# --------------------------------------------------


class Program:
    def __init__(self, ctx: List[EnvObj]):
        self.ctx = list(ctx)

    def is_complete(self):
        return False

    def evaluate(self, input, mapping):
        raise NotImplementedError

    def expand(self):
        programs = [
            Not(self.ctx, Program(self.ctx)),
            And(self.ctx, Program(self.ctx), Program(self.ctx)),
            Or(self.ctx, Program(self.ctx), Program(self.ctx)),
        ]
        programs.extend(Predicate(self.ctx).expand())

        if self.quantifier_count() < MAX_QUANTIFIERS:
            programs.append(Exists(self.ctx, Program(self.ctx)))
            programs.append(ForAll(self.ctx, Program(self.ctx)))

        return programs

    def add_object(self, obj):
        if hasattr(self, "ctx"):
            self.ctx.append(obj)

    def __str__(self):
        return "P"

    def quantifier_count(self):
        return 0


# --------------------------------------------------
# Quantifiers
# --------------------------------------------------


class Exists(Program):
    def __init__(self, ctx, program):
        self.ctx = ctx
        self.program = program
        self.object = Box(fresh_var(ctx))
        if hasattr(program, "ctx") and len(ctx) == len(program.ctx):
            self.program = self._extend_program_ctx(ctx, program, self.object)

    def _extend_program_ctx(self, ctx, program, obj):
        if hasattr(program, "ctx") and len(ctx) == len(program.ctx):
            new_program = program
            new_program.ctx = program.ctx + [obj]
            return new_program
        return program

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        for obj in input["all_box"]:
            if self.program.evaluate(input, {**mapping, self.object.id: obj}):
                return True
        return False

    def evaluate_specific(self, input):
        return self.program.evaluate(input, {self.object.id: input["target"]})
        # return self.evaluate(input, {self.object.id: input["target"]})

    def expand(self):
        children = [Exists(self.ctx, p) for p in self.program.expand()]
        return children

    def __str__(self):
        return f"(∃{self.object}.({self.program}))"

    def quantifier_count(self):
        return 1 + self.program.quantifier_count()


class ForAll(Program):
    def __init__(self, ctx, program):
        self.ctx = ctx
        self.program = program
        self.object = Box(fresh_var(ctx))
        if hasattr(program, "ctx") and len(ctx) == len(program.ctx):
            self.program = self._extend_program_ctx(ctx, program, self.object)

    def _extend_program_ctx(self, ctx, program, obj):
        if hasattr(program, "ctx") and len(ctx) == len(program.ctx):
            new_program = program
            new_program.ctx = program.ctx + [obj]
            return new_program
        return program

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        for obj in input["all_box"]:
            if not self.program.evaluate(input, {**mapping, self.object.id: obj}):
                return False
        return True

    def expand(self):
        return [ForAll(self.ctx, p) for p in self.program.expand()]

    def __str__(self):
        return f"(∀{self.object}.({self.program}))"

    def quantifier_count(self):
        return 1 + self.program.quantifier_count()


# --------------------------------------------------
# Logical operators
# --------------------------------------------------


class Not(Program):
    def __init__(self, ctx, program):
        self.ctx = ctx
        self.program = program

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        return not self.program.evaluate(input, mapping)

    def expand(self):
        return [Not(self.ctx, p) for p in self.program.expand()]

    def __str__(self):
        return f"¬({self.program})"

    def quantifier_count(self):
        return self.program.quantifier_count()


class And(Program):
    def __init__(self, ctx, p1, p2):
        self.ctx = ctx
        self.program1 = p1
        self.program2 = p2

    def is_complete(self):
        return self.program1.is_complete() and self.program2.is_complete()

    def evaluate(self, input, mapping):
        return self.program1.evaluate(input, mapping) and self.program2.evaluate(
            input, mapping
        )

    def expand(self):
        if not self.program1.is_complete():
            return [And(self.ctx, p, self.program2) for p in self.program1.expand()]
        return [And(self.ctx, self.program1, p) for p in self.program2.expand()]

    def __str__(self):
        return f"({self.program1} ∧ {self.program2})"

    def quantifier_count(self):
        return self.program.quantifier_count()


class Or(Program):
    def __init__(self, ctx, p1, p2):
        self.ctx = ctx
        self.program1 = p1
        self.program2 = p2

    def is_complete(self):
        return self.program1.is_complete() and self.program2.is_complete()

    def evaluate(self, input, mapping):
        return self.program1.evaluate(input, mapping) or self.program2.evaluate(
            input, mapping
        )

    def expand(self):
        if not self.program1.is_complete():
            return [Or(self.ctx, p, self.program2) for p in self.program1.expand()]
        return [Or(self.ctx, self.program1, p) for p in self.program2.expand()]

    def __str__(self):
        return f"({self.program1} ∨ {self.program2})"

    def quantifier_count(self):
        return self.program.quantifier_count()


# --------------------------------------------------
# Predicate base
# --------------------------------------------------


class Predicate(Program):
    def __init__(self, ctx):
        self.ctx = ctx

    def is_complete(self):
        return False

    def expand(self):
        programs = []
        # Can reference all objects in ctx (variables + constants)
        for i in range(len(self.ctx)):
            for j in range(len(self.ctx)):
                if i != j:
                    programs.append(ON_star(self.ctx[i], self.ctx[j]))
                    programs.append(Equal(self.ctx[i], self.ctx[j]))
        return programs

    def quantifier_count(self):
        return 0


# --------------------------------------------------
# ON* Predicate
# --------------------------------------------------


class ON_star(Program):
    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        # full mapping includes quantified vars + constants
        full_mapping = {**mapping, **input.get("constants", {})}
        obj1 = full_mapping[self.b1.id]
        obj2 = full_mapping[self.b2.id]
        return on_star_eval((obj1.x, obj1.y, obj1.z), (obj2.x, obj2.y, obj2.z))

    def expand(self):
        return []

    def __str__(self):
        return f"ON*({self.b1},{self.b2})"

    def quantifier_count(self):
        return 0


class Equal(Program):
    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        full_mapping = {**mapping, **input.get("constants", {})}
        obj1 = full_mapping[self.b1.id]
        obj2 = full_mapping[self.b2.id]
        return obj1.x == obj2.x and obj1.y == obj2.y and obj1.z == obj2.z

    def expand(self):
        return []

    def __str__(self):
        return f"({self.b1} == {self.b2})"

    def quantifier_count(self):
        return 0


# --------------------------------------------------
# ----------------- Top-Down Synthesizer -----------------
# --------------------------------------------------


def check_program(all_inputs, program, verbose=True):
    correct = 0
    for inp in all_inputs:
        result = program.evaluate_specific(inp)
        if result == inp["result"]:
            correct += 1
    acc = correct / len(all_inputs)
    if verbose:
        print(f"Accuracy: {acc:.3f} ({correct}/{len(all_inputs)})")
    return acc


def topdown_synthesize(
    all_inputs, max_programs=100000, target_accuracy=0.95, time_limit_sec=60
):
    start_time = time.time()
    queue = deque()
    program_counter = 0

    # Collect all constant IDs from first input
    constants_list = []
    if "constants" in all_inputs[0]:
        for cid in all_inputs[0]["constants"].keys():
            constants_list.append(Box(cid))

    root = Exists(constants_list, Program(constants_list))
    queue.append(root)

    added = 0
    while queue:
        program = queue.popleft()
        program_counter += 1

        if program_counter > max_programs:
            print("Program limit reached.")
            break
        # if time.time() - start_time > time_limit_sec:
        # print("Time limit reached.")
        # break

        print(f"\n[{program_counter}, {added}] Checking program: {program}")
        # if program_counter == 14:
        # import pdb
        # pdb.set_trace()

        if program.is_complete():
            acc = check_program(all_inputs, program)
            if acc >= target_accuracy:
                print("\n🎯 Found successful program!")
                print(program)

                # return program
        else:
            if added < max_programs:
                children = program.expand()
                for child in children:
                    child_str = str(child)
                    if child_str.count("∀") + child_str.count("∃") <= 2:
                        added += 1
                        queue.append(child)

    print("\n❌ No program found.")
    return None


def run_stack_hardcoded_dataset():
    all_inputs = []
    # scene 1
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.0)

    b2 = Box("2")
    b2.set_attribute(1.0, 0.0, 0.0)

    b3 = Box("3")
    b3.set_attribute(2.0, 0.0, 0.0)

    b4 = Box("4")
    b4.set_attribute(3.0, 0.0, 0.0)

    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 2
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.0)

    b2 = Box("2")
    b2.set_attribute(1.0, 0.0, 0.0)

    b3 = Box("3")
    b3.set_attribute(3.0, 0.0, 0.05)

    b4 = Box("4")
    b4.set_attribute(3.0, 0.0, 0.0)

    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b3},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b3},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b3},
            "result": False,
        }
    )

    # scene 3
    b5 = Box("5")
    b5.set_attribute(0.0, 0.0, 0.1)

    b0 = Box("0")
    b0.set_attribute(0.0, 0.0, 0.05)

    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.0)

    b2 = Box("2")
    b2.set_attribute(3.0, 0.0, 0.1)

    b3 = Box("3")
    b3.set_attribute(3.0, 0.0, 0.05)

    b4 = Box("4")
    b4.set_attribute(3.0, 0.0, 0.0)

    all_inputs.append(
        {
            "target": b5,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b0,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b1,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b5, b0, b1, b2, b3, b4],
            "constants": {"b_const": b2},
            "result": False,
        }
    )
    return all_inputs


def run_unstack_hardcoded_dataset():
    all_inputs = []
    # scene 1
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.2)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 2
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 3
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(4.0, 0.0, 0.05)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 4
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(4.0, 0.0, 0.05)

    b3 = Box("3")
    b3.set_attribute(6.0, 0.0, 0.05)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    return all_inputs


def run_reverse_hardcoded_dataset():
    all_inputs = []
    # scene 1
    b1 = Box("1")
    b1.set_attribute(0.0, 0.0, 0.2)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 2
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(0.0, 0.0, 0.15)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 3
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(2.0, 0.0, 0.1)

    b3 = Box("3")
    b3.set_attribute(0.0, 0.0, 0.1)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": True,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 4
    b1 = Box("1")
    b1.set_attribute(2.0, 0.0, 0.05)

    b2 = Box("2")
    b2.set_attribute(2.0, 0.0, 0.1)

    b3 = Box("3")
    b3.set_attribute(2.0, 0.0, 0.15)

    b4 = Box("4")
    b4.set_attribute(0.0, 0.0, 0.05)

    all_inputs.append(
        {
            "target": b1,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b2,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b3,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )
    all_inputs.append(
        {
            "target": b4,
            "all_box": [b1, b2, b3, b4],
            "constants": {"b_const": b4},
            "result": False,
        }
    )

    # scene 5
    # b1 = Box("1")
    # b1.set_attribute(2.0, 0.0, 0.05)

    # b2 = Box("2")
    # b2.set_attribute(2.0, 0.0, 0.1)

    # b3 = Box("3")
    # b3.set_attribute(2.0, 0.0, 0.15)

    # b4 = Box("4")
    # b4.set_attribute(2.0, 0.0, 0.2)

    # all_inputs.append(
    #     {
    #         "target": b1,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    # all_inputs.append(
    #     {
    #         "target": b2,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    # all_inputs.append(
    #     {
    #         "target": b3,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    # all_inputs.append(
    #     {
    #         "target": b4,
    #         "all_box": [b1, b2, b3, b4],
    #         "constants": {"b_const": b4},
    #         "result": False,
    #     }
    # )
    return all_inputs

if __name__ == "__main__":

    # import cProfile
    # import pstats

    # profiler = cProfile.Profile()
    # profiler.enable()

    all_inputs = run_reverse_hardcoded_dataset()

    best_program = topdown_synthesize(
        all_inputs, max_programs=15_000_000, target_accuracy=0.99, time_limit_sec=120
    )

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(100)

    print("\nBest program:")
    print(best_program)
