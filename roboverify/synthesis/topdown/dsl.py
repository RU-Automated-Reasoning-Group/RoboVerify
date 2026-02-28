import copy
import string
from typing import List

# --------------------------------------------------
# Utilities
# --------------------------------------------------


def fresh_var(ctx):
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
# Environment Objects
# --------------------------------------------------


class EnvObj:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.id


class Box(EnvObj):
    def __init__(self, id):
        super().__init__(id)
        self.set = False

    def set_attribute(self, x, y, z, gx=None, gy=None, gz=None):
        self.x = x
        self.y = y
        self.z = z
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.set = True


# --------------------------------------------------
# Base Program
# --------------------------------------------------


class Program:
    def __init__(self, ctx: List[EnvObj]):
        self.ctx = copy.deepcopy(ctx)

    def is_complete(self):
        return False

    def evaluate(self, input, mapping):
        raise NotImplementedError

    def expand(self):
        programs = [
            Exists(self.ctx, Program(self.ctx)),
            ForAll(self.ctx, Program(self.ctx)),
            Not(self.ctx, Program(self.ctx)),
            And(self.ctx, Program(self.ctx), Program(self.ctx)),
            Or(self.ctx, Program(self.ctx), Program(self.ctx)),
        ]
        programs.extend(Predicate(self.ctx).expand())
        return programs

    def add_object(self, obj):
        if hasattr(self, "ctx"):
            self.ctx.append(obj)

    def __str__(self):
        return "P"


# --------------------------------------------------
# Quantifiers
# --------------------------------------------------


class Exists(Program):
    def __init__(self, ctx, program):
        self.ctx = copy.deepcopy(ctx)
        self.program = copy.deepcopy(program)
        self.object = Box(fresh_var(self.ctx))
        self.program.add_object(self.object)

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        for obj in input["all_box"]:
            if self.program.evaluate(input, {**mapping, self.object.id: obj}):
                return True
        return False

    def evaluate_specific(self, input):
        return self.evaluate(input, {self.object.id: input["target"]})

    def expand(self):
        return [Exists(self.ctx, p) for p in self.program.expand()]

    def __str__(self):
        return f"∃{self.object}.({self.program})"


class ForAll(Program):
    def __init__(self, ctx, program):
        self.ctx = copy.deepcopy(ctx)
        self.program = copy.deepcopy(program)
        self.object = Box(fresh_var(self.ctx))
        self.program.add_object(self.object)

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
        return f"∀{self.object}.({self.program})"


# --------------------------------------------------
# Logical Operators
# --------------------------------------------------


class Not(Program):
    def __init__(self, ctx, program):
        self.ctx = copy.deepcopy(ctx)
        self.program = copy.deepcopy(program)

    def is_complete(self):
        return self.program.is_complete()

    def evaluate(self, input, mapping):
        return not self.program.evaluate(input, mapping)

    def expand(self):
        return [Not(self.ctx, p) for p in self.program.expand()]

    def __str__(self):
        return f"¬({self.program})"


class And(Program):
    def __init__(self, ctx, p1, p2):
        self.ctx = copy.deepcopy(ctx)
        self.program1 = copy.deepcopy(p1)
        self.program2 = copy.deepcopy(p2)

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


class Or(Program):
    def __init__(self, ctx, p1, p2):
        self.ctx = copy.deepcopy(ctx)
        self.program1 = copy.deepcopy(p1)
        self.program2 = copy.deepcopy(p2)

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


# --------------------------------------------------
# Predicate Base
# --------------------------------------------------


class Predicate(Program):
    def __init__(self, ctx):
        self.ctx = copy.deepcopy(ctx)

    def is_complete(self):
        return False

    def expand(self):
        programs = []
        for i in range(len(self.ctx)):
            for j in range(len(self.ctx)):
                if i != j:
                    programs.append(ON_star(self.ctx[i], self.ctx[j]))
                    programs.append(Equal(self.ctx[i], self.ctx[j]))
        return programs


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
        obj1 = mapping[self.b1.id]
        obj2 = mapping[self.b2.id]
        return obj1.z > obj2.z

    def expand(self):
        return []

    def __str__(self):
        return f"ON*({self.b1},{self.b2})"


class Equal(Program):
    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def is_complete(self):
        return True

    def evaluate(self, input, mapping):
        obj1 = mapping[self.b1.id]
        obj2 = mapping[self.b2.id]
        return obj1.z > obj2.z

    def expand(self):
        return []

    def __str__(self):
        return f"({self.b1} == {self.b2})"
