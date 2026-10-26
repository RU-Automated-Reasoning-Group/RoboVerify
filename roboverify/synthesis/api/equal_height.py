"""EqualHeight placement effects for complete uniform towers on a common table.

The source must be clear and a physical destination must be distinct and clear;
public Put checks these enabling conditions separately. The abstract domain
axioms alone do not guarantee the complete support geometry used by these rules.
"""

import z3
from synthesis.util.symbols import fresh_const, rewrite_quantifier


def rewrite_equal_height(expr, source, target, context):
    """Rewrite a placement's EqualHeight atoms; target=None means the table.

    A table placement is level with exactly the other roots. A block placement
    is level with x exactly when x has an immediate support level with target.
    Direct support is defined using ON* and absence of an intermediate block.
    Thus the block rule still introduces exists/forall quantifiers; replacing
    Higher by height equality does not by itself give quantifier-free WPs.
    """
    operands = (source,) if target is None else (source, target)
    if z3.is_quantifier(expr):
        return rewrite_quantifier(
            expr,
            lambda body: rewrite_equal_height(body, source, target, context),
            avoid=operands,
        )
    if not z3.is_app(expr):
        return expr
    if expr.decl() != context.EqualHeight:
        return expr.decl()(
            *(rewrite_equal_height(c, source, target, context) for c in expr.children())
        )

    left, right = expr.children()

    def at_new_height(other):
        support = fresh_const(
            context.BoxSort, "height_support", avoid=(expr, *operands)
        )
        physical = other != context.get_consts("tbl") if context.use_tbl else True
        if target is None:
            return z3.And(
                physical,
                z3.ForAll(
                    [support],
                    z3.Implies(context.ON_star(other, support), other == support),
                ),
            )
        between = fresh_const(
            context.BoxSort, "height_between", avoid=(expr, *operands, support)
        )
        return z3.And(
            physical,
            z3.Exists(
                [support],
                z3.And(
                    other != support,
                    context.ON_star(other, support),
                    context.EqualHeight(support, target),
                    z3.ForAll(
                        [between],
                        z3.Implies(
                            z3.And(
                                context.ON_star(other, between),
                                context.ON_star(between, support),
                            ),
                            z3.Or(between == other, between == support),
                        ),
                    ),
                ),
            ),
        )

    return z3.Or(
        z3.And(left == source, right == source),
        z3.And(left != source, right != source, context.EqualHeight(left, right)),
        z3.And(left == source, right != source, at_new_height(right)),
        z3.And(right == source, left != source, at_new_height(left)),
    )
