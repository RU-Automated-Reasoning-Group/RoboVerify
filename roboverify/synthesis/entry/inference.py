import itertools

from sympy import And, Equivalent, Implies, Not, Or, symbols, to_cnf

from synthesis.inference_lib import inference


def generate_truth_table(expression, var_names):
    """
    Generates and prints a truth table for a given SymPy boolean expression.

    Args:
        expression: The SymPy boolean expression.
        var_names: A list of strings with the names of the variables.
    """
    # Define symbols
    syms = symbols(var_names)
    if isinstance(syms, str):  # Handle single variable case
        syms = [syms]

    # Header
    header = " | ".join(var_names) + " || Result"
    separator = "-" * len(header)
    print(header)
    print(separator)

    # Generate all combinations of True/False
    for combo in itertools.product([True, False], repeat=len(syms)):
        # Create a dictionary mapping symbols to their current values
        model = dict(zip(syms, combo))

        # Evaluate the expression using subs
        result = expression.subs(model)

        # Print the row
        row = " | ".join(f"{val!s:^5}" for val in combo) + f" || {result!s:^6}"
        print(row)


# Example Usage:
# Define variables
# A, B, C = symbols('A, B, C')
# # Define a boolean expression: (A and B) or C
# expr = Or(And(A, B), C)

# print("Truth Table for (A & B) | C:")
# generate_truth_table(expr, ['A', 'B', 'C'])

if __name__ == "__main__":
    # x_on_n0, x_on_y, x_on0_y, b_on_x, y_on0_x = symbols("x_on_n0, x_on_y, x_on0_y, b_on_x, y_on0_x")

    # formula = ((x_on_n0 & Equivalent(x_on_y, x_on0_y)) | (Not(x_on_n0) & b_on_x & Equivalent(x_on_y, y_on0_x)))
    # cnf_formula = to_cnf(formula, simplify=True)
    # print(cnf_formula)
    # generate_truth_table(cnf_formula, ["x_on_n0", "x_on_y", "x_on0_y", "b_on_x", "y_on0_x"])

    # formula_1 = (b_on_x | x_on_n0) & (Not(b_on_x) | Not(x_on_n0))
    # formula_2 = Implies(x_on_n0 & Not(y_on0_x), x_on_y)
    # formula_3 = Implies(Not(x_on_n0) & Not(x_on0_y), x_on_y)
    # formula_4 = Implies(b_on_x & y_on0_x, x_on_y)
    # formula_5 = Implies(x_on_y, x_on_y)
    # inferred = (formula_1) & (formula_2) & (formula_3) & (formula_4) & (formula_5)
    # import pdb; pdb.set_trace()
    # generate_truth_table(inferred, ["x_on_n0", "x_on_y", "x_on0_y", "b_on_x", "y_on0_x"])
    inference.run_reverse_example()

    inferred_invariant, candidate_lists = inference.run_reverse_example()
    
