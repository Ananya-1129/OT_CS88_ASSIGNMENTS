import numpy as np

# ---------------------------------------------------------
# Big-M Simplex Method
# ---------------------------------------------------------

M = 1000000

# Objective function:
# Max Z = 3x1 + 5x2
c = [3, 5]

# Constraints are represented as:
# [coefficients, sign, RHS]
constraints = [
    ([1, 1], ">=", 4),
    ([2, 1], "<=", 8),
    ([1, 2], "<=", 10)
]

# ---------------------------------------------------------
# Create variables
# ---------------------------------------------------------

variable_names = ["x1", "x2"]

for i, (_, sign, _) in enumerate(constraints):

    if sign == "<=":
        variable_names.append("s" + str(i + 1))

    elif sign == ">=":
        variable_names.append("s" + str(i + 1))
        variable_names.append("a" + str(i + 1))

    elif sign == "=":
        variable_names.append("a" + str(i + 1))


# Total number of variables
n = len(variable_names)
m = len(constraints)

# Tableau
A = np.zeros((m, n))
b = np.zeros(m)

basis = []
CB = []

column = 2

# ---------------------------------------------------------
# Construct initial tableau
# ---------------------------------------------------------

for i, (coeff, sign, rhs) in enumerate(constraints):

    A[i, 0:2] = coeff
    b[i] = rhs

    if sign == "<=":

        A[i, column] = 1
        basis.append(column)
        CB.append(0)

        column += 1

    elif sign == ">=":

        # Surplus variable
        A[i, column] = -1
        column += 1

        # Artificial variable
        A[i, column] = 1
        basis.append(column)
        CB.append(-M)

        column += 1

    elif sign == "=":

        # Artificial variable
        A[i, column] = 1
        basis.append(column)
        CB.append(-M)

        column += 1


# Complete objective coefficients
C = np.array(c + [0] * (n - len(c)), dtype=float)

iteration = 0

print("\n========== BIG-M SIMPLEX METHOD ==========\n")

while True:

    # Calculate Zj
    CB_array = np.array(CB)
    Zj = CB_array @ A

    # Calculate Cj - Zj
    Cj_Zj = C - Zj

    # Calculate objective value
    Z = CB_array @ b

    print("Iteration:", iteration)
    print("Basic Variables:", [variable_names[i] for i in basis])
    print("RHS:", np.round(b, 4))
    print("Cj - Zj:", np.round(Cj_Zj, 4))
    print("Z =", round(Z, 4))
    print()

    # -----------------------------------------------------
    # Check optimality
    # -----------------------------------------------------

    entering = np.argmax(Cj_Zj)

    if Cj_Zj[entering] <= 0:
        break

    # -----------------------------------------------------
    # Ratio test
    # -----------------------------------------------------

    ratios = []

    for i in range(m):

        if A[i, entering] > 0:
            ratios.append(b[i] / A[i, entering])
        else:
            ratios.append(np.inf)

    leaving = np.argmin(ratios)

    if ratios[leaving] == np.inf:
        print("The problem is unbounded.")
        break

    print("Entering Variable:",
          variable_names[entering])

    print("Leaving Variable:",
          variable_names[basis[leaving]])

    # -----------------------------------------------------
    # Pivot operation
    # -----------------------------------------------------

    pivot = A[leaving, entering]

    # Normalize pivot row
    A[leaving] = A[leaving] / pivot
    b[leaving] = b[leaving] / pivot

    # Make other entries in entering column zero
    for i in range(m):

        if i != leaving:

            factor = A[i, entering]

            A[i] = A[i] - factor * A[leaving]
            b[i] = b[i] - factor * b[leaving]

    # Update basis
    basis[leaving] = entering
    CB[leaving] = C[entering]

    iteration += 1

    print("------------------------------------------")


# ---------------------------------------------------------
# Find final solution
# ---------------------------------------------------------

solution = np.zeros(n)

for i in range(m):
    solution[basis[i]] = b[i]

print("\n========== FINAL SOLUTION ==========\n")

for i in range(n):
    print(variable_names[i], "=", round(solution[i], 4))

optimal_Z = C @ solution

print("\nOptimal Objective Value:")
print("Z =", round(optimal_Z, 4))

# Check artificial variables
artificial_value = solution[
    [i for i, name in enumerate(variable_names)
     if name.startswith("a")]
]

if np.allclose(artificial_value, 0):
    print("\nArtificial variable = 0")
    print("Hence the solution is feasible.")
else:
    print("\nArtificial variable is non-zero.")
    print("No feasible solution exists.")