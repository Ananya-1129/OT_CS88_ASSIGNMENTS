import numpy as np

# ============================================================
# TRANSPORTATION PROBLEM
# VAM + MODI METHOD
# ============================================================

# Cost matrix
cost = np.array([
    [4, 15, 2],
    [10, 2, 11],
    [8, 13, 3]
], dtype=float)

# Supply
supply = np.array([33, 37, 14], dtype=float)

# Demand
demand = np.array([11, 39, 34], dtype=float)

m, n = cost.shape


# ============================================================
# FUNCTION TO CALCULATE VAM INITIAL SOLUTION
# ============================================================

def vogel_approximation(cost, supply, demand):

    allocation = np.zeros((m, n))

    s = supply.copy()
    d = demand.copy()

    while np.any(s > 0) and np.any(d > 0):

        # -------------------------------
        # Row penalties
        # -------------------------------

        row_penalty = np.full(m, -1.0)

        for i in range(m):

            if s[i] > 0:

                values = [
                    cost[i][j]
                    for j in range(n)
                    if d[j] > 0
                ]

                if len(values) >= 2:

                    values.sort()

                    row_penalty[i] = (
                        values[1] - values[0]
                    )

        # -------------------------------
        # Column penalties
        # -------------------------------

        col_penalty = np.full(n, -1.0)

        for j in range(n):

            if d[j] > 0:

                values = [
                    cost[i][j]
                    for i in range(m)
                    if s[i] > 0
                ]

                if len(values) >= 2:

                    values.sort()

                    col_penalty[j] = (
                        values[1] - values[0]
                    )

        # -------------------------------
        # Select largest penalty
        # -------------------------------

        if np.max(row_penalty) >= np.max(col_penalty):

            i = np.argmax(row_penalty)

            j = min(
                [j for j in range(n) if d[j] > 0],
                key=lambda j: cost[i][j]
            )

        else:

            j = np.argmax(col_penalty)

            i = min(
                [i for i in range(m) if s[i] > 0],
                key=lambda i: cost[i][j]
            )

        # -------------------------------
        # Allocate
        # -------------------------------

        quantity = min(s[i], d[j])

        allocation[i][j] = quantity

        s[i] -= quantity
        d[j] -= quantity

    return allocation


# ============================================================
# FUNCTION TO FIND BASIC CELLS
# ============================================================

def get_basic_cells(allocation):

    basic = allocation > 0

    required = m + n - 1

    # Handle degeneracy
    if np.sum(basic) < required:

        # Find an empty cell that does not form a loop
        for i in range(m):
            for j in range(n):

                if not basic[i][j]:

                    basic[i][j] = True

                    if np.sum(basic) == required:
                        return basic

                    basic[i][j] = False

    return basic


# ============================================================
# FUNCTION TO CALCULATE U AND V
# ============================================================

def calculate_uv(basic):

    u = [None] * m
    v = [None] * n

    # Set u1 = 0
    u[0] = 0

    changed = True

    while changed:

        changed = False

        for i in range(m):

            for j in range(n):

                if basic[i][j]:

                    # ui + vj = cij

                    if u[i] is not None and v[j] is None:

                        v[j] = cost[i][j] - u[i]

                        changed = True

                    elif v[j] is not None and u[i] is None:

                        u[i] = cost[i][j] - v[j]

                        changed = True

    return u, v


# ============================================================
# FUNCTION TO CALCULATE DELTA
# ============================================================

def calculate_deltas(basic, u, v):

    deltas = {}

    for i in range(m):

        for j in range(n):

            if not basic[i][j]:

                delta = cost[i][j] - (u[i] + v[j])

                deltas[(i, j)] = delta

    return deltas


# ============================================================
# FIND CLOSED LOOP
# ============================================================

def find_cycle(basic, start):

    # Find a closed loop using DFS
    # Start cell is the entering cell

    def dfs(path):

        current = path[-1]

        # We need at least 4 cells
        if len(path) >= 4:

            # Same row or same column as starting cell
            if (
                current[0] == start[0]
                or current[1] == start[1]
            ):

                # Must alternate row and column
                if current[0] == start[0] or current[1] == start[1]:
                    return path

        i, j = current

        # Alternate between row and column movement
        move_row = len(path) % 2 == 1

        candidates = []

        if move_row:

            # Move in same row
            for col in range(n):

                if col != j:
                    candidates.append((i, col))

        else:

            # Move in same column
            for row in range(m):

                if row != i:
                    candidates.append((row, j))

        for cell in candidates:

            # Other cells must be basic,
            # except the starting cell

            if cell == start:

                if len(path) >= 4:
                    return path

            elif basic[cell[0]][cell[1]]:

                if cell not in path:

                    result = dfs(path + [cell])

                    if result is not None:
                        return result

        return None

    return dfs([start])


# ============================================================
# MODI IMPROVEMENT
# ============================================================

def modi(allocation):

    iteration = 1

    while True:

        print("\n")
        print("==============================================")
        print("MODI ITERATION", iteration)
        print("==============================================")

        basic = get_basic_cells(allocation)

        u, v = calculate_uv(basic)

        print("\nU values =", u)
        print("V values =", v)

        deltas = calculate_deltas(basic, u, v)

        print("\nOpportunity Costs:")

        for cell, delta in deltas.items():

            i, j = cell

            print(
                f"Delta[{i+1},{j+1}] = "
                f"{delta}"
            )

        # Check optimality
        entering = min(
            deltas,
            key=deltas.get
        )

        minimum_delta = deltas[entering]

        # ----------------------------------------------------
        # If all delta >= 0, solution is optimal
        # ----------------------------------------------------

        if minimum_delta >= 0:

            print("\nAll Delta values >= 0.")

            print("Solution is OPTIMAL.")

            break

        # ----------------------------------------------------
        # Negative delta -> improvement required
        # ----------------------------------------------------

        print(
            "\nNegative Delta found at:",
            f"x{entering[0]+1}{entering[1]+1}"
        )

        print("Solution is NOT optimal.")

        # Find closed loop
        cycle = find_cycle(basic, entering)

        print("\nClosed loop:")

        for k, cell in enumerate(cycle):

            sign = "+" if k % 2 == 0 else "-"

            print(
                f"x{cell[0]+1}{cell[1]+1} ({sign})"
            )

        # ----------------------------------------------------
        # Calculate theta
        # ----------------------------------------------------

        minus_cells = cycle[1::2]

        theta = min(
            allocation[i][j]
            for i, j in minus_cells
        )

        print("\nTheta =", theta)

        # ----------------------------------------------------
        # Adjust allocation
        # ----------------------------------------------------

        for k, (i, j) in enumerate(cycle):

            if k % 2 == 0:

                allocation[i][j] += theta

            else:

                allocation[i][j] -= theta

        print("\nNew Allocation:")

        for row in allocation:
            print(row)

        iteration += 1

    return allocation


# ============================================================
# MAIN PROGRAM
# ============================================================

print("================================================")
print("       TRANSPORTATION PROBLEM")
print("       VAM + MODI METHOD")
print("================================================")


print("\nCost Matrix:")

for row in cost:
    print(row)

print("\nSupply =", supply)
print("Demand =", demand)


# ============================================================
# VAM
# ============================================================

allocation = vogel_approximation(
    cost,
    supply,
    demand
)

print("\n")
print("================================================")
print("              VAM SOLUTION")
print("================================================")

print("\nInitial Allocation:")

for row in allocation:
    print(row)

vam_cost = np.sum(allocation * cost)

print("\nVAM Initial Cost =", vam_cost)


# ============================================================
# MODI
# ============================================================

allocation = modi(allocation)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("================================================")
print("              FINAL RESULT")
print("================================================")

print("\nOptimal Allocation:")

for row in allocation:
    print(row)

final_cost = np.sum(allocation * cost)

print("\nMinimum Transportation Cost =", final_cost)

print("\n================================================")