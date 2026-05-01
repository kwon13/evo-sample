import random

CONCEPT_TYPE = "linear_algebra.linear_system_sum"
CONCEPT_GROUP = "linear_algebra"


def generate(seed):
    rng = random.Random(seed)

    x_sol = rng.randint(-5, 5)
    y_sol = rng.randint(-5, 5)
    z_sol = rng.randint(-5, 5)
    coeffs = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
    for _ in range(100):
        a = coeffs
        det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
               - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
               + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        if det != 0:
            break
        coeffs = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
    else:
        coeffs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rhs = [sum(coeffs[i][j] * [x_sol, y_sol, z_sol][j] for j in range(3)) for i in range(3)]
    answer = x_sol + y_sol + z_sol
    eqs = []
    for i in range(3):
        c = coeffs[i]
        eqs.append(f"{c[0]}x + {c[1]}y + {c[2]}z = {rhs[i]}")
    problem = (
        f"Solve the system: {eqs[0]}, {eqs[1]}, {eqs[2]}. Find x + y + z."
    )

    return problem, str(answer)
