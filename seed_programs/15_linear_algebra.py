import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["det_3x3", "solve_system"])

    if problem_type == "det_3x3":
        m = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        answer = det
        rows = [f"[{m[i][0]}, {m[i][1]}, {m[i][2]}]" for i in range(3)]
        problem = (
            f"Compute the determinant of the 3x3 matrix "
            f"with rows {rows[0]}, {rows[1]}, {rows[2]}."
        )

    else:
        x_sol = rng.randint(-5, 5)
        y_sol = rng.randint(-5, 5)
        z_sol = rng.randint(-5, 5)
        coeffs = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
        while True:
            a = coeffs
            det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                   - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                   + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
            if det != 0:
                break
            coeffs = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
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
