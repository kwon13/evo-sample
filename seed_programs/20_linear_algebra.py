"""
Linear Algebra: determinants, matrix equations, eigenvalues.
"""
import random


def generate(seed):
    rng = random.Random(seed)

    problem_type = rng.choice(["det_3x3", "solve_system", "trace_det"])

    if problem_type == "det_3x3":
        # 3x3 determinant with small integers
        m = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
        det = (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
             - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
             + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        answer = det
        rows = [f"  [{m[i][0]:>2}, {m[i][1]:>2}, {m[i][2]:>2}]" for i in range(3)]
        matrix_str = "\n".join(rows)
        problem = (
            f"Compute the determinant of the 3×3 matrix:\n{matrix_str}"
        )

    elif problem_type == "solve_system":
        # 2x2 system with integer solution: ax + by = e, cx + dy = f
        x_sol = rng.randint(-5, 5)
        y_sol = rng.randint(-5, 5)
        a = rng.randint(1, 5)
        b = rng.randint(-5, 5)
        c = rng.randint(-5, 5)
        d = rng.randint(1, 5)
        while a*d - b*c == 0:  # ensure non-singular
            d = rng.randint(1, 5)
        e = a * x_sol + b * y_sol
        f_val = c * x_sol + d * y_sol
        # ask for x + y
        answer = x_sol + y_sol
        problem = (
            f"Solve the system of equations:\n"
            f"  {a}x + {b}y = {e}\n"
            f"  {c}x + {d}y = {f_val}\n"
            f"Find x + y."
        )

    else:  # trace_det: 2x2 matrix, compute trace^2 - 2*det
        # For 2x2 [[a,b],[c,d]]: trace = a+d, det = ad-bc
        # trace^2 - 2*det = (a+d)^2 - 2(ad-bc) = a^2 + d^2 + 2bc
        # This equals sum of squares of eigenvalues: λ1² + λ2²
        a = rng.randint(-4, 4)
        b = rng.randint(-4, 4)
        c = rng.randint(-4, 4)
        d = rng.randint(-4, 4)
        trace = a + d
        det = a * d - b * c
        answer = trace**2 - 2 * det
        problem = (
            f"Let M be the 2×2 matrix [[{a}, {b}], [{c}, {d}]]. "
            f"If λ₁ and λ₂ are the eigenvalues of M, compute λ₁² + λ₂². "
            f"(Hint: use the relation with trace and determinant.)"
        )

    return problem, str(answer)
