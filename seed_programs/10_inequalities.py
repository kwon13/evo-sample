import random

CONCEPT_TYPE = "inequality.am_gm_product"
CONCEPT_GROUP = "inequality"


def generate(seed):
    rng = random.Random(seed)

    s = rng.choice([6, 9, 12, 15, 18])
    k = rng.choice([1, 2, 3])

    # Maximize (a+k)(b+k)(c+k)
    # subject to a+b+c=s, a,b,c>0.
    # Let x=a+k, y=b+k, z=c+k.
    # Then x+y+z=s+3k, so product <= ((s+3k)/3)^3.
    max_product = ((s + 3 * k) // 3) ** 3

    problem = (
        f"Let a, b, c be positive real numbers with a + b + c = {s}. "
        f"What is the maximum value of (a + {k})(b + {k})(c + {k})?"
    )

    return problem, str(max_product)