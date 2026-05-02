# Evolution Comparison: `span_run1` vs `span_run2`

- Run A: `rq_output/verl_ckpt_grpo/evolution_logs` step `9`
- Run B: `rq_output/verl_ckpt/evolution_logs` step `5`

## 1. Metric summary

### span_run1
- step: 9
- uncertainty_metric: `h`
- diversity_axis: `concept_group`
- coverage: 10.0%
- champions: 6 / 60
- hard_champions (H>=H2): 0
- mean R_Q: 0.0598
- max R_Q: 0.1081
- champion p_hat: 0.48 ± 0.11
- champion H: 0.25 ± 0.11

### span_run2
- step: 5
- uncertainty_metric: `h_span_max`
- diversity_axis: `concept_group`
- coverage: 30.0%
- champions: 18 / 60
- hard_champions (H>=H2): 8
- mean R_Q: 0.2590
- max R_Q: 1.0818
- champion p_hat: 0.30 ± 0.16
- champion H: 1.87 ± 1.70

## 2. Niche coverage overlap

- Filled in BOTH: **6**
- Filled only in `span_run1`: **0**
- Filled only in `span_run2`: **12**

## 3. Shared niches — same cell, different metric

| niche | span_run1 R_Q | span_run2 R_Q | Δ | span_run1: problem | span_run2: problem |
|---|---|---|---|---|---|
| (0, 0) | 0.040 | 0.127 | +0.087 | Find the number of solutions to the congruence equation 55x ≡ 7 (mod 365). | Modulo the prime p=163, g=2 is a primitive root. Let a be the least positive residue of g^108 modulo p, so a=58. How many residue classes x ... |
| (0, 1) | 0.036 | 0.025 | -0.011 | 8 people each write their name on a card. The cards are shuffled and redistributed. In how many redistributions do exactly 4 people receive ... | There are 9 people in a group. How many ways can you choose a committee of 7 people? cannot include any of the 4 excluded members |
| (0, 2) | 0.108 | 0.132 | +0.024 | A geometric sequence has the first term 7 and a common ratio 5. Calculate the sum of the first 3 terms of the sequence. However, there is a ... | A geometric sequence has first term 5 and common ratio 3. Let S be the sum of the first 4 terms. An arithmetic sequence has first term S and... |
| (0, 3) | 0.057 | 0.067 | +0.010 | Find the number of solutions to the congruence equation 55x ≡ 7 (mod 365). | The quadratic equation x^2 + -6x + -7 = 0 has two real roots r1 and r2. Let u = 2 + 1/r1 and v = 2 + 1/r2. Consider the monic quadratic equa... |
| (0, 4) | 0.047 | 0.123 | +0.076 | Given a circle with radius 5 and a point at a distance 4 from the center, what is the power of the point with respect to the circle? Additio... | In a cyclic quadrilateral with angles 7, 7, 1, and 5, and sides of lengths 30, 30.0, 4.285714285714286, and 21.42857142857143, find the sum ... |
| (0, 5) | 0.071 | 0.000 | -0.071 | Given positive integers a = 4 and b = 4, what is the ratio of the arithmetic mean to the geometric mean of a and b? Additionally, explain wh... | Let a and b be positive integers, with a = 7 and b = 7. Find the difference between the arithmetic mean and the geometric mean of a and b. |

## 4. Niches only in `span_run1`

_(none — both runs covered the same niches that span_run1 covered)_

## 5. Niches only in `span_run2`

| niche | concept | RQ | p | H | problem |
|---|---|---|---|---|---|
| (1, 0) | number_theory.kth_root_mod_prime | 0.275 | 0.38 | 1.17 | Modulo the prime p=97, g=5 is a primitive root. Let a be the least positive residue of g^54 modulo p, so a=89. How many residue classes x mo... |
| (1, 1) | combinatorics.committee_count | 0.132 | 0.25 | 0.71 | A club has 8 men and 7 women. A committee of 3 people is to be formed with at least 1 man and at least 1 woman. One member of the committee ... |
| (1, 3) | algebra.quadratic_vieta_reciprocal | 0.098 | 0.12 | 0.90 | Given the quadratic equation 7x^2 + 7x + 1 = 0, find the product of its roots. |
| (1, 4) | geometry.trig_area | 0.219 | 0.62 | 0.94 | Triangle ABC has A = (0, 0), B = (10, 8), and C = (7, 6). Find its area. |
| (3, 3) | algebra.quadratic_vieta_reciprocal | 0.469 | 0.38 | 2.00 | Given the quadratic equation 4x^2 + 4x + 1 = 0, find the product of its roots. |
| (4, 1) | combinatorics.committee_count | 0.635 | 0.38 | 2.71 | There are 9 people in a group. How many ways can you choose a committee of 7 people? |
| (4, 3) | algebra.quadratic_vieta_reciprocal | 0.325 | 0.12 | 2.98 | Given the quadratic equation 4x^2 + 4x + 1 = 0, find the product of its roots. |
| (4, 4) | geometry.ptolemy_cyclic_quadrilateral | 0.000 | 0.00 | 2.42 | In a cyclic quadrilateral, the lengths of the sides are a=8, b=8, c=5, and d=7. Using Ptolemy's theorem, find the product of the lengths of ... |
| (7, 0) | number_theory.legendre_symbol | 0.486 | 0.12 | 4.45 | The Legendre symbol (a/p) is defined as 1 if the positive integer a is a quadratic residue modulo the prime p, and -1 otherwise. Given that ... |
| (7, 3) | linear_algebra.linear_system_sum | 0.000 | 0.00 | 4.32 | Solve the following system of linear equations: 1*x + -5*y + -1*z = 8 3*x + 2*y + 1*z = -4 -1*x + 2*y + 0*z = 6  What is the value of x? |
| (7, 4) | geometry.trig_area | 0.466 | 0.12 | 4.26 | A triangle has side lengths 1, 8.94427190999916, and 9. It is a acute triangle. What is the area of the triangle? |
| (9, 3) | algebra.quadratic_vieta_reciprocal | 1.082 | 0.25 | 5.77 | Given the quadratic equation 4x^2 + 4x + 1 = 0, find the product of its roots. |

## 6. Top-5 champions per run

### span_run1 — top 5 champions by R_Q

**#1** niche=(0,2) R_Q=0.1081 p=0.38 H=0.461 concept=`sequence.geometric_to_arithmetic_sum`

Problem:

```
A geometric sequence has the first term 7 and a common ratio 5. Calculate the sum of the first 3 terms of the sequence. However, there is a slight ambiguity in the problem statement. Consider whether the sum should be interpreted as (a(r^n - 1)/(r - 1)) or (a(r^n + 1)/(r + 1)). Which interpretation is correct, and why?
```

Answer: `217`

**#2** niche=(0,5) R_Q=0.0708 p=0.38 H=0.302 concept=`inequality.am_gm_product`

Problem:

```
Given positive integers a = 4 and b = 4, what is the ratio of the arithmetic mean to the geometric mean of a and b? Additionally, explain why the ratio equals 1 when a = b.
```

Answer: `1.0`

**#3** niche=(0,3) R_Q=0.0571 p=0.62 H=0.244 concept=`algebra.quadratic_vieta_reciprocal`

Problem:

```
Find the number of solutions to the congruence equation 55x ≡ 7 (mod 365).
```

Answer: `0`

**#4** niche=(0,4) R_Q=0.0469 p=0.50 H=0.187 concept=`geometry.power_of_point_secants`

Problem:

```
Given a circle with radius 5 and a point at a distance 4 from the center, what is the power of the point with respect to the circle? Additionally, explain why the power is positive when the point is outside the circle and negative when it is inside. Additionally, consider a secant line intersecting the circle at two points. What is the relationship between the power of the point and the length of the secant line? The power of the point is -9, and the length of the secant line is 2.
```

Answer: `-9`

**#5** niche=(0,0) R_Q=0.0397 p=0.38 H=0.169 concept=`number_theory.crt_count`

Problem:

```
Find the number of solutions to the congruence equation 55x ≡ 7 (mod 365).
```

Answer: `0`

### span_run2 — top 5 champions by R_Q

**#1** niche=(9,3) R_Q=1.0818 p=0.25 H=5.769 concept=`algebra.quadratic_vieta_reciprocal`

Problem:

```
Given the quadratic equation 4x^2 + 4x + 1 = 0, find the product of its roots.
```

Answer: `0.25`

**#2** niche=(4,1) R_Q=0.6346 p=0.38 H=2.707 concept=`combinatorics.committee_count`

Problem:

```
There are 9 people in a group. How many ways can you choose a committee of 7 people?
```

Answer: `36`

**#3** niche=(7,0) R_Q=0.4865 p=0.12 H=4.448 concept=`number_theory.legendre_symbol`

Problem:

```
The Legendre symbol (a/p) is defined as 1 if the positive integer a is a quadratic residue modulo the prime p, and -1 otherwise. Given that p=97 and a=54, compute the value of (a/p).
```

Answer: `1`

**#4** niche=(3,3) R_Q=0.4692 p=0.38 H=2.002 concept=`algebra.quadratic_vieta_reciprocal`

Problem:

```
Given the quadratic equation 4x^2 + 4x + 1 = 0, find the product of its roots.
```

Answer: `0.25`

**#5** niche=(7,4) R_Q=0.4657 p=0.12 H=4.258 concept=`geometry.trig_area`

Problem:

```
A triangle has side lengths 1, 8.94427190999916, and 9. It is a acute triangle. What is the area of the triangle?
```

Answer: `4.47213595499958`
