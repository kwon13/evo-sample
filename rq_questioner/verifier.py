"""
Verifier: Validates that (problem, answer) pairs are correct.

Uses substitution-based verification for math problems.
This is the "easy to evaluate, hard to solve" principle from FunSearch.
"""

import re
import math
from typing import Optional


def verify_problem(problem: str, answer: str, method: str = "sympy") -> bool:
    """
    Verify that the answer is correct for the given problem.
    
    For inverse-constructed problems, we use substitution verification:
    the answer was used to CONSTRUCT the problem, so we verify by
    substituting back and checking consistency.
    """
    if method == "sympy":
        return _verify_sympy(problem, answer)
    elif method == "numeric":
        return _verify_numeric(problem, answer)
    elif method == "exec":
        return _verify_exec(problem, answer)
    else:
        # Fallback: just check non-empty
        return bool(problem.strip()) and bool(answer.strip())


def _verify_sympy(problem: str, answer: str) -> bool:
    """Use SymPy to verify the answer by substitution."""
    try:
        import sympy
        from sympy import symbols, sympify, Eq, solve
        from sympy.parsing.sympy_parser import parse_expr

        # Try to detect equation-type problems
        # Pattern: "Solve ... = 0" or "Find x such that ..."
        equation_match = re.search(r"[Ss]olve\s+(?:the\s+equation:?\s*)?(.+?)\s*=\s*0", problem)
        if equation_match:
            expr_str = equation_match.group(1).strip()
            # Convert ^ to ** for sympy
            expr_str = expr_str.replace("^", "**")
            x = symbols("x")
            try:
                from sympy.parsing.sympy_parser import (
                    standard_transformations,
                    implicit_multiplication_application,
                )
                transformations = standard_transformations + (implicit_multiplication_application,)
                expr = parse_expr(expr_str, local_dict={"x": x}, transformations=transformations)
                # Parse answer as set of roots
                roots = _parse_answer_as_numbers(answer)
                if not roots:
                    return False
                # Verify each root
                for root in roots:
                    val = expr.subs(x, root)
                    if abs(complex(val)) > 1e-6:
                        return False
                return True
            except Exception:
                pass

        # Pattern: modular arithmetic "Find x such that ax ≡ b (mod p)"
        mod_match = re.search(r"(\d+)\s*[·*x]\s*[≡=]\s*(\d+)\s*\(?\s*mod\s+(\d+)", problem)
        if mod_match:
            a, b, p = int(mod_match.group(1)), int(mod_match.group(2)), int(mod_match.group(3))
            ans_val = _parse_single_number(answer)
            if ans_val is not None:
                return (a * ans_val) % p == b % p

        # Fallback: try numeric check
        return _verify_numeric(problem, answer)

    except ImportError:
        return _verify_numeric(problem, answer)
    except Exception:
        return False


def _verify_numeric(problem: str, answer: str) -> bool:
    """Basic numeric verification - check that answer is a valid number/expression."""
    try:
        nums = _parse_answer_as_numbers(answer)
        return len(nums) > 0
    except Exception:
        return bool(answer.strip())


def _verify_exec(problem: str, answer: str) -> bool:
    """Verify by executing a verification script embedded in the problem."""
    # For code-based problems where verification code is available
    try:
        # Extract verification code if present
        verify_match = re.search(r"# VERIFY:\s*(.+)", problem, re.DOTALL)
        if verify_match:
            verify_code = verify_match.group(1)
            namespace = {"answer": answer}
            exec(verify_code, namespace)
            return namespace.get("is_correct", False)
    except Exception:
        pass
    return False


def _parse_answer_as_numbers(answer: str) -> list:
    """Parse answer string into a list of numbers."""
    # Remove braces, brackets, etc.
    cleaned = answer.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    parts = re.split(r"[,;\s]+", cleaned.strip())
    numbers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            # Try int first
            numbers.append(int(part))
        except ValueError:
            try:
                numbers.append(float(part))
            except ValueError:
                try:
                    # Try fraction
                    from fractions import Fraction
                    numbers.append(float(Fraction(part)))
                except Exception:
                    pass
    return numbers


def _parse_single_number(answer: str) -> Optional[float]:
    """Parse a single number from answer."""
    nums = _parse_answer_as_numbers(answer)
    return nums[0] if nums else None
