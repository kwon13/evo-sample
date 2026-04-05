"""
Prepare GAIR/LIMR dataset as initial seed programs.

Since LIMR has (prompt, answer) pairs but no generation programs,
we create wrapper programs that return the original problems.
These serve as initial seeds for the evolutionary process.

Additionally, we create inverse-construction templates inspired by
the problem types observed in LIMR.
"""

import json
import os
import re
import random
from pathlib import Path


def load_limr_dataset(split: str = "train", max_samples: int = 1000):
    """Load GAIR/LIMR dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("GAIR/LIMR", split=split)
        samples = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "prompt": item["prompt"],
                "answer": str(item["answer"]),
            })
        return samples
    except Exception as e:
        print(f"Error loading LIMR dataset: {e}")
        print("Using built-in sample problems instead.")
        return _builtin_samples()


def _builtin_samples():
    """Fallback sample problems if LIMR is unavailable."""
    return [
        {"prompt": "Solve for x: 2x + 5 = 13", "answer": "4"},
        {"prompt": "What is the sum of the first 10 positive integers?", "answer": "55"},
        {"prompt": "If f(x) = x^2 - 3x + 2, find f(5).", "answer": "12"},
        {"prompt": "Find the GCD of 48 and 36.", "answer": "12"},
        {"prompt": "How many 3-digit numbers are divisible by 7?", "answer": "128"},
    ]


def create_static_program(prompt: str, answer: str, idx: int) -> str:
    """Create a static wrapper program for a LIMR problem."""
    # Escape strings for embedding in Python code
    prompt_escaped = prompt.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    answer_escaped = answer.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    
    return f'''"""Static problem from LIMR dataset (sample {idx})."""
import random

def generate(seed):
    # This is a static seed problem from the LIMR dataset.
    # The evolutionary process will mutate this into dynamic generators.
    problem = "{prompt_escaped}"
    answer = "{answer_escaped}"
    return problem, answer
'''


def analyze_problem_type(prompt: str) -> str:
    """Simple heuristic to classify problem type."""
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ["solve", "equation", "find x", "root"]):
        return "algebra"
    elif any(w in prompt_lower for w in ["how many", "count", "ways", "choose", "combination"]):
        return "combinatorics"
    elif any(w in prompt_lower for w in ["mod", "remainder", "divisible", "gcd", "lcm"]):
        return "number_theory"
    elif any(w in prompt_lower for w in ["triangle", "circle", "area", "perimeter", "angle"]):
        return "geometry"
    elif any(w in prompt_lower for w in ["probability", "dice", "coin", "expected"]):
        return "probability"
    elif any(w in prompt_lower for w in ["sequence", "series", "sum of", "term"]):
        return "sequences"
    else:
        return "general"


INVERSE_TEMPLATES = {
    "algebra": '''"""Algebra problem generator (inverse construction, inspired by LIMR)."""
import random

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose answer
    x = rng.randint(-20, 20)
    
    # Step 2: Build equation from answer
    a = rng.randint(1, 10)
    b = rng.randint(-15, 15)
    c = a * x + b
    
    if b >= 0:
        problem = f"Solve for x: {a}x + {b} = {c}"
    else:
        problem = f"Solve for x: {a}x - {-b} = {c}"
    
    answer = str(x)
    return problem, answer
''',

    "number_theory": '''"""Number theory problem generator (inverse construction)."""
import random
import math

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose answer (GCD)
    gcd_val = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15])
    
    # Step 2: Construct two numbers with that GCD
    m1 = rng.randint(2, 10)
    m2 = rng.randint(2, 10)
    while math.gcd(m1, m2) != 1:
        m2 = rng.randint(2, 10)
    
    a = gcd_val * m1
    b = gcd_val * m2
    
    problem = f"Find the greatest common divisor (GCD) of {a} and {b}."
    answer = str(gcd_val)
    
    return problem, answer
''',

    "sequences": '''"""Sequence problem generator (inverse construction)."""
import random

def generate(seed):
    rng = random.Random(seed)
    
    # Step 1: Choose arithmetic sequence parameters and answer
    a0 = rng.randint(1, 20)
    d = rng.randint(1, 10)
    n = rng.randint(5, 15)
    
    # Sum of arithmetic sequence
    answer_val = n * (2 * a0 + (n - 1) * d) // 2
    
    # Step 2: Build problem
    problem = (f"Find the sum of the first {n} terms of an arithmetic sequence "
               f"where the first term is {a0} and the common difference is {d}.")
    answer = str(answer_val)
    
    return problem, answer
''',
}


def prepare_seed_programs(
    output_dir: str = "./seed_programs",
    max_static: int = 10,
    max_limr: int = 500,
):
    """
    Prepare seed programs from LIMR dataset.
    
    Creates:
    1. Static wrapper programs for a few LIMR problems (as initial seeds)
    2. Inverse construction templates for different problem types
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading LIMR dataset...")
    samples = load_limr_dataset(max_samples=max_limr)
    print(f"Loaded {len(samples)} samples")
    
    # Analyze problem types
    type_counts = {}
    for s in samples:
        ptype = analyze_problem_type(s["prompt"])
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    print(f"Problem type distribution: {type_counts}")
    
    # Create static seed programs from LIMR samples
    created = 0
    for i, sample in enumerate(samples[:max_static]):
        source = create_static_program(sample["prompt"], sample["answer"], i)
        fname = output_path / f"limr_static_{i:03d}.py"
        fname.write_text(source)
        created += 1
    
    # Create inverse construction templates
    for ptype, template in INVERSE_TEMPLATES.items():
        fname = output_path / f"template_{ptype}.py"
        fname.write_text(template)
        created += 1
    
    print(f"Created {created} seed programs in {output_dir}")
    
    # Save LIMR dataset as reference (for veRL training)
    ref_path = output_path.parent / "data" / "limr_reference.jsonl"
    ref_path.parent.mkdir(exist_ok=True)
    with open(ref_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} LIMR reference problems to {ref_path}")


if __name__ == "__main__":
    prepare_seed_programs()
