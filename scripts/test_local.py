#!/usr/bin/env python
"""
Quick validation test - runs the core pipeline logic WITHOUT GPU.

Tests:
1. Seed program execution and verification
2. MAP-Elites grid operations
3. R_Q score computation
4. Training data preparation

Usage:
    python scripts/test_local.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rq_questioner.program import ProblemProgram
from rq_questioner.verifier import verify_problem
from rq_questioner.rq_score import compute_rq_full, h_prefilter
from rq_questioner.map_elites import MAPElitesGrid


def test_seed_programs():
    """Test that seed programs execute and produce valid outputs."""
    print("=" * 50)
    print("TEST 1: Seed Program Execution")
    print("=" * 50)

    seed_dir = os.path.join(os.path.dirname(__file__), "..", "seed_programs")
    passed = 0
    failed = 0

    for fname in sorted(os.listdir(seed_dir)):
        if not fname.endswith(".py"):
            continue

        source = open(os.path.join(seed_dir, fname)).read()
        prog = ProblemProgram(source_code=source)

        success_count = 0
        for seed in range(5):
            inst = prog.execute(seed)
            if inst is not None:
                success_count += 1

        if success_count >= 3:
            print(f"  [PASS] {fname}: {success_count}/5 seeds produced valid output")
            # Show one example
            inst = prog.execute(0)
            if inst:
                print(f"         Problem: {inst.problem[:80]}...")
                print(f"         Answer:  {inst.answer}")
            passed += 1
        else:
            print(f"  [FAIL] {fname}: only {success_count}/5 seeds worked")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed\n")
    return failed == 0


def test_verification():
    """Test the verifier on known problems."""
    print("=" * 50)
    print("TEST 2: Verification")
    print("=" * 50)

    test_cases = [
        ("Solve the equation: x^2 - 5x + 6 = 0", "[2, 3]", True),
        ("Find x such that 3x ≡ 6 (mod 7)", "2", True),
    ]

    # This test requires sympy for negative verification
    try:
        import sympy
        test_cases.append(
            ("Solve the equation: x^2 - 5x + 6 = 0", "[1, 4]", False)
        )
    except ImportError:
        print("  [SKIP] Negative verification test (requires sympy)")

    passed = 0
    for problem, answer, expected in test_cases:
        result = verify_problem(problem, answer)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] verify('{problem[:50]}...', '{answer}') "
              f"= {result} (expected {expected})")

    print(f"\nResult: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


def test_rq_score():
    """Test R_Q score computation."""
    print("=" * 50)
    print("TEST 3: R_Q Score Computation")
    print("=" * 50)

    # Test 1: p=0.5 should give max p(1-p)
    result = compute_rq_full([True, False, True, False] * 4, h_bar=2.0)
    print(f"  p=0.5, H=2.0: R_Q={result.rq_score:.4f}, p(1-p)={result.p_variance:.4f}")
    assert abs(result.p_variance - 0.25) < 0.01, "p(1-p) should be ~0.25"

    # Test 2: p=0 should give R_Q=0
    result = compute_rq_full([False] * 16, h_bar=2.0)
    print(f"  p=0.0, H=2.0: R_Q={result.rq_score:.4f}, p(1-p)={result.p_variance:.4f}")
    assert result.rq_score == 0.0, "R_Q should be 0 when p=0"

    # Test 3: p=1 should give R_Q=0
    result = compute_rq_full([True] * 16, h_bar=2.0)
    print(f"  p=1.0, H=2.0: R_Q={result.rq_score:.4f}, p(1-p)={result.p_variance:.4f}")
    assert result.rq_score == 0.0, "R_Q should be 0 when p=1"

    # Test 4: Higher H should give higher R_Q at same p
    r1 = compute_rq_full([True, False] * 8, h_bar=1.0)
    r2 = compute_rq_full([True, False] * 8, h_bar=3.0)
    print(f"  p=0.5, H=1.0: R_Q={r1.rq_score:.4f}")
    print(f"  p=0.5, H=3.0: R_Q={r2.rq_score:.4f}")
    assert r2.rq_score > r1.rq_score, "Higher H should give higher R_Q"

    # Test 5: H pre-filter
    assert h_prefilter(0.5, threshold=0.1) == True
    assert h_prefilter(0.05, threshold=0.1) == False
    print(f"  H pre-filter(0.5, thr=0.1) = True  [PASS]")
    print(f"  H pre-filter(0.05, thr=0.1) = False [PASS]")

    print(f"\nAll R_Q tests passed!\n")
    return True


def test_map_elites():
    """Test MAP-Elites grid operations."""
    print("=" * 50)
    print("TEST 4: MAP-Elites Grid")
    print("=" * 50)

    grid = MAPElitesGrid(n_h_bins=4, n_div_bins=4, h_range=(0.0, 4.0))
    print(f"  Grid size: {grid.n_h_bins}x{grid.n_div_bins} = {len(grid.grid)} niches")
    print(f"  Initial coverage: {grid.coverage():.2%}")

    # Insert some programs
    for i in range(10):
        source = f'def generate(seed):\n    return "Problem {i}", "Answer {i}"'
        prog = ProblemProgram(source_code=source)
        h_val = i * 0.4  # Varying H values
        rq_val = 0.1 * (i + 1)
        
        inserted = grid.try_insert(
            program=prog,
            h_value=h_val,
            problem_text=f"Problem {i}",
            rq_score=rq_val,
        )
        
    print(f"  After 10 insertions: coverage={grid.coverage():.2%}")
    print(f"  Mean R_Q: {grid.mean_rq():.4f}")

    # Test champion replacement
    source_better = 'def generate(seed):\n    return "Better problem", "Better answer"'
    prog_better = ProblemProgram(source_code=source_better)
    replaced = grid.try_insert(
        program=prog_better,
        h_value=0.5,  # Same H bin as program 1
        problem_text="Better problem",
        rq_score=100.0,  # Much higher R_Q
    )
    print(f"  Champion replacement with higher R_Q: {replaced}")
    assert replaced, "Should replace champion with higher R_Q"

    # Test sampling
    parent = grid.sample_parent()
    print(f"  Sampled parent: id={parent.program_id if parent else 'None'}")
    assert parent is not None, "Should be able to sample a parent"

    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        grid.save(tmpdir)
        grid2 = MAPElitesGrid(n_h_bins=4, n_div_bins=4)
        grid2.load(tmpdir)
        print(f"  Save/load: {len(grid2.get_all_champions())} champions recovered")

    stats = grid.stats()
    print(f"  Final stats: {stats}")
    print(f"\nAll MAP-Elites tests passed!\n")
    return True


def test_pipeline_dry_run():
    """Test pipeline initialization without models."""
    print("=" * 50)
    print("TEST 5: Pipeline Dry Run")
    print("=" * 50)

    from rq_questioner.pipeline import EvolutionaryPipeline, PipelineConfig
    import tempfile

    seed_dir = os.path.join(os.path.dirname(__file__), "..", "seed_programs")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            num_epochs=1,
            num_generations=3,
            candidates_per_generation=2,
            num_rollouts=8,
            output_dir=tmpdir,
            seed_programs_dir=seed_dir,
        )

        pipeline = EvolutionaryPipeline(config)

        # Load seed programs
        programs = pipeline.load_seed_programs(seed_dir)
        print(f"  Loaded {len(programs)} seed programs")
        assert len(programs) > 0, "Should load at least one seed program"

        # Initialize grid
        pipeline.initialize_grid(programs)
        print(f"  Grid coverage after init: {pipeline.grid.coverage():.2%}")

        # Prepare training data
        training_data = pipeline._prepare_training_data(epoch=0)
        print(f"  Prepared {len(training_data)} training instances")

        if training_data:
            print(f"  Sample: {training_data[0]['prompt'][:60]}...")

    print(f"\nPipeline dry run passed!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("R_Q Evolutionary Pipeline - Local Validation Tests")
    print("=" * 60 + "\n")

    results = {
        "Seed Programs": test_seed_programs(),
        "Verification": test_verification(),
        "R_Q Score": test_rq_score(),
        "MAP-Elites": test_map_elites(),
        "Pipeline Dry Run": test_pipeline_dry_run(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed! Ready for GPU experiments.")
    else:
        print("Some tests failed. Fix issues before running on GPU.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
