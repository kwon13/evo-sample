# python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --n_evo 5 --candidates 8 --uncertainty_metric h --out_dir ./token_entropy --n_rollouts 5
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --n_evo 5 --candidates 8 --uncertainty_metric h_span_max --out_dir ./token_entropy_span_max --n_rollouts 5
