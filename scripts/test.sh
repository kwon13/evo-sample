python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric semantic_entropy  --out_dir ./semantic_entropy
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric gini  --out_dir ./gini
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric step_max_entropy --out_dir ./step_max_entropy
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric entropy  --out_dir ./entropy
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric vote_entropy  --out_dir ./vote_entropy
