python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric semantic_entropy  --out_dir ./semantic_entropy_new --n_evo 5
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric gini  --out_dir ./gini_new --n_evo 5
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric step_max_entropy --out_dir ./step_max_entropy_new --n_evo 5
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric entropy  --out_dir ./entropy_new --n_evo 5
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric vote_entropy  --out_dir ./vote_entropy_new --n_evo 5
