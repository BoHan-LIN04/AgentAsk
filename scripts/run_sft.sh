set -e
python -m src.sft_train --config configs/config_sft.yaml
python -m src.eval_sft --model_path outputs/Your Model Path --data data/clarify_samples.jsonl
