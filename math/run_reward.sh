python reward.py \
  --input ./hmmt_run_qwen25_math/generations.jsonl \
  --output ./hmmt_run_qwen25_math/rewards_7b.jsonl \
  --model Qwen/Qwen2.5-Math-PRM-7B \
  --dtype bfloat16 --tensor-parallel-size 2
