# 1. Generate. Medium+hard only, first 5 problems, 16 samples each.
python gen_lcb.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --difficulty medium,hard \
  --n-samples 512 \
  --n-problems 100 \
  --out-dir run_mh_lcb_3b_16

# 2. Filter solvable
python filter_solvable.py \
  --input smoke_lcb_3b_16/generations.jsonl \
  --output smoke_lcb_3b_16/generations_solvable.jsonl

# 3. Reward (CodeScaler-4B for smoke)
python reward.py \
  --generations run_mh_lcb_3b_16/generations.jsonl \
  --model LARK-Lab/CodeScaler-4B \
  --batch-size 4 \
  --dtype bfloat16 \
  --out run_mh_lcb_3b_16/rewards.jsonl


python reward_vllm.py \
  --generations run_mh_lcb_3b_16/generations.jsonl \
  --model LARK-Lab/CodeScaler-8B \
  --tp 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --out run_mh_lcb_3b_16/rewards.jsonl


# 4. AUC
python auc.py \
  --generations smoke_lcb_3b_16/generations_solvable.jsonl \
  --rewards smoke_lcb_3b_16/rewards.jsonl \
  --bootstrap 200

# 5. Compare
python compare.py \
  --generations smoke_lcb_3b_16/generations_solvable.jsonl \
  --rewards smoke_lcb_3b_16/rewards.jsonl \
  --reward-key r_score \
  --c-rew 1.0 --c-ver 10.0 \
  --n-perm 5 \
  --out-dir smoke_lcb_3b_16/meta_eval_smoke




python compare.py \
  --generations run_mh_lcb_3b_16/generations_verified.jsonl \
  --rewards run_mh_lcb_3b_16/rewards.jsonl \
  --reward-key r_score \
  --c-rew 1.0 --c-ver 10.0 \
  --n-perm 10 \
  --out-dir run_mh_lcb_3b_16/meta_eval





python reward2.py \
  --generations run_mh_lcb_3b_16/generations.jsonl \
  --model LARK-Lab/CodeScaler-4B \
  --batch-size 16 \
  --dtype bfloat16 \
  --attn flash_attention_2 \
  --out run_mh_lcb_3b_16/rewards2.jsonl


# 4. AUC
python auc.py \
  --generations run_mh_lcb_3b_16/generations.jsonl \
  --rewards run_mh_lcb_3b_16/rewards.jsonl \
  --bootstrap 200