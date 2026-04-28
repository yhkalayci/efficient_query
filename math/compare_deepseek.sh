rm -rf hmmt_deepseek/meta_eval_rew1_ver10
python compare.py \
  --generations hmmt_deepseek/generations.jsonl \
  --rewards    hmmt_deepseek/rewards_7b.jsonl \
  --reward-key r_last \
  --c-rew 1.0 --c-ver 10.0 \
  --n-perm 10 \
  --out-dir hmmt_deepseek/meta_eval_rew1_ver10
