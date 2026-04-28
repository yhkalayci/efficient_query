rm -rf hmmt_smoketest_qwen25_14b_512/meta_eval_rew1_ver10
python compare.py \
  --generations hmmt_smoketest_qwen25_14b_512/generations.jsonl \
  --rewards    hmmt_smoketest_qwen25_14b_512/rewards_7b.jsonl \
  --reward-key r_last \
  --c-rew 1.0 --c-ver 10.0 \
  --n-perm 10 \
  --out-dir hmmt_smoketest_qwen25_14b_512/meta_eval_rew1_ver10


rm -rf hmmt_smoketest_qwen25_14b_512/meta_eval_rew1_ver20
python compare.py \
  --generations hmmt_smoketest_qwen25_14b_512/generations.jsonl \
  --rewards    hmmt_smoketest_qwen25_14b_512/rewards_7b.jsonl \
  --reward-key r_last \
  --c-rew 1.0 --c-ver 20.0 \
  --n-perm 10 \
  --out-dir hmmt_smoketest_qwen25_14b_512/meta_eval_rew1_ver20


rm -rf hmmt_smoketest_qwen25_14b_512/meta_eval_rew1_ver30
python compare.py \
  --generations hmmt_smoketest_qwen25_14b_512/generations.jsonl \
  --rewards    hmmt_smoketest_qwen25_14b_512/rewards_7b.jsonl \
  --reward-key r_last \
  --c-rew 1.0 --c-ver 30.0 \
  --n-perm 10 \
  --out-dir hmmt_smoketest_qwen25_14b_512/meta_eval_rew1_ver30
