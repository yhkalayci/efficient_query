python gen4.py \
  --model Qwen/Qwen2.5-Math-7B \
  --dataset hendrycks_math \
  --split test \
  --level 5 \
  --subjects Algebra Geometry Number_theory \
  --candidate-limit 256 \
  --pilot-samples 64 \
  --final-samples 256