# Adaptive Generate-Rank-Verify: Inference-Time Search with Costly Verification

Code for the experiments in [*Adaptive Generate-Rank-Verify: Inference-Time Search with Costly Verification*](https://arxiv.org/abs/2605.17609) by Shaddin Dughmi*, Mahdi Haghifam*, Yusuf Hakan Kalayci*.

Evaluates how well **reward-guided generation** reduces the compute needed to solve math competition and coding problems. The core question: given a fixed budget, is it cheaper to generate many samples and verify one, or to generate fewer and use a reward model to pick which one to verify?



---

## What This Does

For each problem the pipeline:

1. **Generates** N candidate solutions (default 512) using a language model
2. **Scores** each solution with a reward model — without running any tests
3. **Compares** strategies across a range of cost budgets:
   - **Fixed (N_rew, N_ver)**: always generate N_rew samples, verify the top-N_ver by reward
   - **Adaptive (ADAP)**: uses a doubling schedule to draw more samples only when needed
   - **DAP_k**: partitions problems into k difficulty groups, assigns the optimal fixed strategy per group
4. **Evaluates** how well the reward signal predicts correctness (`reward_quality.py`)
5. **Analyzes** how problem difficulty drives compute cost (`difficulty_cost.py`)

The math and coding pipelines are structurally identical — they differ only in how solutions are verified (answer matching vs. test execution).

---

## Models

### Math

| Role | Model |
|---|---|
| Generation | `Qwen/Qwen2.5-Math-7B` |
| Generation | `Qwen/Qwen2.5-14B` |
| Generation | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Reward (PRM) | `Qwen/Qwen2.5-Math-PRM-7B` |

Problems: **HMMT February 2024 + 2025** (~60 competition math problems).

### Coding

| Role | Model |
|---|---|
| Generation | `Qwen/Qwen2.5-Coder-3B` |
| Reward | `LARK-Lab/CodeScaler-8B` |

Problems: **LiveCodeBench** (`bzantium/livecodebench`, `release_v2`).

---

## Pipeline Overview

```
Math (per model):
  gen.py → reward.py → compare.py (×4) → compare_with_adaptive.py (×4)
         → reward_quality.py → difficulty_cost.py

Coding:
  gen.py → verify.py → reward.py → compare.py (×4) → compare_with_adaptive.py (×4)
         → reward_quality.py → difficulty_cost.py
```

Both pipelines run `compare.py` and `compare_with_adaptive.py` four times each, once per cost ratio (`c_ver ∈ {1, 10, 20, 30}`). The coding pipeline has an extra `verify.py` step that executes generated code against test cases using 64 CPU workers; this step is CPU-only and can take several hours.

---

## Requirements

- Python 3.10+
- At least 1 CUDA GPU (for generation and reward scoring)
- HuggingFace access to the datasets (all public):
  - `MathArena/hmmt_feb_2024`, `MathArena/hmmt_feb_2025`
  - `bzantium/livecodebench`

```bash
pip install -r requirements.txt
```

---

## Running

### Full experiment (math + coding)

```bash
bash experiment.sh
```

### Math pipeline only

```bash
bash math.sh
```

Runs all three math models sequentially. Each model's outputs go into its own subdirectory under `results/math/`.

### Coding pipeline only

```bash
bash coding.sh
```

> **Note**: The verification step (`verify.py`) runs generated code against test cases using 8 CPU workers. For 512 samples across ~400 problems this takes tens of minutes to hours.

### Smoke test (fast, 4 samples per problem)

```bash
N_SAMPLES=4 MATH_MODELS="Qwen/Qwen2.5-Math-7B" bash math.sh
N_SAMPLES=4 bash coding.sh
```

---

## Configuration

All settings are controlled by environment variables — no config files to edit.

| Variable | Default | Description |
|---|---|---|
| `MATH_OUTDIR` | `./results/math` | Root output directory for math results |
| `CODING_OUTDIR` | `./results/coding` | Output directory for coding results |
| `N_SAMPLES` | `512` | Candidate solutions generated per problem |
| `MATH_MODELS` | all 3 | Space-separated list of math model IDs to run |
| `CODING_MODEL` | `Qwen/Qwen2.5-Coder-3B` | Code generation model |
| `VERIFY_WORKERS` | `8` | CPU workers for parallel code verification |

**Examples:**

```bash
# Run only one math model
MATH_MODELS="Qwen/Qwen2.5-Math-7B" bash math.sh

# Custom output location
MATH_OUTDIR=./run1/math CODING_OUTDIR=./run1/coding bash experiment.sh

# Fewer samples for a quick check
N_SAMPLES=32 bash math.sh
```

---

## Resume Support

Each pipeline stage checks whether its output already exists before running. Interrupted runs can be restarted with the same command — completed stages are skipped automatically.

```
[math] skip gen    (exists): results/math/qwen-qwen2.5-math-7b/generations.jsonl
[math] skip reward (exists): results/math/qwen-qwen2.5-math-7b/rewards.jsonl
[math] compare c_ver=1: results/math/qwen-qwen2.5-math-7b
...
```

---

## Output Structure

```
results/
├── math/
│   ├── qwen-qwen2.5-math-7b/
│   │   ├── generations.jsonl                        # Raw model outputs + correctness labels
│   │   ├── rewards.jsonl                            # PRM step scores (r_min, r_mean, r_last, r_prod)
│   │   ├── compare_c_ver_{1,10,20,30}/              # ADAP vs fixed-strategy plots
│   │   ├── compare_with_adaptive_c_ver_{1,10,20,30}/ # DAP_k vs ADAP plots
│   │   ├── reward_quality/                          # Reward-correctness alignment diagnostics
│   │   └── difficulty_cost/                         # Per-problem optimal (N_rew*, N_ver*) analysis
│   ├── qwen-qwen2.5-14b/
│   └── deepseek-ai-deepseek-r1-distill-qwen-7b/
└── coding/
    ├── generations.jsonl                            # Raw code outputs
    ├── verified.jsonl                               # + test execution results (correct, n_tests_passed)
    ├── rewards.jsonl                                # CodeScaler-8B scalar scores (r_score)
    ├── compare_c_ver_{1,10,20,30}/
    ├── compare_with_adaptive_c_ver_{1,10,20,30}/
    ├── reward_quality/
    └── difficulty_cost/
```

---

## Scripts

### `math/gen.py`

Generates N solutions per HMMT problem using vLLM. Uses a few-shot chain-of-thought prompt and extracts `\boxed{}` answers for correctness checking.

```bash
python math/gen.py --model Qwen/Qwen2.5-Math-7B --n-samples 512 --output-dir ./results/math/run1
```

### `math/reward.py`

Scores each solution with `Qwen2.5-Math-PRM-7B`, a process reward model that assigns a probability to each reasoning step. The primary reward key used downstream is `r_last` (final step score).

```bash
python math/reward.py --input ./results/math/run1/generations.jsonl --output ./results/math/run1/rewards.jsonl
```

### `math/compare.py`

Compares the adaptive strategy against all fixed (M, K) baselines across a cost grid. `c_rew` is the cost of generating + scoring one sample; `c_ver` is the cost of verifying one answer. Run four times with `c_ver ∈ {1, 10, 20, 30}` to see how the tradeoff shifts.

```bash
python math/compare.py \
  --generations results/math/run1/generations.jsonl \
  --rewards     results/math/run1/rewards.jsonl \
  --reward-key  r_last \
  --c-rew 1 --c-ver 10 \
  --out-dir     results/math/run1/compare_c_ver_10
```

### `math/reward_quality.py`

Diagnostic plots that answer: *does a higher reward score actually mean the solution is more likely to be correct?* Produces reward distribution histograms, calibration curves, top-K correctness rates, and per-problem AUC scores.

```bash
python math/reward_quality.py \
  --generations results/math/run1/generations.jsonl \
  --rewards     results/math/run1/rewards.jsonl \
  --reward-key  r_last \
  --out-dir     results/math/run1/reward_quality
```

### `math/difficulty_cost.py`

For each problem, estimates the average optimal (M\*, K\*) — how many samples to draw and how many to verify — by simulating 100 random orderings. Produces scatter plots showing how problem difficulty drives compute cost.

```bash
python math/difficulty_cost.py \
  --generations results/math/run1/generations.jsonl \
  --rewards     results/math/run1/rewards.jsonl \
  --reward-key  r_last \
  --c-rew 1 --c-ver 10 \
  --out-dir     results/math/run1/difficulty_cost
```

### `coding/gen.py`

Generates code solutions for LiveCodeBench problems. Problems include their test cases in the output so `verify.py` can run without re-downloading.

```bash
python coding/gen.py --model Qwen/Qwen2.5-Coder-3B --n-samples 512 --output-dir ./results/coding
```

### `coding/verify.py`

Executes each generated code sample against the problem's test cases using a pool of 8 CPU workers. Populates `correct`, `n_tests_passed`, and `exec_time_s` fields on each sample.

> This step is CPU-only but time-intensive. For 512 samples × 400 problems, expect 30 minutes to several hours depending on hardware.

```bash
python coding/verify.py \
  --input   results/coding/generations.jsonl \
  --output  results/coding/verified.jsonl \
  --workers 8
```

### `coding/reward.py`

Scores code samples with `LARK-Lab/CodeScaler-8B`. Produces a single scalar `r_score` per sample using the Qwen3 chat template.

```bash
python coding/reward.py \
  --generations results/coding/verified.jsonl \
  --out         results/coding/rewards.jsonl
```

`coding/compare.py`, `coding/reward_quality.py`, and `coding/difficulty_cost.py` work identically to their math counterparts — just pass `--reward-key r_score`.

### `math/compare_with_adaptive.py` / `coding/compare_with_adaptive.py`

Computes the DAP_k (Difficulty-Adaptive Policy) baselines. For each k from 1 to `G-max`, it finds the optimal contiguous partition of problems (sorted by pass rate) into k groups and assigns each group the cheapest fixed `(N_rew, N_ver)` strategy that maximises success. Produces cost-vs-k plots and a cost/success scatter comparing DAP_k against ADAP.

```bash
python math/compare_with_adaptive.py \
  --generations results/math/run1/generations.jsonl \
  --rewards     results/math/run1/rewards.jsonl \
  --reward-key  r_last \
  --c-rew 1 --c-ver 10 \
  --G-max 10 \
  --out-dir     results/math/run1/compare_with_adaptive_c_ver_10 \
  --task Math \
  --model-name "Qwen2.5-Math-7B" \
  --reward-model-name "Qwen2.5-Math-PRM-7B"
```

---

## Cost Parameters

The `--c-rew` and `--c-ver` arguments in `compare.py` and `difficulty_cost.py` represent the *relative* cost of reward scoring vs. verification.

- **`c_rew=1, c_ver=1`**: verification is free — reward is the bottleneck
- **`c_rew=1, c_ver=10`**: verification costs 10× more than scoring one sample
- **`c_rew=1, c_ver=30`**: verification is very expensive (e.g. LLM-as-judge or human eval)

Running all four values shows how the adaptive strategy's advantage changes as verification becomes more expensive.

---

## Sampling Parameters

All generation scripts use **temperature=0.7, top-p=0.95** by default. These can be overridden:

```bash
python math/gen.py --temperature 0.9 --top-p 0.95 ...
```
