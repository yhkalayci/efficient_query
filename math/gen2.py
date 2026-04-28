#!/usr/bin/env python3
import argparse
import collections
import dataclasses
import functools
import io
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple


def ensure_package(pkg: str, import_name: Optional[str] = None) -> None:
    import importlib
    import subprocess
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# Lightweight auto-installs for a one-file setup.
ensure_package("pypdf")
ensure_package("sympy")
ensure_package("tqdm")
ensure_package("vllm")

from pypdf import PdfReader
from sympy import E, I, Rational, pi, simplify, sympify
from sympy.parsing.latex import parse_latex
from tqdm import tqdm


ARCHIVES = {
    "nov2024": {
        "name": "November 2024",
        "url": "https://www.hmmt.org/www/archive/281",
        "rounds": {
            "general": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/gen/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/gen/solutions.pdf",
                "kind": "individual",
            },
            "theme": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/theme/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/theme/solutions.pdf",
                "kind": "individual",
            },
            "team": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/team/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/team/solutions.pdf",
                "kind": "team",
            },
            "guts": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/guts/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/nov/guts/solutions.pdf",
                "kind": "guts",
            },
        },
    },
    "feb2024": {
        "name": "February 2024",
        "url": "https://www.hmmt.org/www/archive/272",
        "rounds": {
            "algebra_number_theory": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/algnum/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/algnum/solutions.pdf",
                "kind": "individual",
            },
            "combinatorics": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/combo/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/combo/solutions.pdf",
                "kind": "individual",
            },
            "geometry": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/geom/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/geom/solutions.pdf",
                "kind": "individual",
            },
            "team": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/team/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/team/solutions.pdf",
                "kind": "team",
            },
            "guts": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/guts/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2024/feb/guts/solutions.pdf",
                "kind": "guts",
            },
        },
    },
    "feb2025": {
        "name": "February 2025",
        "url": "https://www.hmmt.org/www/archive/282",
        "rounds": {
            "algebra_number_theory": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/algnum/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/algnum/solutions.pdf",
                "kind": "individual",
            },
            "combinatorics": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/combo/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/combo/solutions.pdf",
                "kind": "individual",
            },
            "geometry": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/geom/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/geom/solutions.pdf",
                "kind": "individual",
            },
            "team": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/team/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/team/solutions.pdf",
                "kind": "team",
            },
            "guts": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/guts/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/feb/guts/solutions.pdf",
                "kind": "guts",
            },
        },
    },
    "nov2025": {
        "name": "November 2025",
        "url": "https://www.hmmt.org/www/archive/291",
        "rounds": {
            "general": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/gen/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/gen/solutions.pdf",
                "kind": "individual",
            },
            "theme": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/theme/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/theme/solutions.pdf",
                "kind": "individual",
            },
            "team": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/team/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/team/solutions.pdf",
                "kind": "team",
            },
            "guts": {
                "problems": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/guts/problems.pdf",
                "solutions": "https://hmmt-archive.s3.amazonaws.com/tournaments/2025/nov/guts/solutions.pdf",
                "kind": "guts",
            },
        },
    },
}


def download(url: str, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    path.write_bytes(data)


def pdf_text(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    text = "\n\n".join(pages)
    text = text.replace("\r", "")
    text = re.sub(r"\u00a0", " ", text)
    return text


@dataclasses.dataclass
class Question:
    qid: str
    tournament: str
    round_name: str
    kind: str
    number: int
    prompt: str
    gold_raw: str
    gold_canonical: Optional[Tuple[str, ...]]


def clean_ws(s: str) -> str:
    s = s.replace("\u2212", "-")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = s.replace("\t", " ")
    s = re.sub(r" +", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_problem_blocks(text: str) -> List[Tuple[int, str]]:
    text = clean_ws(text)
    pattern = re.compile(r"(?:^|\n)(\d{1,2})\.\s", re.MULTILINE)
    matches = list(pattern.finditer(text))
    blocks = []
    for idx, m in enumerate(matches):
        qnum = int(m.group(1))
        start = m.start(1)
        end = matches[idx + 1].start(1) if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        block = re.sub(rf"^{qnum}\.\s*", "", block)
        blocks.append((qnum, clean_ws(block)))
    return blocks


def split_solution_answers(text: str) -> Dict[int, str]:
    text = clean_ws(text)
    pattern = re.compile(r"(?:^|\n)(\d{1,2})\.\s", re.MULTILINE)
    matches = list(pattern.finditer(text))
    answers = {}
    for idx, m in enumerate(matches):
        qnum = int(m.group(1))
        start = m.start(1)
        end = matches[idx + 1].start(1) if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        am = re.search(r"Answer:\s*(.*?)(?=\bSolution:|$)", block, flags=re.DOTALL | re.IGNORECASE)
        if am:
            ans = clean_ws(am.group(1))
            ans = re.sub(r"\s+", " ", ans)
            answers[qnum] = ans
    return answers


def maybe_parse_latex(expr: str):
    try:
        return parse_latex(expr)
    except Exception:
        return None


def normalize_expr_text(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("^\circ", "")
    s = s.replace("◦", "")
    s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("^", "**")
    s = s.replace("sqrt", "sqrt")
    s = s.replace("π", "pi")
    s = s.replace("∞", "oo")
    s = re.sub(r"(?<![A-Za-z])ln(?![A-Za-z])", "log", s)
    s = re.sub(r"(?<![A-Za-z])(sin|cos|tan|sec|csc|cot)(?![A-Za-z])", r"\1", s)
    s = re.sub(r"(?<![A-Za-z])(pi|e)(?![A-Za-z])", r"\1", s)
    s = s.replace(" ", "")
    s = re.sub(r"(?<=\d)(?=pi|sqrt|[A-Za-z(])", "*", s)
    s = re.sub(r"(?<=[)a-zA-Z])(?=\d)", "*", s)
    s = re.sub(r"(?<=[)a-zA-Z])(?=pi|sqrt|[A-Za-z(])", "*", s)
    s = s.replace("..", ".")
    return s


def sympify_candidate(s: str):
    candidates = [s]
    latex_obj = maybe_parse_latex(s)
    if latex_obj is not None:
        return simplify(latex_obj)
    ns = normalize_expr_text(s)
    candidates.append(ns)
    for cand in candidates:
        try:
            return simplify(sympify(cand, locals={"pi": pi, "e": E, "i": I}))
        except Exception:
            continue
    return None


def split_answer_items(s: str) -> List[str]:
    s = clean_ws(s)
    s = s.strip(". ")
    s = re.sub(r"^Answer:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^The answer is\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^The answers are\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^Final answer:?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\\boxed\{", "", s)
    if s.endswith("}") and s.count("{") < s.count("}") + 2:
        pass
    parts = re.split(r"\s*(?:,|\bor\b|\band\b)\s*", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [s]


def canonicalize_answer(s: str) -> Optional[Tuple[str, ...]]:
    items = split_answer_items(s)
    out = []
    for item in items:
        obj = sympify_candidate(item)
        if obj is None:
            item_norm = normalize_expr_text(item)
            if item_norm:
                out.append(item_norm)
            continue
        out.append(str(simplify(obj)))
    if not out:
        return None
    return tuple(sorted(out))


def extract_last_boxed(text: str) -> Optional[str]:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def extract_final_answer(text: str) -> str:
    text = clean_ws(text)
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    final_markers = [
        r"Final answer\s*[:\-]\s*(.+)$",
        r"Answer\s*[:\-]\s*(.+)$",
        r"Therefore\s*,?\s*the answer is\s*(.+)$",
        r"Thus\s*,?\s*the answer is\s*(.+)$",
    ]
    for pat in final_markers:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

    numberish = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?(?:\s*sqrt\(\d+\))?", text)
    if numberish:
        return numberish[-1].strip()
    last_line = text.strip().splitlines()[-1]
    return last_line.strip()


def equivalent_answers(pred: str, gold_canonical: Optional[Tuple[str, ...]]) -> bool:
    if gold_canonical is None:
        return False
    pred_canonical = canonicalize_answer(pred)
    return pred_canonical == gold_canonical


def build_prompt(problem_text: str) -> str:
    system = (
        "You are a competitive math solver. Solve the problem carefully. "
        "Please reason step by step, and put your final answer within \\boxed{} . "
        "If there are multiple possible values, put all of them inside one \\boxed{} separated by commas."
    )
    user = f"Problem:\n{problem_text}\n"
    return user, system


def load_questions(data_dir: pathlib.Path, include_kinds: Sequence[str]) -> List[Question]:
    questions: List[Question] = []
    pdf_dir = data_dir / "pdfs"

    for tournament_key, tournament_info in ARCHIVES.items():
        for round_name, round_info in tournament_info["rounds"].items():
            if round_info["kind"] not in include_kinds:
                continue
            prob_path = pdf_dir / tournament_key / f"{round_name}_problems.pdf"
            sol_path = pdf_dir / tournament_key / f"{round_name}_solutions.pdf"
            download(round_info["problems"], prob_path)
            download(round_info["solutions"], sol_path)

            problems_text = pdf_text(prob_path)
            solutions_text = pdf_text(sol_path)
            problem_blocks = dict(split_problem_blocks(problems_text))
            answer_blocks = split_solution_answers(solutions_text)

            common = sorted(set(problem_blocks).intersection(answer_blocks))
            if not common:
                print(f"Warning: no matched problems for {tournament_key}/{round_name}")
                continue

            for qnum in common:
                qtext = problem_blocks[qnum]
                gold = answer_blocks[qnum]
                questions.append(
                    Question(
                        qid=f"{tournament_key}:{round_name}:{qnum}",
                        tournament=tournament_key,
                        round_name=round_name,
                        kind=round_info["kind"],
                        number=qnum,
                        prompt=qtext,
                        gold_raw=gold,
                        gold_canonical=canonicalize_answer(gold),
                    )
                )

    return questions


def get_llm(args):
    import torch
    from vllm import LLM

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs found. This script is meant for GPU inference with vLLM.")

    dtype = args.dtype
    if dtype == "auto":
        bf16_ok = torch.cuda.is_bf16_supported()
        dtype = "bfloat16" if bf16_ok else "float16"

    max_num_seqs = args.max_num_seqs
    if max_num_seqs <= 0:
        # Conservative automatic cap. vLLM may still rebatch internally.
        max_num_seqs = min(256, max(32, 8 * num_gpus))

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
        max_model_len=args.max_model_len,
        swap_space=args.swap_space,
        seed=args.seed,
    )
    return llm, dtype, num_gpus


def generate_samples(llm, questions: Sequence[Question], args, out_dir: pathlib.Path) -> Dict[str, List[dict]]:
    from vllm import SamplingParams

    prompts = []
    meta = []
    for q in questions:
        user_prompt, system_prompt = build_prompt(q.prompt)
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for sample_idx in range(args.samples_per_question):
            prompts.append(prompt_messages)
            meta.append((q.qid, sample_idx))

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_generations.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
        stop=args.stop_strings,
        skip_special_tokens=True,
    )

    all_results: Dict[str, List[dict]] = collections.defaultdict(list)
    chunk = args.request_batch_size
    total_chunks = math.ceil(len(prompts) / chunk)

    for chunk_idx in tqdm(range(total_chunks), desc="Generating"):
        lo = chunk_idx * chunk
        hi = min(len(prompts), (chunk_idx + 1) * chunk)
        prompt_batch = prompts[lo:hi]
        meta_batch = meta[lo:hi]
        outputs = llm.chat(prompt_batch, sampling_params=sampling_params, use_tqdm=False)

        with raw_path.open("a", encoding="utf-8") as f:
            for output, (qid, sample_idx) in zip(outputs, meta_batch):
                text = output.outputs[0].text
                rec = {
                    "qid": qid,
                    "sample_idx": sample_idx,
                    "text": text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                all_results[qid].append(rec)

    for qid in all_results:
        all_results[qid].sort(key=lambda x: x["sample_idx"])
    return all_results


def evaluate(questions: Sequence[Question], generations: Dict[str, List[dict]], args, out_dir: pathlib.Path) -> dict:
    question_map = {q.qid: q for q in questions}
    ks = sorted(set([1] + [int(k) for k in args.pass_k]))

    scored_questions = []
    agg = {k: [] for k in ks}

    detailed_path = out_dir / "scored_samples.jsonl"
    if detailed_path.exists():
        detailed_path.unlink()

    with detailed_path.open("a", encoding="utf-8") as f:
        for qid, samples in generations.items():
            q = question_map[qid]
            sample_records = []
            for rec in samples:
                pred = extract_final_answer(rec["text"])
                ok = equivalent_answers(pred, q.gold_canonical)
                item = {
                    "qid": qid,
                    "sample_idx": rec["sample_idx"],
                    "gold_raw": q.gold_raw,
                    "gold_canonical": q.gold_canonical,
                    "predicted_final": pred,
                    "correct": ok,
                    "text": rec["text"],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                sample_records.append(item)

            correctness = [x["correct"] for x in sorted(sample_records, key=lambda x: x["sample_idx"])]
            qmetrics = {}
            for k in ks:
                qmetrics[f"pass@{k}"] = any(correctness[: min(k, len(correctness))])
                agg[k].append(float(qmetrics[f"pass@{k}"]))

            scored_questions.append(
                {
                    "qid": qid,
                    "tournament": q.tournament,
                    "round_name": q.round_name,
                    "kind": q.kind,
                    "number": q.number,
                    "gold_raw": q.gold_raw,
                    "gold_canonical": q.gold_canonical,
                    "num_samples": len(sample_records),
                    "num_correct": sum(correctness),
                    **qmetrics,
                }
            )

    summary = {
        "num_questions": len(scored_questions),
        "samples_per_question": args.samples_per_question,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "model": args.model,
        "pass_at_k": {f"pass@{k}": sum(agg[k]) / max(1, len(agg[k])) for k in ks},
        "questions": scored_questions,
    }

    by_round = collections.defaultdict(list)
    by_kind = collections.defaultdict(list)
    for row in scored_questions:
        by_round[f"{row['tournament']}::{row['round_name']}"] .append(row)
        by_kind[row["kind"]].append(row)

    summary["by_round"] = {}
    for key, rows in by_round.items():
        summary["by_round"][key] = {
            f"pass@{k}": sum(float(r[f"pass@{k}"]) for r in rows) / len(rows) for k in ks
        }

    summary["by_kind"] = {}
    for key, rows in by_kind.items():
        summary["by_kind"][key] = {
            f"pass@{k}": sum(float(r[f"pass@{k}"]) for r in rows) / len(rows) for k in ks
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Math-7B on HMMT 2024+2025 with many stochastic samples and compute pass@k.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--data-dir", default="./hmmt_qwen_data")
    parser.add_argument("--out-dir", default="./hmmt_qwen_runs/run1")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--swap-space", type=float, default=16.0)
    parser.add_argument("--max-num-seqs", type=int, default=0, help="0 = auto")
    parser.add_argument("--request-batch-size", type=int, default=128, help="How many independent sampled requests to submit to vLLM at once.")
    parser.add_argument("--samples-per-question", type=int, default=512)
    parser.add_argument("--pass-k", nargs="*", default=[1, 8, 16, 32, 64, 100, 256, 512])
    parser.add_argument("--include-kinds", nargs="*", default=["individual", "team", "guts"], choices=["individual", "team", "guts"])
    parser.add_argument("--stop-strings", nargs="*", default=[])
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(data_dir, include_kinds=args.include_kinds)
    questions = sorted(questions, key=lambda q: q.qid)

    if args.limit_questions > 0:
        questions = questions[: args.limit_questions]

    if not questions:
        raise RuntimeError("No questions loaded. Check PDF links or parsing.")

    print(f"Loaded {len(questions)} questions.")
    print(f"Kinds included: {args.include_kinds}")
    print(f"Samples per question: {args.samples_per_question}")

    llm, dtype, num_gpus = get_llm(args)
    print(f"Using {num_gpus} GPUs with dtype={dtype} and model={args.model}")

    generations = generate_samples(llm, questions, args, out_dir)
    summary = evaluate(questions, generations, args, out_dir)

    print(json.dumps({
        "num_questions": summary["num_questions"],
        "samples_per_question": summary["samples_per_question"],
        "pass_at_k": summary["pass_at_k"],
    }, indent=2))


if __name__ == "__main__":
    main()
