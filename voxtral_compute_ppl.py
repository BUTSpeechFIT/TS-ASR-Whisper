"""
Evaluate perplexity of train_by_utterance.txt and train_by_speaker.txt
using mistralai/Ministral-3-3B-Base-2512.

Usage:
    python eval_perplexity.py \
        --utterance train_by_utterance.txt \
        --speaker   train_by_speaker.txt \
        [--batch_size 4] \
        [--max_length 2048] \
        [--stride 512] \
        [--device cuda]
"""

import argparse
import re
import math
import torch
from pathlib import Path

from torch import dtype
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Collapse multiple spaces / tabs into a single space and strip."""
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_and_clean(path: str) -> list[str]:
    """Load a txt file (one session per line), clean whitespace, drop empty lines."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cleaned = [clean_text(l) for l in lines]
    cleaned = [l for l in cleaned if l]
    return cleaned


def save_cleaned(lines: list[str], path: str):
    """Write cleaned lines back to file."""
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved cleaned file → {path}")


# ---------------------------------------------------------------------------
# Perplexity with sliding window
# ---------------------------------------------------------------------------

def compute_perplexity(
    lines: list[str],
    model,
    tokenizer,
    max_length: int,
    stride: int,
    batch_size: int,
    device: str,
) -> dict:
    """
    Compute perplexity over a list of lines using a sliding-window approach
    so that sequences longer than max_length are handled correctly.

    Returns a dict with per-line ppls and the overall corpus ppl.
    """
    model.eval()
    loss_fn = CrossEntropyLoss(reduction="sum")

    total_nll = 0.0
    total_tokens = 0
    per_line_ppls = []

    for i in tqdm(range(0, len(lines), batch_size), desc="  batches"):
        batch_lines = lines[i : i + batch_size]

        for line in batch_lines:
            enc = tokenizer(line, return_tensors="pt", truncation=False)
            input_ids = enc["input_ids"].to(device)  # (1, seq_len)
            seq_len = input_ids.size(1)

            line_nll = 0.0
            line_tokens = 0
            prev_end = 0

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                # The "new" tokens in this window (avoid double-counting overlap)
                target_len = end - prev_end

                chunk = input_ids[:, begin:end]
                with torch.no_grad():
                    logits = model(chunk).logits  # (1, chunk_len, vocab)

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk[:, 1:].contiguous()

                # Only score the new (non-overlapping) portion
                new_logits = shift_logits[:, -target_len:, :]
                new_labels = shift_labels[:, -target_len:]

                nll = loss_fn(
                    new_logits.view(-1, new_logits.size(-1)),
                    new_labels.view(-1),
                )
                line_nll += nll.item()
                line_tokens += target_len

                prev_end = end
                if end == seq_len:
                    break

            if line_tokens > 0:
                line_ppl = math.exp(line_nll / line_tokens)
                per_line_ppls.append(line_ppl)
                total_nll += line_nll
                total_tokens += line_tokens

    corpus_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

    return {
        "corpus_ppl": corpus_ppl,
        "per_line_ppls": per_line_ppls,
        "mean_line_ppl": sum(per_line_ppls) / len(per_line_ppls) if per_line_ppls else float("inf"),
        "num_lines": len(per_line_ppls),
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance", default="train_by_utterance.txt")
    parser.add_argument("--speaker",   default="train_by_speaker.txt")
    parser.add_argument("--model",     default="mistralai/Ministral-3-3B-Base-2512")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of lines processed per outer loop iteration")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max token window length (model context)")
    parser.add_argument("--stride",     type=int, default=512,
                        help="Sliding-window stride for long sequences")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # --- Load & clean files ---
    print("\n[1/4] Loading and cleaning files...")
    utterance_lines = load_and_clean(args.utterance)
    speaker_lines   = load_and_clean(args.speaker)

    save_cleaned(utterance_lines, args.utterance)
    save_cleaned(speaker_lines,   args.speaker)

    print(f"  Utterance file: {len(utterance_lines)} sessions")
    print(f"  Speaker file:   {len(speaker_lines)} sessions")

    # --- Load model ---
    print(f"\n[2/4] Loading model {args.model} ...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        offload_folder="./",
        dtype=torch.bfloat16,
        quantization_config=None,

    )
    tokenizer = MistralCommonBackend.from_pretrained(args.model)
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
    #     device_map=args.device,
    # )
    model.eval()

    # --- Evaluate ---
    print("\n[3/4] Evaluating train_by_utterance.txt ...")
    utt_results = compute_perplexity(
        utterance_lines, model, tokenizer,
        args.max_length, args.stride, args.batch_size, args.device,
    )

    print("\n[4/4] Evaluating train_by_speaker.txt ...")
    spk_results = compute_perplexity(
        speaker_lines, model, tokenizer,
        args.max_length, args.stride, args.batch_size, args.device,
    )

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, res in [("train_by_utterance", utt_results), ("train_by_speaker", spk_results)]:
        print(f"\n{label}:")
        print(f"  Corpus perplexity : {res['corpus_ppl']:.2f}")
        print(f"  Mean line ppl     : {res['mean_line_ppl']:.2f}")
        print(f"  Lines evaluated   : {res['num_lines']}")
        print(f"  Total tokens      : {res['total_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    main()