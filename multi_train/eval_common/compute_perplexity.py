#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser("Compute continuation perplexity from eval outputs.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a CSV or JSON evaluation result file.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="output",
        help="Deprecated alias for `response_column`. Defaults to `output`.",
    )
    parser.add_argument(
        "--response_column",
        type=str,
        default=None,
        help="Response/output column to score. Defaults to `text_column`.",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Prompt/context column used for conditional perplexity. Defaults to `prompt`.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional suffix appended after the prompt before the response.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2-xl",
        help="Language model used for perplexity computation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for perplexity computation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length passed to the PPL model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for perplexity model.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the aggregated perplexity summary.",
    )
    parser.add_argument(
        "--strip_prompt_echo",
        action="store_true",
        help="Strip prompt/template echo when the response starts by copying the prompt.",
    )
    parser.add_argument(
        "--prompt_echo_min_chars",
        type=int,
        default=32,
        help="Minimum prompt length required before prompt-echo stripping is attempted.",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("JSON input must be a list of records.")
    raise ValueError(f"Unsupported input suffix: {input_path.suffix}")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _candidate_prompt_prefixes(prompt: str) -> List[str]:
    prompt = prompt.strip()
    if not prompt:
        return []

    variants = [
        prompt,
        f"### Input:\n{prompt}\n\n### Response:\n",
        f"### Input:\n{prompt}\n### Response:\n",
        f"Input:\n{prompt}\n\nResponse:\n",
        f"Input:\n{prompt}\nResponse:\n",
        f"User: {prompt}\nAssistant:",
        f"User:\n{prompt}\nAssistant:",
        f"Prompt: {prompt}\nResponse:",
        f"Question: {prompt}\nAnswer:",
    ]
    # Longer prefixes should be matched first.
    return sorted(set(variants), key=len, reverse=True)


def strip_prompt_echo(response: str, prompt: str, min_prompt_chars: int) -> Tuple[str, bool]:
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or len(prompt) < min_prompt_chars or not response:
        return response, False

    candidates = _candidate_prompt_prefixes(prompt)
    for candidate in candidates:
        if response.startswith(candidate):
            cleaned = response[len(candidate) :].lstrip()
            return (cleaned or response), True

    normalized_response = _normalize_ws(response)
    normalized_prompt = _normalize_ws(prompt)
    if (
        normalized_prompt
        and normalized_response.startswith(normalized_prompt)
        and len(normalized_prompt) >= min_prompt_chars
    ):
        prefix_len = len(normalized_prompt)
        suffix = normalized_response[prefix_len:].lstrip(" \n\r\t:,-")
        return (suffix or response), True

    return response, False


def normalize_texts(frame: pd.DataFrame, response_column: str, max_samples: int | None) -> List[str]:
    if response_column not in frame.columns:
        raise ValueError(f"Column `{response_column}` not found. Available columns: {list(frame.columns)}")
    texts = frame[response_column].fillna("").astype(str).tolist()
    texts = [text.strip() for text in texts if text and str(text).strip()]
    if max_samples is not None:
        texts = texts[:max_samples]
    if not texts:
        raise ValueError("No non-empty texts found for perplexity computation.")
    return texts


def normalize_prompt_response_pairs(
    frame: pd.DataFrame,
    prompt_column: str,
    response_column: str,
    prompt_suffix: str,
    max_samples: int | None,
    strip_prompt_echo_enabled: bool,
    prompt_echo_min_chars: int,
) -> Tuple[List[Tuple[str, str]], int]:
    if response_column not in frame.columns:
        raise ValueError(f"Column `{response_column}` not found. Available columns: {list(frame.columns)}")
    if prompt_column not in frame.columns:
        return [], 0

    prompts = frame[prompt_column].fillna("").astype(str).tolist()
    responses = frame[response_column].fillna("").astype(str).tolist()

    pairs: List[Tuple[str, str]] = []
    stripped_count = 0
    for prompt, response in zip(prompts, responses):
        response = response.strip()
        if not response:
            continue
        prompt = prompt if prompt else ""
        if strip_prompt_echo_enabled:
            cleaned_response, was_stripped = strip_prompt_echo(response, prompt, prompt_echo_min_chars)
            if was_stripped:
                stripped_count += 1
            response = cleaned_response
        pairs.append((prompt + prompt_suffix, response))

    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs, stripped_count


T = TypeVar("T")


def batched(items: List[T], batch_size: int) -> Iterable[List[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    frame = load_records(input_path)
    response_column = args.response_column or args.text_column
    prompt_response_pairs = normalize_prompt_response_pairs(
        frame,
        args.prompt_column,
        response_column,
        args.prompt_suffix,
        args.max_samples,
        args.strip_prompt_echo,
        args.prompt_echo_min_chars,
    )
    use_conditional = len(prompt_response_pairs[0]) > 0
    if use_conditional:
        texts = [response for _, response in prompt_response_pairs[0]]
    else:
        texts = normalize_texts(frame, response_column, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    per_sample_nll: List[float] = []
    per_sample_tokens: List[int] = []
    total_batches = math.ceil(len(texts) / args.batch_size)
    skipped_samples = 0

    if use_conditional:
        prompt_response_pairs, stripped_prompt_echo_count = prompt_response_pairs
    else:
        stripped_prompt_echo_count = 0

    progress = tqdm(
        batched(prompt_response_pairs if use_conditional else texts, args.batch_size),
        total=total_batches,
        desc="Computing conditional perplexity" if use_conditional else "Computing perplexity",
        unit="batch",
    )

    for batch in progress:
        if use_conditional:
            valid_pairs: List[Tuple[str, str]] = []
            prefix_lens: List[int] = []
            full_texts: List[str] = []

            for prompt, response in batch:
                prefix_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_length,
                ).input_ids
                full_text = prompt + response
                full_ids = tokenizer(
                    full_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_length,
                ).input_ids

                prefix_len = int(prefix_ids.shape[1])
                full_len = int(full_ids.shape[1])
                if prefix_len >= full_len:
                    skipped_samples += 1
                    continue

                valid_pairs.append((prompt, response))
                prefix_lens.append(prefix_len)
                full_texts.append(full_text)

            if not valid_pairs:
                continue

            enc = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            for row_idx, prefix_len in enumerate(prefix_lens):
                labels[row_idx, :prefix_len] = -100
        else:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels != -100
            safe_labels = shift_labels.masked_fill(~valid_mask, 0)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_loss = loss_fct(logits.reshape(-1, logits.size(-1)), safe_labels.reshape(-1))
            token_loss = token_loss.view(safe_labels.size())
            token_loss = token_loss * valid_mask

            sample_nll = token_loss.sum(dim=1)
            sample_tokens = valid_mask.sum(dim=1)

        per_sample_nll.extend(sample_nll.detach().cpu().tolist())
        per_sample_tokens.extend(sample_tokens.detach().cpu().tolist())
        total_nll += float(sample_nll.sum().item())
        total_tokens += int(sample_tokens.sum().item())
        if total_tokens > 0:
            progress.set_postfix(
                tokens=total_tokens,
                running_ppl=f"{math.exp(total_nll / total_tokens):.2f}",
            )

    if total_tokens <= 0:
        raise ValueError("No valid tokens found for perplexity computation.")

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)

    sample_ppls = [
        math.exp(nll / max(tokens, 1))
        for nll, tokens in zip(per_sample_nll, per_sample_tokens)
        if tokens > 0
    ]

    summary = {
        "input_path": str(input_path),
        "mode": "conditional" if use_conditional else "unconditional",
        "prompt_column": args.prompt_column if use_conditional else None,
        "response_column": response_column,
        "model_name_or_path": args.model_name_or_path,
        "num_texts": len(texts),
        "skipped_samples": skipped_samples,
        "strip_prompt_echo": bool(args.strip_prompt_echo),
        "prompt_echo_min_chars": int(args.prompt_echo_min_chars),
        "num_prompt_echo_stripped": int(stripped_prompt_echo_count),
        "total_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "sample_perplexity_mean": float(sum(sample_ppls) / len(sample_ppls)),
        "sample_perplexity_median": float(pd.Series(sample_ppls).median()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
