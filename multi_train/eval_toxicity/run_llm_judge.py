#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Run LLM-as-judge on prepared toxicity audit JSONL.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--api_model", type=str, default="deepseek-chat")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key_env", type=str, default="DEEPSEEK_API")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=5.0)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def record_uid(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("audit_split", "")),
        str(record.get("judge_type", "")),
        str(record.get("panel_id", "")),
    ]
    if record.get("judge_type") == "absolute":
        parts.append(str(record.get("method", "")))
    else:
        parts.append(str(record.get("comparison", "")))
    return "|".join(parts)


def extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        if "choices" in response:
            try:
                return str(response["choices"][0]["message"]["content"]).strip()
            except Exception:
                pass
        if "content" in response:
            return str(response["content"]).strip()
    try:
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise ValueError(f"Cannot extract response text: {exc}") from exc


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response.")

    depth = 0
    in_string = False
    escape = False
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError("No complete JSON object found in model response.")

    payload = json.loads(text[start:end])
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON payload is not an object.")
    return payload


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否"}:
        return False
    return None


def normalize_choice(value: Any, allowed: set[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in allowed:
        return text
    lowered = text.lower()
    for item in allowed:
        if lowered == item.lower():
            return item
    return None


def normalize_int_score(value: Any, min_value: int, max_value: int) -> int | None:
    try:
        number = int(value)
    except Exception:
        return None
    if min_value <= number <= max_value:
        return number
    return None


def fallback_parse_payload(judge_type: str, text: str) -> dict[str, Any]:
    lowered = text.lower()

    def extract_bool(field: str) -> bool | None:
        patterns = [
            rf"{field}\s*[:=]\s*(true|false|yes|no|1|0)",
            rf"{field}\s*[：]\s*(true|false|yes|no|1|0|是|否)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                return normalize_bool(match.group(1))
        return None

    def extract_int(field: str) -> int | None:
        patterns = [
            rf"{field}\s*[:=]\s*([0-3])",
            rf"{field}\s*[：]\s*([0-3])",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                return normalize_int_score(match.group(1), 0, 3)
        return None

    def extract_choice(field: str, allowed: set[str]) -> str | None:
        patterns = [
            rf"{field}\s*[:=]\s*([A-Za-z]+)",
            rf"{field}\s*[：]\s*([A-Za-z]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                normalized = normalize_choice(match.group(1), allowed)
                if normalized is not None:
                    return normalized
        return None

    if judge_type == "absolute":
        payload = {
            "true_toxicity": extract_int("true_toxicity"),
            "quoted_or_negated_toxicity": extract_bool("quoted_or_negated_toxicity"),
            "contextual_coherence": extract_int("contextual_coherence"),
            "over_refusal": extract_bool("over_refusal"),
            "rationale_brief": text.strip(),
        }
        if payload["true_toxicity"] is None:
            standalone = re.search(r"\b([0-3])\b", text)
            if standalone:
                payload["true_toxicity"] = normalize_int_score(standalone.group(1), 0, 3)
        return payload

    payload = {
        "safer_response": extract_choice("safer_response", {"A", "B", "Tie"}),
        "better_continuation": extract_choice("better_continuation", {"A", "B", "Tie"}),
        "more_generic_refusal": extract_choice("more_generic_refusal", {"A", "B", "Neither"}),
        "overall_preference": extract_choice("overall_preference", {"A", "B", "Tie"}),
        "rationale_brief": text.strip(),
    }
    if payload["overall_preference"] is None:
        first_token = re.search(r"\b(A|B|Tie)\b", text, flags=re.IGNORECASE)
        if first_token:
            payload["overall_preference"] = normalize_choice(first_token.group(1), {"A", "B", "Tie"})
    return payload


def normalize_payload(judge_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if judge_type == "absolute":
        return {
            "true_toxicity": normalize_int_score(payload.get("true_toxicity"), 0, 3),
            "quoted_or_negated_toxicity": normalize_bool(payload.get("quoted_or_negated_toxicity")),
            "contextual_coherence": normalize_int_score(payload.get("contextual_coherence"), 0, 3),
            "over_refusal": normalize_bool(payload.get("over_refusal")),
            "rationale_brief": str(payload.get("rationale_brief", "")).strip(),
        }

    if judge_type == "pairwise":
        return {
            "safer_response": normalize_choice(payload.get("safer_response"), {"A", "B", "Tie"}),
            "better_continuation": normalize_choice(payload.get("better_continuation"), {"A", "B", "Tie"}),
            "more_generic_refusal": normalize_choice(payload.get("more_generic_refusal"), {"A", "B", "Neither"}),
            "overall_preference": normalize_choice(payload.get("overall_preference"), {"A", "B", "Tie"}),
            "rationale_brief": str(payload.get("rationale_brief", "")).strip(),
        }

    raise ValueError(f"Unsupported judge_type: {judge_type}")


def is_valid_normalized_payload(judge_type: str, payload: dict[str, Any]) -> bool:
    if judge_type == "absolute":
        required = ["true_toxicity", "quoted_or_negated_toxicity", "contextual_coherence", "over_refusal"]
    else:
        required = ["safer_response", "better_continuation", "more_generic_refusal", "overall_preference"]
    return all(payload.get(key) is not None for key in required)


def create_client(args) -> openai.OpenAI:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Environment variable `{args.api_key_env}` is not set.")
    return openai.OpenAI(
        api_key=api_key,
        base_url=args.api_base if args.api_base else None,
        timeout=args.request_timeout,
        max_retries=0,
    )


_THREAD_LOCAL = threading.local()


def get_thread_client(args) -> openai.OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = create_client(args)
        _THREAD_LOCAL.client = client
    return client


def run_request(client: openai.OpenAI, args, record: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    messages = [
        {"role": "user", "content": record["judge_prompt"]},
    ]

    last_error = None
    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(
                model=args.api_model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
            )
            text = extract_text(response)
            fallback_used = False
            try:
                payload = extract_first_json_object(text)
            except Exception:
                payload = fallback_parse_payload(record["judge_type"], text)
                fallback_used = True
            normalized = normalize_payload(record["judge_type"], payload)
            if not is_valid_normalized_payload(record["judge_type"], normalized):
                raise ValueError(f"Normalized payload missing required fields: {normalized}")
            usage = {}
            if getattr(response, "usage", None) is not None:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            return text, payload, {"normalized": normalized, "usage": usage, "fallback_used": fallback_used}
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))
                continue
            raise RuntimeError(last_error) from exc

    raise RuntimeError(last_error or "Unknown judge request failure.")


def process_record(args, record: dict[str, Any]) -> dict[str, Any]:
    uid = record_uid(record)
    result = {
        **record,
        "record_uid": uid,
        "judge_model": args.api_model,
        "judge_api_base": args.api_base,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }
    client = get_thread_client(args)
    try:
        raw_text, raw_payload, extra = run_request(client, args, record)
        result["status"] = "ok"
        result["raw_response_text"] = raw_text
        result["parsed_response"] = raw_payload
        result["normalized_judgement"] = extra["normalized"]
        result["usage"] = extra["usage"]
        result["fallback_used"] = extra.get("fallback_used", False)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["raw_response_text"] = None
        result["parsed_response"] = None
        result["normalized_judgement"] = None
        result["usage"] = {}
        result["fallback_used"] = False
    return result


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    completed: dict[str, dict[str, Any]] = {}
    if output_path.exists() and not args.overwrite:
        for row in load_jsonl(output_path):
            completed[record_uid(row)] = row
    elif args.overwrite and output_path.exists():
        output_path.unlink()

    pending = [record for record in records if record_uid(record) not in completed]
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    mode = "a" if output_path.exists() and not args.overwrite else "w"
    processed_since_flush = 0
    with output_path.open(mode, encoding="utf-8") as fout:
        progress = tqdm(pending, desc=f"Judging {input_path.name}")
        if args.concurrency == 1:
            for record in progress:
                result = process_record(args, record)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_since_flush += 1
                if processed_since_flush >= args.save_every:
                    fout.flush()
                    processed_since_flush = 0
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(process_record, args, record) for record in pending]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_since_flush += 1
                    if processed_since_flush >= args.save_every:
                        fout.flush()
                        processed_since_flush = 0
                    progress.update(1)

        if processed_since_flush > 0:
            fout.flush()

    print(
        json.dumps(
            {
                "input_jsonl": str(input_path),
                "output_jsonl": str(output_path),
                "total_records": len(records),
                "already_completed": len(completed),
                "newly_processed": len(pending),
                "concurrency": args.concurrency,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
