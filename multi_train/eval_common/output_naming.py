from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple


def _clean_parts(path_value: str) -> list[str]:
    normalized = os.path.normpath(str(path_value).strip())
    return [part for part in normalized.split(os.sep) if part and part not in {".", ".."}]


def _sanitize_component(value: str, lowercase: bool = False) -> str:
    component = str(value).strip().strip(os.sep)
    component = component.replace("/", "_").replace("\\", "_").replace(":", "_")
    component = re.sub(r"\s+", "", component)
    if lowercase:
        component = component.lower()
    return component or "unknown"


def _is_checkpoint_like(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    return (
        normalized.startswith("checkpoint-")
        or normalized.startswith("chpoint-")
        or normalized.startswith("ckpt-")
    )


def _strip_model_prefix(exp_name: str, base_model_name: Optional[str]) -> str:
    candidates = [
        base_model_name,
        "Llama3-8b",
        "Llama3.1-8b",
        "llama3-8b",
        "llama3.1-8b",
    ]
    variants: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        value = str(candidate).strip("-_")
        if not value:
            continue
        variants.extend([value, value.replace("-", "_"), value.replace("_", "-")])

    # Keep order while deduplicating.
    seen: set[str] = set()
    deduped_variants = []
    for variant in variants:
        key = variant.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_variants.append(variant)

    if not deduped_variants:
        return exp_name

    pattern = re.compile(
        r"^(?:" + "|".join(re.escape(v) for v in deduped_variants) + r")[-_]+",
        re.IGNORECASE,
    )
    return pattern.sub("", exp_name, count=1)


def normalize_base_model_name(base_model_path: str) -> str:
    parts = _clean_parts(base_model_path)

    if "models" in parts:
        idx = parts.index("models")
        if idx + 1 < len(parts):
            return _sanitize_component(parts[idx + 1], lowercase=True)

    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx > 0:
            return _sanitize_component(parts[idx - 1], lowercase=True)

    if parts and _is_checkpoint_like(parts[-1]):
        if len(parts) >= 2:
            return _sanitize_component(parts[-2], lowercase=True)
        return _sanitize_component(parts[-1], lowercase=True)

    if parts:
        return _sanitize_component(parts[-1], lowercase=True)

    fallback = Path(str(base_model_path)).name
    return _sanitize_component(fallback, lowercase=True)


def _extract_checkpoint_name(path_value: str) -> Optional[str]:
    parts = _clean_parts(path_value)
    if parts and _is_checkpoint_like(parts[-1]):
        return _sanitize_component(parts[-1])
    return None


def normalize_reft_exp_name(
    reft_weights_path: str,
    base_model_name: Optional[str] = None,
) -> Tuple[str, str]:
    path = Path(os.path.normpath(str(reft_weights_path).rstrip("/")))

    if _is_checkpoint_like(path.name):
        ckpt_name = path.name
        exp_name = path.parent.name
    else:
        ckpt_name = path.parent.name
        exp_name = path.parent.parent.name if path.parent.parent != Path("") else path.parent.name

    if exp_name.endswith("_token"):
        exp_name = exp_name[: -len("_token")]

    exp_name = _strip_model_prefix(exp_name, base_model_name)

    return _sanitize_component(exp_name), _sanitize_component(ckpt_name)


def build_model_tag(
    base_model_path: str,
    reft_weights_path: Optional[str] = None,
    lora_weights_path: Optional[str] = None,
) -> str:
    base_model_name = normalize_base_model_name(base_model_path)

    if reft_weights_path:
        exp_name, ckpt_name = normalize_reft_exp_name(reft_weights_path, base_model_name)
        return f"{base_model_name}-{exp_name}-{ckpt_name}"

    if lora_weights_path:
        lora_path = Path(os.path.normpath(str(lora_weights_path).rstrip("/")))
        lora_name = lora_path.name
        if lora_name in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}:
            lora_name = lora_path.parent.name

        lora_suffix = _sanitize_component(lora_name)
        if not re.search(r"(^|[-_])lora($|[-_])", lora_suffix, re.IGNORECASE):
            lora_suffix = f"lora-{lora_suffix}"
        return f"{base_model_name}-{lora_suffix}"

    base_ckpt = _extract_checkpoint_name(base_model_path)
    if base_ckpt:
        return f"{base_model_name}-{base_ckpt}"

    return base_model_name


def build_output_path(
    output_dir: str,
    base_model_path: str,
    reft_weights_path: Optional[str] = None,
    lora_weights_path: Optional[str] = None,
    suffix: str = "",
    ext: str = ".json",
) -> str:
    model_tag = build_model_tag(
        base_model_path=base_model_path,
        reft_weights_path=reft_weights_path,
        lora_weights_path=lora_weights_path,
    )

    normalized_suffix = suffix or ""
    if normalized_suffix and not normalized_suffix.startswith(("-", "_")):
        normalized_suffix = f"-{normalized_suffix}"

    if not ext.startswith("."):
        ext = f".{ext}"

    return str(Path(output_dir) / f"{model_tag}{normalized_suffix}{ext}")


def _extract_checkpoint_from_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None

    raw = str(path_value).strip().rstrip("/\\")
    if not raw:
        return None

    path = Path(os.path.normpath(raw))
    candidates = [path.name]

    if path.name == "intervenable_model":
        candidates.append(path.parent.name)

    candidates.append(path.parent.name)

    for candidate in candidates:
        if candidate and _is_checkpoint_like(candidate):
            return _sanitize_component(candidate)

    return None


def extract_checkpoint_label(
    base_model_path: Optional[str] = None,
    reft_weights_path: Optional[str] = None,
    lora_weights_path: Optional[str] = None,
) -> str:
    for path_value in (reft_weights_path, lora_weights_path, base_model_path):
        checkpoint = _extract_checkpoint_from_path(path_value)
        if checkpoint:
            return checkpoint
    return ""


def resolve_output_prefix(
    prefix: Optional[str],
    base_model_path: Optional[str],
    mode_label: str = "merge",
) -> str:
    if prefix and str(prefix).strip():
        return _sanitize_component(prefix)

    base_name = normalize_base_model_name(base_model_path or "") if base_model_path else "unknown_model"
    mode = _sanitize_component(mode_label or "merge", lowercase=True)
    return f"{base_name}-{mode}"


def build_results_prefix(prefix: str, checkpoint_label: Optional[str] = None) -> str:
    parts = [_sanitize_component(prefix)]
    if checkpoint_label:
        parts.append(_sanitize_component(checkpoint_label))
    return "-".join(parts)


def build_results_csv_path(
    output_dir: str,
    prefix: str,
    task_name: str,
    checkpoint_label: Optional[str] = None,
) -> str:
    stem_parts = [build_results_prefix(prefix, checkpoint_label), _sanitize_component(task_name)]
    stem = "-".join(stem_parts)
    return str(Path(output_dir) / f"{stem}.csv")


def build_single_summary_path(
    summary_dir: str,
    prefix: str,
    task_name: str,
    run_ts: str,
    checkpoint_label: Optional[str] = None,
) -> str:
    stem_parts = [build_results_prefix(prefix, checkpoint_label), _sanitize_component(task_name)]
    stem = "-".join(stem_parts)
    ts = _sanitize_component(run_ts)
    return str(Path(summary_dir) / f"{stem}-summary_{ts}.json")


def build_merge_summary_path(
    summary_dir: str,
    prefix: str,
    run_ts: str,
) -> str:
    safe_prefix = _sanitize_component(prefix)
    ts = _sanitize_component(run_ts)
    return str(Path(summary_dir) / f"{safe_prefix}_merge_summary_{ts}.json")


__all__ = [
    "normalize_base_model_name",
    "normalize_reft_exp_name",
    "build_model_tag",
    "build_output_path",
    "extract_checkpoint_label",
    "resolve_output_prefix",
    "build_results_prefix",
    "build_results_csv_path",
    "build_single_summary_path",
    "build_merge_summary_path",
]
