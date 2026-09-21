from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data import ParsedTrajectory


@dataclass(frozen=True)
class StandardizationStats:
    mean_u: float
    std_u: float
    mean_rho: float
    std_rho: float
    source: str

    def __post_init__(self) -> None:
        values = (self.mean_u, self.std_u, self.mean_rho, self.std_rho)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("standardization statistics must be finite")
        if self.std_u <= 0 or self.std_rho <= 0:
            raise ValueError("standardization standard deviations must be positive")


@dataclass(frozen=True)
class WeightParameters:
    lambda_u: float
    lambda_d: float
    weight_floor: float = 0.05
    epsilon: float = 1e-3

    def __post_init__(self) -> None:
        values = (self.lambda_u, self.lambda_d, self.weight_floor, self.epsilon)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("weight parameters must be finite")
        if self.lambda_u < 0 or self.lambda_d < 0:
            raise ValueError("lambda_u and lambda_d must be non-negative")
        if self.weight_floor <= 0 or self.epsilon <= 0:
            raise ValueError("weight_floor and epsilon must be positive")


def fit_standardization(
    records: Iterable[dict[str, Any]], *, source: str
) -> StandardizationStats:
    values = list(records)
    if not values:
        raise ValueError("cannot fit standardization without raw signal records")
    us = [float(record["U"]) for record in values]
    rhos = [float(record["rho"]) for record in values]

    def moments(items: Sequence[float]) -> tuple[float, float]:
        if not all(math.isfinite(item) for item in items):
            raise ValueError("raw signals contain non-finite values")
        mean = math.fsum(items) / len(items)
        variance = math.fsum((item - mean) ** 2 for item in items) / len(items)
        standard_deviation = math.sqrt(variance)
        if standard_deviation == 0:
            raise ValueError("cannot standardize a constant signal")
        return mean, standard_deviation

    mean_u, std_u = moments(us)
    mean_rho, std_rho = moments(rhos)
    return StandardizationStats(mean_u, std_u, mean_rho, std_rho, source)


def compute_weight(
    uncertainty: float,
    relative_disagreement: float,
    stats: StandardizationStats,
    parameters: WeightParameters,
) -> tuple[float, float, float]:
    """Apply the Phase-2 equation from the CoReD paper exactly once."""

    u_hat = (float(uncertainty) - stats.mean_u) / stats.std_u
    rho_hat = (float(relative_disagreement) - stats.mean_rho) / stats.std_rho
    weight = max(
        parameters.weight_floor,
        1.0
        + parameters.lambda_u * math.tanh(u_hat)
        + parameters.lambda_d * math.tanh(rho_hat),
    )
    return weight, u_hat, rho_hat


def assign_high_weight(weights: Sequence[float], fraction: float = 0.25) -> tuple[bool, ...]:
    if not weights:
        return ()
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not all(math.isfinite(float(weight)) for weight in weights):
        raise ValueError("weights must be finite")
    high_count = max(1, math.ceil(fraction * len(weights)))
    # Python's stable ordering plus the explicit index key makes exact ties deterministic.
    ranked = sorted(range(len(weights)), key=lambda index: (-float(weights[index]), index))
    selected = set(ranked[:high_count])
    return tuple(index in selected for index in range(len(weights)))


def finalize_weight_records(
    raw_records: Sequence[dict[str, Any]],
    stats: StandardizationStats,
    parameters: WeightParameters,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in raw_records:
        grouped.setdefault(str(record["uuid"]), []).append(dict(record))

    output: list[dict[str, Any]] = []
    for uuid in sorted(grouped):
        records = sorted(grouped[uuid], key=lambda item: int(item["step_index"]))
        weighted: list[dict[str, Any]] = []
        for record in records:
            weight, u_hat, rho_hat = compute_weight(
                record["U"], record["rho"], stats, parameters
            )
            record.update(weight=weight, U_hat=u_hat, rho_hat=rho_hat)
            weighted.append(record)
        labels = assign_high_weight([record["weight"] for record in weighted])
        for record, is_high in zip(weighted, labels):
            record["is_high_weight"] = is_high
            record["weight_group"] = "High-W" if is_high else "Low-W"
            output.append(record)
    return output


def _dtype(name: str):
    import torch

    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = name.lower().replace("torch.", "")
    if normalized not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[normalized]


def load_council_model(
    backbone: str,
    adapter_paths: Sequence[str | Path],
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str | None = "flash_attention_2",
):
    """Load one base model and named, frozen PEFT adapters."""

    if len(adapter_paths) < 2:
        raise ValueError("at least two Phase-1 expert adapters are required")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": _dtype(dtype),
        "device_map": device_map,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        base = AutoModelForCausalLM.from_pretrained(backbone, **kwargs)
    except (ImportError, ValueError) as error:
        if attn_implementation != "flash_attention_2":
            raise
        kwargs["attn_implementation"] = "sdpa"
        base = AutoModelForCausalLM.from_pretrained(backbone, **kwargs)

    names = [f"expert_{index}" for index in range(len(adapter_paths))]
    model = PeftModel.from_pretrained(
        base, str(adapter_paths[0]), adapter_name=names[0], is_trainable=False
    )
    for name, path in zip(names[1:], adapter_paths[1:]):
        model.load_adapter(str(path), adapter_name=name, is_trainable=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        backbone, revision=revision, use_fast=True, trust_remote_code=False
    )
    if not tokenizer.is_fast:
        raise RuntimeError("weight extraction requires a fast tokenizer with offsets")
    return model, tokenizer, names


def render_teacher_forcing_text(
    tokenizer: Any,
    trajectory: ParsedTrajectory,
    *,
    enable_thinking: bool | None,
    content_template: str = "{think_content}",
) -> tuple[str, str]:
    """Render only the parsed think trajectory as an assistant turn."""

    if "{think_content}" not in content_template:
        raise ValueError("teacher-forcing content template must contain {think_content}")
    assistant_content = content_template.format(think_content=trajectory.think_content)

    messages = [
        {"role": "user", "content": trajectory.problem},
        {"role": "assistant", "content": assistant_content},
    ]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs), assistant_content


def _step_character_spans(
    rendered: str,
    assistant_content: str,
    trajectory: ParsedTrajectory,
) -> list[tuple[int, int]]:
    content_at = rendered.rfind(assistant_content)
    if content_at < 0:
        raise ValueError("chat template did not preserve the think content verbatim")
    cursor = 0
    spans: list[tuple[int, int]] = []
    for step in trajectory.eligible_steps:
        at = assistant_content.find(step, cursor)
        if at < 0:
            raise ValueError("could not locate an eligible step in teacher-forcing content")
        spans.append((content_at + at, content_at + at + len(step)))
        cursor = at + len(step)
    return spans


def _token_step_map(offsets: Sequence[Sequence[int]], spans: Sequence[tuple[int, int]]) -> list[list[int]]:
    step_tokens: list[list[int]] = []
    for start, end in spans:
        positions = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and token_end > start and token_start < end and index > 0
        ]
        if not positions:
            raise ValueError("an eligible step mapped to zero predicted tokens")
        step_tokens.append(positions)
    return step_tokens


def _decoder_and_head(peft_model: Any) -> tuple[Any, Any]:
    causal_lm = peft_model.get_base_model()
    decoder = getattr(causal_lm, "model", None)
    head = causal_lm.get_output_embeddings()
    if decoder is None or head is None:
        raise RuntimeError("unsupported causal LM structure: decoder/output head not found")
    if getattr(head, "lora_A", None):
        raise RuntimeError(
            "the Phase-1 adapters target the LM output head; the chunked shared-head entropy "
            "path would be invalid, so this checkpoint convention needs an explicit extractor"
        )
    return decoder, head


def extract_council_signals(
    model: Any,
    tokenizer: Any,
    adapter_names: Sequence[str],
    trajectory: ParsedTrajectory,
    *,
    epsilon: float,
    logit_chunk_size: int = 64,
    enable_thinking: bool | None = None,
    content_template: str = "{think_content}",
) -> list[dict[str, float]]:
    """Teacher-force one trajectory and compute full-vocabulary Phase-2 signals.

    The implementation stores only selected final hidden states per expert and
    applies the shared output head in chunks, avoiding materializing an entire
    sequence-by-vocabulary tensor for every expert.
    """

    import torch

    rendered, assistant_content = render_teacher_forcing_text(
        tokenizer,
        trajectory,
        enable_thinking=enable_thinking,
        content_template=content_template,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    step_tokens = _token_step_map(
        offsets, _step_character_spans(rendered, assistant_content, trajectory)
    )
    flattened = [position for positions in step_tokens for position in positions]
    if len(set(flattened)) != len(flattened):
        raise ValueError("step token spans overlap")
    prediction_rows = torch.tensor([position - 1 for position in flattened], dtype=torch.long)

    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    decoder, head = _decoder_and_head(model)
    expert_hidden: list[Any] = []
    model.eval()
    with torch.inference_mode():
        for adapter_name in adapter_names:
            model.set_adapter(adapter_name)
            outputs = decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            selected = outputs.last_hidden_state[0].index_select(
                0, prediction_rows.to(outputs.last_hidden_state.device)
            )
            expert_hidden.append(selected.detach().cpu())
            del outputs, selected

        token_u = torch.empty(len(flattened), dtype=torch.float32)
        token_v = torch.empty(len(flattened), dtype=torch.float32)
        for start in range(0, len(flattened), logit_chunk_size):
            end = min(start + logit_chunk_size, len(flattened))
            sum_probabilities = None
            sum_entropies = torch.zeros(end - start, device=device, dtype=torch.float32)
            for hidden in expert_hidden:
                logits = head(hidden[start:end].to(device)).float()
                log_probabilities = torch.log_softmax(logits, dim=-1)
                probabilities = log_probabilities.exp()
                sum_entropies += -(probabilities * log_probabilities).sum(dim=-1)
                sum_probabilities = (
                    probabilities
                    if sum_probabilities is None
                    else sum_probabilities + probabilities
                )
                del logits, log_probabilities
            mean_entropy = sum_entropies / len(adapter_names)
            mixture = sum_probabilities / len(adapter_names)
            mixture_entropy = -(mixture * mixture.clamp_min(1e-45).log()).sum(dim=-1)
            token_u[start:end] = mean_entropy.cpu()
            # Do not clamp: the paper defines the raw entropy difference. Tiny
            # negative values from floating-point arithmetic remain observable.
            token_v[start:end] = (mixture_entropy - mean_entropy).cpu()

    output: list[dict[str, float]] = []
    cursor = 0
    for positions in step_tokens:
        count = len(positions)
        uncertainty = float(token_u[cursor : cursor + count].mean())
        disagreement = float(token_v[cursor : cursor + count].mean())
        output.append(
            {
                "U": uncertainty,
                "V": disagreement,
                "rho": disagreement / (uncertainty + epsilon),
                "token_count": count,
            }
        )
        cursor += count
    return output
