from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from .cache import CacheProvenanceError, JsonlCache
from .generation import DecodingConfig, GenerationRequest, VLLMGenerator, rollout_seed
from .io import atomic_write_jsonl
from .metrics.answer_gain import answer_gains
from .metrics.removal import removal_damages
from .prompts import (
    answer_gain_user_prompt,
    audit_prompt_sources,
    removal_user_prompt,
    render_chat_prompt,
    require_clean_audit,
)
from .records import WeightedSample
from .reporting import ExclusionReport
from .verifier import MathVerifier, verification_statistics


LOGGER = logging.getLogger(__name__)


def _trajectory_view(sample: WeightedSample) -> Any:
    return SimpleNamespace(
        problem=sample.problem,
        answer=sample.answer,
        excluded_last_step_text=sample.excluded_last_step_text,
        outside_before_think=sample.outside_before_think,
        outside_after_think=sample.outside_after_think,
    )


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _require_matching_decoding(cache: JsonlCache, decoding: DecodingConfig) -> None:
    expected = decoding.__dict__
    for record in cache.values():
        if record.get("decoding") != expected:
            raise CacheProvenanceError(
                f"cached decoding configuration differs from this run in {cache.path}; "
                "use a separate output directory"
            )


def _verified_record(
    output: dict[str, Any],
    *,
    base: dict[str, Any],
    verifier: MathVerifier,
    answer: str,
    decoding: DecodingConfig,
) -> dict[str, Any]:
    verification = verifier(output["generated_text"], answer)
    return {
        **base,
        "generated_text": output["generated_text"],
        "parsed_answer": verification.parsed_answer,
        "parsed_reference": verification.parsed_reference,
        "correct": verification.correct,
        "verification_status": verification.status,
        "verification_error": verification.error,
        "generation_seed": output["generation_seed"],
        "finish_reason": output["finish_reason"],
        "stop_reason": output["stop_reason"],
        "generated_token_count": output["generated_token_count"],
        "decoding": decoding.__dict__,
    }


def _correctness(
    cache: JsonlCache,
    keys: Sequence[tuple[Any, ...]],
    num_rollouts: int,
) -> float | None:
    records = [cache.get(key) for key in keys]
    if len(records) != num_rollouts or any(record is None for record in records):
        return None
    return sum(record["verification_status"] == "correct" for record in records) / num_rollouts


def run_answer_gain(
    samples: Sequence[WeightedSample],
    *,
    tokenizer: Any,
    generator: VLLMGenerator,
    decoding: DecodingConfig,
    num_rollouts: int,
    seed: int,
    batch_size: int,
    results_dir: Path,
    is_qwen3: bool,
    resume: bool,
    save_prompts: bool,
    exclusions: ExclusionReport,
) -> dict[str, Any]:
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    generation_cache = JsonlCache(
        results_dir / "answer_gain_generations.jsonl",
        key=lambda record: (
            str(record["uuid"]),
            int(record["prefix_index"]),
            int(record["rollout_index"]),
        ),
        resume=resume,
    )
    _require_matching_decoding(generation_cache, decoding)
    selected_uuids = {sample.uuid for sample in samples}
    if any(
        str(record["uuid"]) in selected_uuids
        and int(record["rollout_index"]) >= num_rollouts
        for record in generation_cache.values()
    ):
        raise CacheProvenanceError(
            "the cache contains more rollouts than requested; refusing to overwrite summaries "
            "with a smaller denominator"
        )
    prompt_cache = (
        JsonlCache(
            results_dir / "answer_gain_prompts.jsonl",
            key=lambda record: (str(record["uuid"]), int(record["prefix_index"])),
            resume=resume,
        )
        if save_prompts
        else None
    )
    verifier = MathVerifier()
    started = time.monotonic()
    generated_now = 0
    tokens_now = 0

    for sample_number, sample in enumerate(samples, start=1):
        requests: list[GenerationRequest] = []
        request_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
        try:
            for prefix_index in range(len(sample.steps) + 1):
                supplied = sample.steps[:prefix_index]
                user_prompt = answer_gain_user_prompt(sample.problem, supplied)
                require_clean_audit(
                    audit_prompt_sources(_trajectory_view(sample), supplied, user_prompt)
                )
                rendered = render_chat_prompt(
                    tokenizer, user_prompt, is_qwen3=is_qwen3, enable_thinking=False
                )
                prompt_sha = _prompt_hash(rendered)
                if prompt_cache is not None:
                    existing_prompt = prompt_cache.get((sample.uuid, prefix_index))
                    if (
                        existing_prompt is not None
                        and existing_prompt.get("prompt_sha256") != prompt_sha
                    ):
                        raise CacheProvenanceError(
                            f"cached debug prompt differs for {(sample.uuid, prefix_index)}"
                        )
                    prompt_cache.append(
                        {
                            "uuid": sample.uuid,
                            "prefix_index": prefix_index,
                            "rendered_prompt": rendered,
                            "prompt_sha256": prompt_sha,
                        }
                    )
                for rollout_index in range(num_rollouts):
                    key = (sample.uuid, prefix_index, rollout_index)
                    if key in generation_cache:
                        if generation_cache.get(key).get("prompt_sha256") != prompt_sha:
                            raise CacheProvenanceError(f"cached prompt differs for {key}")
                        continue
                    requests.append(
                        GenerationRequest(
                            request_id=key,
                            prompt=rendered,
                            seed=rollout_seed(seed, "answer_gain", *key),
                        )
                    )
                    request_meta[key] = {
                        "uuid": sample.uuid,
                        "prefix_index": prefix_index,
                        "rollout_index": rollout_index,
                        "prompt_sha256": prompt_sha,
                    }
        except CacheProvenanceError:
            raise
        except Exception as error:
            exclusions.add("prompt_construction_failure", sample.uuid, str(error))
            continue

        try:
            outputs = generator.generate(requests, decoding, batch_size=batch_size) if requests else []
        except Exception as error:
            reason = "oom" if "out of memory" in str(error).lower() else "generation_failure"
            exclusions.add(reason, sample.uuid, f"{type(error).__name__}: {error}")
            continue
        for output in outputs:
            key = tuple(output["request_id"])
            record = _verified_record(
                output,
                base=request_meta[key],
                verifier=verifier,
                answer=sample.answer,
                decoding=decoding,
            )
            generation_cache.append(record)
            generated_now += 1
            tokens_now += int(record["generated_token_count"])
        elapsed = max(time.monotonic() - started, 1e-9)
        LOGGER.info(
            "answer-gain samples=%d/%d generations=%d tokens=%d generations/sec=%.3f",
            sample_number,
            len(samples),
            generated_now,
            tokens_now,
            generated_now / elapsed,
        )

    prefix_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    for sample in samples:
        correctness: list[float] = []
        complete = True
        for prefix_index in range(len(sample.steps) + 1):
            keys = [(sample.uuid, prefix_index, rollout) for rollout in range(num_rollouts)]
            value = _correctness(generation_cache, keys, num_rollouts)
            if value is None:
                complete = False
                exclusions.add("incomplete_rollouts", sample.uuid, f"prefix={prefix_index}")
                break
            correctness.append(value)
            prefix_records.append(
                {"uuid": sample.uuid, "prefix_index": prefix_index, "correctness": value}
            )
        if not complete:
            continue
        for index, gain in enumerate(answer_gains(correctness)):
            weight = sample.step_records[index]
            step_records.append(
                {
                    "uuid": sample.uuid,
                    "step_index": index,
                    "step_text": sample.steps[index],
                    "weight": weight["weight"],
                    "is_high_weight": weight["is_high_weight"],
                    "weight_group": weight["weight_group"],
                    "correctness_before": correctness[index],
                    "correctness_after": correctness[index + 1],
                    "answer_gain": gain,
                }
            )
    atomic_write_jsonl(results_dir / "answer_gain_prefixes.jsonl", prefix_records)
    atomic_write_jsonl(results_dir / "answer_gain.jsonl", step_records)
    return {
        "steps": len(step_records),
        "generation_statistics": verification_statistics(list(generation_cache.values())),
        "generated_this_run": generated_now,
        "tokens_generated_this_run": tokens_now,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_removal(
    samples: Sequence[WeightedSample],
    *,
    tokenizer: Any,
    generator: VLLMGenerator,
    decoding: DecodingConfig,
    num_rollouts: int,
    seed: int,
    batch_size: int,
    results_dir: Path,
    is_qwen3: bool,
    resume: bool,
    save_prompts: bool,
    exclusions: ExclusionReport,
) -> dict[str, Any]:
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    generation_cache = JsonlCache(
        results_dir / "removal_generations.jsonl",
        key=lambda record: (
            str(record["uuid"]),
            str(record["condition"]),
            int(record["step_index"]),
            int(record["rollout_index"]),
        ),
        resume=resume,
    )
    _require_matching_decoding(generation_cache, decoding)
    selected_uuids = {sample.uuid for sample in samples}
    if any(
        str(record["uuid"]) in selected_uuids
        and int(record["rollout_index"]) >= num_rollouts
        for record in generation_cache.values()
    ):
        raise CacheProvenanceError(
            "the cache contains more rollouts than requested; refusing to overwrite summaries "
            "with a smaller denominator"
        )
    prompt_cache = (
        JsonlCache(
            results_dir / "removal_prompts.jsonl",
            key=lambda record: (
                str(record["uuid"]), str(record["condition"]), int(record["step_index"])
            ),
            resume=resume,
        )
        if save_prompts
        else None
    )
    verifier = MathVerifier()
    started = time.monotonic()
    generated_now = 0
    tokens_now = 0

    for sample_number, sample in enumerate(samples, start=1):
        requests: list[GenerationRequest] = []
        request_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
        conditions = [("full", -1, sample.steps)] + [
            ("minus", index, sample.steps[:index] + sample.steps[index + 1 :])
            for index in range(len(sample.steps))
        ]
        try:
            for condition, step_index, supplied in conditions:
                user_prompt = removal_user_prompt(sample.problem, supplied)
                require_clean_audit(
                    audit_prompt_sources(_trajectory_view(sample), supplied, user_prompt)
                )
                rendered = render_chat_prompt(
                    tokenizer, user_prompt, is_qwen3=is_qwen3, enable_thinking=False
                )
                prompt_sha = _prompt_hash(rendered)
                if prompt_cache is not None:
                    prompt_key = (sample.uuid, condition, step_index)
                    existing_prompt = prompt_cache.get(prompt_key)
                    if (
                        existing_prompt is not None
                        and existing_prompt.get("prompt_sha256") != prompt_sha
                    ):
                        raise CacheProvenanceError(
                            f"cached debug prompt differs for {prompt_key}"
                        )
                    prompt_cache.append(
                        {
                            "uuid": sample.uuid,
                            "condition": condition,
                            "step_index": step_index,
                            "rendered_prompt": rendered,
                            "prompt_sha256": prompt_sha,
                        }
                    )
                for rollout_index in range(num_rollouts):
                    key = (sample.uuid, condition, step_index, rollout_index)
                    if key in generation_cache:
                        if generation_cache.get(key).get("prompt_sha256") != prompt_sha:
                            raise CacheProvenanceError(f"cached prompt differs for {key}")
                        continue
                    requests.append(
                        GenerationRequest(
                            request_id=key,
                            prompt=rendered,
                            seed=rollout_seed(seed, "removal", *key),
                        )
                    )
                    request_meta[key] = {
                        "uuid": sample.uuid,
                        "condition": condition,
                        "step_index": step_index,
                        "rollout_index": rollout_index,
                        "prompt_sha256": prompt_sha,
                    }
        except CacheProvenanceError:
            raise
        except Exception as error:
            exclusions.add("prompt_construction_failure", sample.uuid, str(error))
            continue

        try:
            outputs = generator.generate(requests, decoding, batch_size=batch_size) if requests else []
        except Exception as error:
            reason = "oom" if "out of memory" in str(error).lower() else "generation_failure"
            exclusions.add(reason, sample.uuid, f"{type(error).__name__}: {error}")
            continue
        for output in outputs:
            key = tuple(output["request_id"])
            record = _verified_record(
                output,
                base=request_meta[key],
                verifier=verifier,
                answer=sample.answer,
                decoding=decoding,
            )
            generation_cache.append(record)
            generated_now += 1
            tokens_now += int(record["generated_token_count"])
        elapsed = max(time.monotonic() - started, 1e-9)
        LOGGER.info(
            "removal samples=%d/%d generations=%d tokens=%d generations/sec=%.3f",
            sample_number,
            len(samples),
            generated_now,
            tokens_now,
            generated_now / elapsed,
        )

    condition_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    for sample in samples:
        full_keys = [(sample.uuid, "full", -1, rollout) for rollout in range(num_rollouts)]
        full = _correctness(generation_cache, full_keys, num_rollouts)
        if full is None:
            exclusions.add("incomplete_rollouts", sample.uuid, "full")
            continue
        minus_values: list[float] = []
        complete = True
        condition_records.append(
            {"uuid": sample.uuid, "condition": "full", "step_index": -1, "correctness": full}
        )
        for index in range(len(sample.steps)):
            keys = [(sample.uuid, "minus", index, rollout) for rollout in range(num_rollouts)]
            value = _correctness(generation_cache, keys, num_rollouts)
            if value is None:
                exclusions.add("incomplete_rollouts", sample.uuid, f"minus={index}")
                complete = False
                break
            minus_values.append(value)
            condition_records.append(
                {"uuid": sample.uuid, "condition": "minus", "step_index": index, "correctness": value}
            )
        if not complete:
            continue
        for index, damage in enumerate(removal_damages(full, minus_values)):
            weight = sample.step_records[index]
            step_records.append(
                {
                    "uuid": sample.uuid,
                    "step_index": index,
                    "step_text": sample.steps[index],
                    "weight": weight["weight"],
                    "is_high_weight": weight["is_high_weight"],
                    "weight_group": weight["weight_group"],
                    "C_full": full,
                    "C_minus_i": minus_values[index],
                    "removal_damage": damage,
                }
            )
    atomic_write_jsonl(results_dir / "removal_conditions.jsonl", condition_records)
    atomic_write_jsonl(results_dir / "removal_damage.jsonl", step_records)
    return {
        "steps": len(step_records),
        "generation_statistics": verification_statistics(list(generation_cache.values())),
        "generated_this_run": generated_now,
        "tokens_generated_this_run": tokens_now,
        "elapsed_seconds": time.monotonic() - started,
    }
