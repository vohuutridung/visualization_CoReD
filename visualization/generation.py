from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 32768


@dataclass(frozen=True)
class GenerationRequest:
    request_id: tuple[Any, ...]
    prompt: str
    seed: int
    lora_path: str | None = None
    lora_name: str | None = None
    stop: tuple[str, ...] = ()


def rollout_seed(global_seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(global_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def chunked(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


class VLLMGenerator:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 65536,
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 64,
        dtype: str = "bfloat16",
    ) -> None:
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": model,
            "revision": revision,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": False,
            "dtype": dtype,
            "enable_lora": enable_lora,
        }
        if enable_lora:
            kwargs.update(max_loras=max_loras, max_lora_rank=max_lora_rank)
        self.llm = LLM(**kwargs)
        self.enable_lora = enable_lora
        self._lora_ids: dict[str, int] = {}

    def _lora_request(self, request: GenerationRequest):
        if request.lora_path is None:
            return None
        if not self.enable_lora:
            raise ValueError("a LoRA request was supplied to a base-only generator")
        from vllm.lora.request import LoRARequest

        if request.lora_path not in self._lora_ids:
            self._lora_ids[request.lora_path] = len(self._lora_ids) + 1
        return LoRARequest(
            request.lora_name or f"expert_{self._lora_ids[request.lora_path]}",
            self._lora_ids[request.lora_path],
            request.lora_path,
        )

    def generate(
        self,
        requests: Sequence[GenerationRequest],
        decoding: DecodingConfig,
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        records: list[dict[str, Any]] = []
        for batch in chunked(requests, batch_size):
            params = [
                SamplingParams(
                    n=1,
                    temperature=decoding.temperature,
                    top_p=decoding.top_p,
                    top_k=decoding.top_k,
                    min_p=decoding.min_p,
                    repetition_penalty=decoding.repetition_penalty,
                    max_tokens=decoding.max_new_tokens,
                    seed=request.seed,
                    stop=list(request.stop) or None,
                    include_stop_str_in_output=False,
                )
                for request in batch
            ]
            lora_requests = [self._lora_request(request) for request in batch]
            kwargs: dict[str, Any] = {"use_tqdm": False}
            if any(value is not None for value in lora_requests):
                kwargs["lora_request"] = lora_requests
            outputs = self.llm.generate(
                [request.prompt for request in batch], params, **kwargs
            )
            for request, output in zip(batch, outputs):
                candidate = output.outputs[0]
                text = candidate.text.split("\n\n", 1)[0] if "\n\n" in request.stop else candidate.text
                delimiter_stop = getattr(candidate, "stop_reason", None) in request.stop
                finish_reason = getattr(candidate, "finish_reason", None)
                records.append(
                    {
                        "request_id": list(request.request_id),
                        "generated_text": text,
                        "generation_seed": request.seed,
                        "finish_reason": finish_reason,
                        "stop_reason": getattr(candidate, "stop_reason", None),
                        "generated_token_count": len(getattr(candidate, "token_ids", [])),
                        "terminated_by_delimiter": delimiter_stop,
                        "termination": (
                            "delimiter"
                            if delimiter_stop
                            else "max_token_cap"
                            if finish_reason == "length"
                            else "eos_or_other"
                        ),
                    }
                )
        return records
