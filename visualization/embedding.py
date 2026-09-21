from __future__ import annotations

from typing import Sequence


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[rows, sequence_lengths]


class QwenStepEmbedder:
    """Symmetric, instruction-free Qwen3 step encoder with normalized output."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 32768,
        attn_implementation: str | None = "flash_attention_2",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            padding_side="left",
            trust_remote_code=False,
        )
        dtype_value = getattr(torch, dtype.replace("bf16", "bfloat16").replace("fp16", "float16"))
        kwargs = {
            "revision": revision,
            "device_map": device_map,
            "torch_dtype": dtype_value,
            "trust_remote_code": False,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModel.from_pretrained(model_name, **kwargs)
        except (ImportError, ValueError):
            if attn_implementation != "flash_attention_2":
                raise
            kwargs["attn_implementation"] = "sdpa"
            self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.max_length = max_length
        self.eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if self.eod_id is None:
            raise RuntimeError("embedding tokenizer has no <|endoftext|> token")

    def _tokenize(self, texts: Sequence[str]):
        encoded = self.tokenizer(
            list(texts), padding=False, truncation=True, max_length=self.max_length - 2
        )
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            ids.append(self.eod_id)
            mask.append(1)
        return self.tokenizer.pad(encoded, padding=True, return_tensors="pt")

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        import torch
        import torch.nn.functional as functional

        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        device = next(self.model.parameters()).device
        result: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self._tokenize(texts[start : start + batch_size]).to(device)
                output = self.model(**batch)
                embeddings = last_token_pool(output.last_hidden_state, batch["attention_mask"])
                embeddings = functional.normalize(embeddings.float(), p=2, dim=1)
                result.extend(embeddings.cpu().tolist())
        return result
