"""Hugging Face Transformers implementation of the model-adapter contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TransformersAdapter:
    def __init__(
        self,
        model_spec: dict[str, Any],
        *,
        precision: str,
        quantization: str,
        allow_download: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Transformers execution requires the optional dependencies. "
                "Install with: python3 -m pip install -e '.[transformers]'"
            ) from error

        self._torch = torch
        repository = model_spec["repository"]
        revision = model_spec["revision"]
        source: str | Path = repository
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            local_files_only=not allow_download,
            trust_remote_code=False,
        )

        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        dtype_name = self._resolve_precision(precision, device)
        dtype = getattr(torch, dtype_name)
        load_options: dict[str, Any] = {
            "revision": revision,
            "local_files_only": not allow_download,
            "trust_remote_code": False,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        applied_quantization = "none"
        if quantization == "4bit_if_supported" and device == "cuda":
            try:
                import bitsandbytes  # noqa: F401 - verifies the runtime is usable
                from transformers import BitsAndBytesConfig

                load_options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                load_options["device_map"] = "auto"
                applied_quantization = "bitsandbytes_4bit"
            except ImportError:
                applied_quantization = "none"

        self.model = AutoModelForCausalLM.from_pretrained(source, **load_options).eval()
        if "device_map" not in load_options:
            self.model.to(device)
        self._identity = {
            "family_id": model_spec["family_id"],
            "model_alias": model_spec["alias"],
            "repository": repository,
            "revision": revision,
            "backend": "transformers",
            "precision": dtype_name,
            "quantization": applied_quantization,
            "device": device,
        }

    def _resolve_precision(self, requested: str, device: str) -> str:
        if requested == "float32":
            return "float32"
        if requested in {"bfloat16", "bfloat16_if_supported"}:
            if device == "cuda" and self._torch.cuda.is_bf16_supported():
                return "bfloat16"
            if device == "mps":
                return "float16"
            return "float32"
        if requested == "auto":
            return "float16" if device in {"cuda", "mps"} else "float32"
        raise ValueError(f"Unsupported precision policy: {requested}")

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def score_options(self, prompt: str, options: list[str]) -> dict[str, Any]:
        from ..mcq import score_options, score_single_token_options

        token_lengths = [len(self.tokenizer(option, add_special_tokens=False)["input_ids"]) for option in options]
        result = (
            score_single_token_options(self.model, self.tokenizer, prompt, options)
            if all(length == 1 for length in token_lengths)
            else score_options(self.model, self.tokenizer, prompt, options)
        )
        return result.to_dict()

    def generate(self, prompt: str, **generation: Any) -> dict[str, Any]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            output = self.model.generate(**encoded, **generation)
        generated = output[0, encoded["input_ids"].shape[1] :]
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True)}

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
