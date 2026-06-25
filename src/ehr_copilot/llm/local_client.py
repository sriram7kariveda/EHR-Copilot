"""Local LLM Client — loads model directly with transformers.

Used when vLLM is not available. Loads Qwen 3.5 4B directly on GPU
using HuggingFace transformers. Suitable for A100 (40/80GB).
"""

from __future__ import annotations

import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LocalLLMClient(LLMClient):
    """LLM client that loads model directly on GPU."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        self._model_name = model_name
        logger.info("Loading model %s...", model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info("Model loaded. GPU memory: %.1f GB", mem_gb)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 0.01),
                do_sample=request.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        response_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        elapsed = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response_text,
            model=self._model_name,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=outputs.shape[1] - inputs["input_ids"].shape[1],
            latency_ms=round(elapsed, 2),
        )

    async def is_available(self) -> bool:
        return torch.cuda.is_available()
