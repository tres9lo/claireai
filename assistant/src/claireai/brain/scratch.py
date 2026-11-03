"""Adapter for ClaireAI's locally trained, from-scratch transformer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from claireai.brain.base import (
    BrainHealth, BrainResponse, BrainResponseError, BrainUnavailable, Message, ToolCall,
)


class ScratchBrain:
    """Load only a checkpoint trained by ``brain-lab``; no remote runtime."""

    name = "scratch"
    _tool_open = "<tool_call>"
    _tool_close = "</tool_call>"

    def __init__(self, checkpoint: Path, *, max_output_tokens: int = 128) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        self.max_output_tokens = max_output_tokens
        self._model = None
        self._torch = None
        self._tokenizer = None
        self._manifest: dict[str, Any] = {}

    async def health(self) -> BrainHealth:
        if not self.checkpoint.is_file():
            return BrainHealth(False, False, detail=f"checkpoint not found: {self.checkpoint}")
        try:
            self._load()
        except (ImportError, RuntimeError, ValueError, OSError, BrainResponseError) as exc:
            return BrainHealth(False, False, detail=str(exc))
        label = str(self._manifest.get("name", self.checkpoint.stem))
        return BrainHealth(True, True, (label,), "scratch checkpoint loaded on CPU")

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from claire_brain.model import ScratchTransformer
            from claire_brain.tokenizer import BytePairTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "scratch model dependencies are missing; install assistant[model] and brain-lab"
            ) from exc
        try:
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            manifest = checkpoint["manifest"]
            if manifest.get("tokenizer") != "corpus-bpe-v1":
                raise ValueError("unsupported tokenizer in scratch checkpoint")
            architecture = manifest["architecture"]
            model = ScratchTransformer(**architecture)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise BrainResponseError(f"invalid scratch checkpoint {self.checkpoint}: {exc}") from exc
        tokenizer = BytePairTokenizer([tuple(pair) for pair in manifest["tokenizer_merges"]])
        self._torch, self._model, self._tokenizer, self._manifest = torch, model, tokenizer, manifest

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        temperature: float | None = None,
    ) -> BrainResponse:
        try:
            self._load()
        except (ImportError, RuntimeError, ValueError, OSError, KeyError) as exc:
            raise BrainUnavailable(str(exc)) from exc
        prompt = "\n".join(
            f"{message.role.upper()}{' ' + message.tool_name if message.tool_name else ''}: {message.content}"
            for message in messages
        )
        if tools:
            prompt += (
                "\nAvailable tools (untrusted request routing; runtime will validate):\n"
                + json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))
                + "\nIf a tool is needed, emit exactly <tool_call>{\"name\":\"...\","
                  "\"arguments\":{...}}</tool_call>. Otherwise answer normally.\n"
            )
        prompt += "\nASSISTANT:"
        torch = self._torch
        model = self._model
        tokens = self._tokenizer.encode(prompt, append_eos=False)
        context = model.context
        tokens = tokens[-context:]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        try:
            result = model.generate(
                input_ids,
                max_new_tokens=self.max_output_tokens,
                temperature=0.0 if temperature is None else max(0.0, temperature),
            )[0].tolist()[len(tokens):]
        except (RuntimeError, ValueError) as exc:
            raise BrainResponseError(f"scratch model inference failed: {exc}") from exc
        output = self._tokenizer.decode(result).rstrip("\ufffd").strip()
        call = self._parse_tool_call(output)
        if call:
            return BrainResponse(content="", model=self._model_name(), tool_calls=(call,))
        return BrainResponse(content=output, model=self._model_name(), completion_tokens=len(result))

    def _model_name(self) -> str:
        return str(self._manifest.get("name", f"scratch:{self.checkpoint.stem}"))

    def _parse_tool_call(self, output: str) -> ToolCall | None:
        start = output.find(self._tool_open)
        end = output.find(self._tool_close, start + len(self._tool_open))
        if start < 0 or end < 0:
            return None
        try:
            payload = json.loads(output[start + len(self._tool_open):end])
        except ValueError as exc:
            raise BrainResponseError("scratch model emitted malformed tool-call JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise BrainResponseError("scratch model tool call must contain a string name")
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise BrainResponseError("scratch model tool arguments must be an object")
        return ToolCall(name=payload["name"], arguments=arguments)

    async def aclose(self) -> None:
        self._model = None
