"""Adapter for ClaireAI's locally trained, from-scratch transformer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from claireai.brain.base import (
    BrainHealth, BrainResponse, BrainResponseError, BrainStopped, BrainUnavailable, Message, ToolCall,
)
from claireai.runtime_control import RuntimeControl


class ScratchBrain:
    """Load only a checkpoint trained by ``brain-lab``; no remote runtime."""

    name = "scratch"
    _tool_open = "<tool_call>"
    _tool_close = "</tool_call>"

    def __init__(
        self,
        checkpoint: Path,
        *,
        max_output_tokens: int = 128,
        control_path: Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        self.max_output_tokens = max_output_tokens
        self.control_path = Path(control_path) if control_path is not None else None
        self._model = None
        self._torch = None
        self._tokenizer = None
        self._manifest: dict[str, Any] = {}

    def _stop_requested(self) -> bool:
        return bool(self.control_path and RuntimeControl(self.control_path).read()["shutdown"])

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
        try:
            from claire_brain.prompt import render_prompt
        except ImportError:
            # The model itself needs brain-lab, so this only happens in a unit
            # test with a fake model.  Keep a plain renderer so the adapter's
            # stop/clean behaviour stays testable without the lab installed.
            render_prompt = self._fallback_render

        # The prompt is rendered by the same code that built the training data.
        # If it does not fit, drop the oldest turns first and only then shorten
        # the current request; the system prompt and its tools are never the
        # thing thrown away.
        offered = list(tools or ())
        # Routing normally narrows this to one or two tools.  For an unrouted
        # request, six compact descriptions leave room for the request itself in
        # a 256-token window; the full registry still validates every call.
        if len(offered) > 6:
            offered = offered[:6]
        system = next((message.content for message in messages if message.role == "system"), "")
        history = [
            {
                "role": message.role,
                "content": message.content,
                "tool_name": message.tool_name,
                "tool_calls": message.tool_calls,
            }
            for message in messages
            if message.role != "system"
        ]
        prompt = self._fit_prompt(system, history, offered, render_prompt)
        tokens = self._tokenizer.encode(prompt, append_eos=False)
        torch = self._torch
        model = self._model
        input_ids = torch.tensor([tokens], dtype=torch.long)
        try:
            result = model.generate(
                input_ids,
                max_new_tokens=self.max_output_tokens,
                temperature=0.7 if temperature is None else max(0.0, temperature),
                top_p=0.9,
                repetition_penalty=1.15,
                should_stop=self._stop_requested if self.control_path else None,
            )[0].tolist()[len(tokens):]
        except (RuntimeError, ValueError) as exc:
            raise BrainResponseError(f"scratch model inference failed: {exc}") from exc
        if self._stop_requested():
            raise BrainStopped("ClaireAI was stopped from the local control panel.")
        output = self._tokenizer.decode(result).rstrip("\ufffd").strip()
        output = self._clean(output)
        call = self._parse_tool_call(output)
        if call is None and offered:
            # The model was trained on both the runtime tag format and the
            # plain-JSON format used by the evaluation suite.  Accept a bare
            # JSON call only when its name is one of the tools actually offered.
            call = self._parse_plain_tool_call(output, offered)
        if call:
            return BrainResponse(content="", model=self._model_name(), tool_calls=(call,))
        return BrainResponse(content=output, model=self._model_name(), completion_tokens=len(result))

    @staticmethod
    def _fallback_render(system: str, history: list[dict[str, Any]], tools: list[Mapping[str, Any]]) -> str:
        parts = [f"SYSTEM: {system}"]
        for message in history:
            role = str(message.get("role") or "user").upper()
            name = f" {message['tool_name']}" if message.get("tool_name") else ""
            parts.append(f"{role}{name}: {message.get('content') or ''}")
        parts.append("ASSISTANT: ")
        return "\n".join(parts)

    def _fit_prompt(self, system: str, history: list[dict[str, Any]], tools: list[Mapping[str, Any]], render) -> str:  # noqa: ANN001
        """Render a prompt inside the model's context, keeping the request.

        Priority order: the current turn, the tool block, the system prompt.
        Dropping the user's request to make room for a long instruction list is
        how a small model ends up answering nothing.
        """
        context = int(self._model.context)

        def length(text: str, kept: list[dict[str, Any]]) -> int:
            return len(self._tokenizer.encode(render(text, kept, tools), append_eos=False))

        def fits(text: str, kept: list[dict[str, Any]]) -> bool:
            return length(text, kept) <= context

        kept = list(history)
        # Drop the oldest turns first, never the current one.
        while len(kept) > 1 and not fits(system, kept):
            drop = 2 if kept[0].get("role") == "user" else 1
            kept = kept[drop:] or [kept[-1]]

        if fits(system, kept):
            return render(system, kept, tools)

        # The system prompt itself is too long.  Keep the current turn and trim
        # the middle of the system prompt, so both the instructions at the start
        # and the allowed-folder listing at the end survive.
        if kept and not fits("", kept):
            message = dict(kept[-1])
            content = str(message.get("content") or "")
            low, high, best = 0, len(content), ""
            while low <= high:
                middle = (low + high) // 2
                message["content"] = content[:middle]
                if fits("", [message]):
                    best = message["content"]
                    low = middle + 1
                else:
                    high = middle - 1
            message["content"] = best
            return render("", [message], tools)

        def clip_middle(text: str, keep: int) -> str:
            if keep <= 0:
                return ""
            if len(text) <= keep:
                return text
            head = keep * 2 // 3
            tail = keep - head
            return text[:head] + "\n...\n" + text[-tail:] if tail else text[:head]

        low, high, best = 0, len(system), ""
        while low <= high:
            middle = (low + high) // 2
            candidate = clip_middle(system, middle)
            if fits(candidate, kept):
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return render(best, kept, tools)

    @staticmethod
    def _clean(output: str) -> str:
        """Remove a prompt echo and stop at a turn boundary."""
        text = output.strip()
        if text.upper().startswith("ASSISTANT:"):
            text = text[len("ASSISTANT:"):].strip()
        for marker in ("\nSYSTEM:", "\nUSER:", "\nASSISTANT:"):
            index = text.find(marker)
            if index >= 0:
                text = text[:index].rstrip()
        return text.strip()

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

    @staticmethod
    def _parse_plain_tool_call(output: str, tools: Sequence[Mapping[str, Any]]) -> ToolCall | None:
        text = output.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        offered = {
            str((schema.get("function") or schema).get("name"))
            for schema in tools
            if isinstance(schema, Mapping)
        }
        if name not in offered:
            return None
        return ToolCall(name=name, arguments=arguments)

    async def aclose(self) -> None:
        self._model = None
