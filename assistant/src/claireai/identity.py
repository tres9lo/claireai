"""What ClaireAI can honestly say about itself.

A small local model cannot be relied on for questions about the assistant it is
running inside: it does not know which tools exist, which folders are allowed, or
whether a voice engine is installed, and asking it produces confident guesses.
The same questions are cheap to answer from the runtime's real state, so they are
answered here instead -- before a model is asked, and even when no model is
installed at all.

Everything in this module is derived from :class:`claireai.config.Settings` and
the control file. Nothing is invented, and a feature that is off says it is off.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

NAME = "ClaireAI"
TAGLINE = "a local-first assistant running on this computer"

_WHO = re.compile(
    r"\b(who are you|what are you|your name|introduce yourself|tell me about yourself)\b",
    re.IGNORECASE,
)
_CAN = re.compile(
    r"\b(what can you do|what do you do|your capabilit|what are you able to do|"
    r"how can you help|what can i ask you)\b",
    re.IGNORECASE,
)
_CANNOT = re.compile(
    r"\b(what can'?t you do|what are your limitations|your limits|"
    r"what are you not able to do|what can you not do)\b",
    re.IGNORECASE,
)
_LEARN = re.compile(
    r"\b(how do you learn|do you learn|are you learning|self.?improv\w*|"
    r"do you get better|can you train yourself)\b",
    re.IGNORECASE,
)
_MODEL = re.compile(
    r"\b(which model|what model|are you (an? )?(ai|human|chatgpt|claude|gpt)|"
    r"what are you running on|do you use the internet)\b",
    re.IGNORECASE,
)
_MADE = re.compile(
    r"\b(who (made|built|trained) you|how were you (made|built|trained)|"
    r"are you (trained|pretrained|pre-trained)|from scratch|do you use pretrained)\b",
    re.IGNORECASE,
)
_HOW_ARE_YOU = re.compile(
    r"\b(how are you|how'?s it going|how are things|how do you feel|comment vas[- ]tu)\b",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^\s*(thanks|thank you|thx|merci)\b", re.IGNORECASE)
_BYE = re.compile(
    r"^\s*(bye|goodbye|good night|see you|au revoir|à plus)\b", re.IGNORECASE
)


def _checkpoint_name(settings, model_ready: bool | None) -> str | None:
    if getattr(settings, "brain_provider", "ollama") != "scratch":
        return None
    path = getattr(settings, "resolved_model_checkpoint", None)
    if path is None:
        return None
    if model_ready is None:
        model_ready = path.is_file()
    return path.stem if model_ready else None


def _model_line(settings, model_ready: bool | None) -> str:
    provider = getattr(settings, "brain_provider", "ollama")
    model = str(getattr(settings, "brain_model", "") or "")
    coding = str(getattr(settings, "brain_coding_model", "") or "")
    if provider == "ollama":
        line = (
            f"Replies come from {model or 'a Qwen model'} through the local Ollama "
            "runtime; the model and the conversation stay on this machine."
        )
        if coding:
            line += f" Code-heavy turns use {coding}."
        return line
    if provider in {"openai", "anthropic"}:
        return (
            f"Replies come from the {provider} connector you configured; "
            "that hosted endpoint can see the conversation you send it."
        )
    checkpoint = getattr(settings, "resolved_model_checkpoint", None)
    if model_ready is None:
        model_ready = bool(checkpoint and checkpoint.is_file())
    if model_ready:
        return f"Replies come from the archived local checkpoint {checkpoint.name}, running here with no network access."
    return (
        "No local scratch checkpoint is installed. The normal runtime is Ollama + Qwen3; "
        "set CLAIREAI_BRAIN_PROVIDER=ollama to use it."
    )


def capability_lines(
    settings,
    *,
    model_ready: bool | None = None,
    control: Mapping[str, bool] | None = None,
    learning: Mapping[str, Any] | None = None,
) -> list[str]:
    """One honest line per capability, in the order a person would ask."""
    control = dict(control or {})
    folders = [str(path) for path in getattr(settings, "allowed_folders", ())]
    lines = [f"I am {NAME}, {TAGLINE}."]
    lines.append(_model_line(settings, model_ready))

    if folders:
        shown = ", ".join(folders[:3]) + (" …" if len(folders) > 3 else "")
        lines.append(
            f"Files: inside {shown} I can list, search, read, create, edit with exact "
            "replacements, rename and move files or folders."
        )
    else:
        lines.append("Files: no folders are allowed yet, so file tools can reach nothing.")

    lines.append(
        "Calendar: I keep a local calendar and can list or add appointments; "
        "the runtime answers date and agenda questions itself."
    )
    lines.append(
        "Actions: every write and every move-to-trash asks you first. The allow-list is "
        "deny-by-default, and removal uses a recoverable trash folder rather than "
        "permanent deletion."
    )
    capabilities = {str(item).strip().lower() for item in getattr(settings, "allowed_capabilities", ())}
    if capabilities:
        lines.append(
            "Machine access: enabled for "
            + ", ".join(sorted(capabilities))
            + ". Every screen, pointer or keyboard action still asks for approval."
        )
    else:
        lines.append("Machine access: off; screen, pointer and keyboard tools are refused.")
    lines.append("Web: browsing and downloads are " + ("on." if control.get("browser") else "off."))

    voices = []
    try:
        from claireai.voice import available_voices

        voices = list(available_voices().values())
    except Exception:  # noqa: BLE001 - a missing speech engine must not break the answer
        voices = []
    lines.append("Voice: " + (f"speaks locally ({', '.join(voices)})." if voices else "no local speech engine is installed."))

    if control.get("self_improvement"):
        count = (learning or {}).get("total", 0)
        lines.append(
            f"Learning: on. I keep a private journal of exchanges ({count} so far) for review; "
            "it does not retrain the Qwen weights."
        )
    else:
        lines.append("Learning: off. Turn on Background learning in Settings and I will journal exchanges locally for review.")
    return lines


def answer_self_question(
    question: str,
    settings,
    *,
    model_ready: bool | None = None,
    control: Mapping[str, bool] | None = None,
    learning: Mapping[str, Any] | None = None,
) -> str | None:
    """A grounded answer to an identity/capability question, or ``None``.

    Returning ``None`` is the important case: a normal request must still reach
    the model, so only questions that are clearly about the assistant match.
    """
    text = " ".join(question.split())
    if not text or len(text) > 400:
        return None

    # Ordinary conversational turns are answered here rather than handed to a
    # three-million-parameter model that has no reliable idea of who it is.
    # These are honest runtime replies, not a scripted imitation of a person.
    if _HOW_ARE_YOU.search(text):
        if _checkpoint_name(settings, model_ready) or getattr(settings, "brain_provider", "ollama") != "scratch":
            return (
                "I do not have feelings, but I am running well: the local model runtime is "
                "configured, the tools are ready, and inference stays on this machine. "
                "How are you?"
            )
        return (
            "I am running, though no scratch checkpoint is installed, so I can only answer "
            "questions about myself until one is. How are you?"
        )
    if _MADE.search(text):
        provider = getattr(settings, "brain_provider", "ollama")
        if provider == "ollama":
            model = str(getattr(settings, "brain_model", "") or "Qwen3 1.7B")
            return (
                f"I run the pretrained {model} model through Ollama on this machine, and "
                "this Python runtime adds tools, calendar, storage and approvals. I do not "
                "train the model from scratch."
            )
        checkpoint = _checkpoint_name(settings, model_ready)
        if checkpoint:
            return (
                f"I am running the archived scratch checkpoint {checkpoint}, which brain-lab "
                "trained here. The supported default is Ollama + Qwen3, not scratch training."
            )
        return (
            "This runtime is designed around pretrained Qwen models served by Ollama; "
            "no scratch checkpoint is installed for the legacy path."
        )
    if _THANKS.search(text):
        return "You are welcome. Anything else you want to get done?"
    if _BYE.search(text):
        return "See you. Your data stays here, and I will pick up where we left off."

    if _WHO.match(text) or _WHO.search(text):
        return (
            f"I am {NAME}, {TAGLINE}. My job is to be useful without sending your "
            "data anywhere: I run tools on this machine, keep my own calendar and "
            "storage, and ask before anything that changes a file."
        )
    if _CANNOT.search(text):
        return (
            "Honest limits: I only reach folders you allowed and I cannot act without your "
            "approval on a write. I have no permanent delete. I do not know things that are "
            "not on this machine unless web tools are on and I look them up. The model can "
            "still be wrong or make something up, so verify facts that matter."
        )
    if _LEARN.search(text):
        return (
            "The Qwen weights are pretrained and I do not retrain them during a conversation. "
            "With Background learning on, I can journal exchanges locally for review, but "
            "that journal does not silently rewrite the model."
        )
    if _MODEL.search(text):
        return (
            _model_line(settings, model_ready)
            + " I am not a person and I am not a hosted chatbot; the runtime around me is "
            "ordinary Python on your machine."
        )
    if _CAN.search(text):
        return "\n".join(
            "• " + line for line in capability_lines(
                settings, model_ready=model_ready, control=control, learning=learning
            )
        )
    return None


def system_prompt(settings, *, control: Mapping[str, bool] | None = None) -> str:
    """A compact, accurate self-description for a conversational model."""
    folders = [str(path) for path in getattr(settings, "allowed_folders", ())]
    return (
        f"You are {NAME}, {TAGLINE}. Be friendly and concrete, and say plainly when you are "
        "unsure. You can read files only inside the folders the owner allowed "
        f"({', '.join(folders) if folders else 'none yet'}), keep a local calendar, and use "
        "tools when they are offered. Writes wait for the owner's approval, and nothing is "
        "deleted permanently. Web tools are " + ("on" if (control or {}).get("browser") else "off")
        + ". Never claim an action you did not take."
    )
