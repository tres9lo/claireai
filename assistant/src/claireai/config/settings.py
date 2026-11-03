"""Validated runtime configuration.

Configuration is validated once at startup and fails loudly: a partially
configured assistant that silently points at the wrong model or the wrong data
directory is worse than one that refuses to start.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from platformdirs import user_data_path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "ClaireAI"

# Machine-access capabilities (docs/architecture/machine-access.md). Knowing the
# names here lets configuration reject typos early; the tools that use them land
# in Month 1 Week 3 and Months 2-3.
KNOWN_CAPABILITIES = frozenset({"screen", "camera", "cursor", "keyboard", "clipboard"})


class Settings(BaseSettings):
    """Settings sourced from ``CLAIREAI_*`` environment variables or ``.env``.

    The brain is an on-disk checkpoint; inference makes no network requests.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAIREAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    model_checkpoint: Path | None = None
    max_output_tokens: int = Field(default=128, ge=8, le=1024)
    owner_name: str = "Bikorimana Ange Tresor"

    # Tool sandbox (roadmap 7.2). Empty by default: file tools can reach nothing
    # until the owner names the folders. Set as a JSON array, e.g.
    # CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]'.
    allowed_folders: list[Path] = Field(default_factory=list)
    agent_max_steps: int = Field(default=8, ge=1, le=24)

    # Tool routing (claireai.routing). `auto` narrows the schemas sent to the
    # model to the tools its intent can reach; `off` sends all of them. Routing
    # changes what the model is shown, never what it is allowed to do.
    tool_routing: Literal["auto", "off"] = "auto"

    # Presence renderer (docs/architecture/presence-renderer.md). `auto` means
    # WebGL when the webview supports it and holds the frame budget, else 2D.

    # Decoding temperature, chosen per phase by the agent loop rather than fixed:
    # while no tool has run the model is deciding what to do, which wants
    # repeatability; once it has real data it is phrasing an answer, which wants
    # natural language. Defaults are deliberately not 0 — measured, a smaller
    # model pinned at 0 emitted a malformed tool call on every single attempt.
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    temperature_tools: float = Field(default=0.2, ge=0.0, le=2.0)
    response_style: Literal["concise", "balanced", "detailed", "playful"] = "balanced"

    presence_mode: Literal["off", "auto", "2d", "webgl"] = "auto"
    presence_fps_cap: int = Field(default=30, ge=5, le=60)
    presence_reduce_motion: bool = False

    # Machine access is deny-by-default, like the folder allow-list.
    allowed_capabilities: list[str] = Field(default_factory=list)
    camera_device: int = Field(default=0, ge=0)

    # Offline voice output is opt-in; pitch/rate map to the local eSpeak voice.
    voice_enabled: bool = False
    voice_pitch: int = Field(default=50, ge=0, le=99)
    voice_rate: int = Field(default=175, ge=80, le=450)

    # Development override. Production data belongs in the OS application-data
    # directory resolved by ``default_data_dir``.
    data_dir: Path | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        owner_name = self.owner_name.strip()
        if not owner_name:
            raise ValueError("CLAIREAI_OWNER_NAME must not be empty")
        object.__setattr__(self, "owner_name", owner_name)
        cleaned: list[Path] = []
        for folder in self.allowed_folders:
            text = str(folder).strip()
            if not text:
                raise ValueError("CLAIREAI_ALLOWED_FOLDERS must not contain empty entries")
            cleaned.append(Path(text).expanduser())
        object.__setattr__(self, "allowed_folders", cleaned)

        capabilities = [str(name).strip().lower() for name in self.allowed_capabilities]
        unknown = sorted({name for name in capabilities if name not in KNOWN_CAPABILITIES})
        if unknown:
            raise ValueError(
                f"CLAIREAI_ALLOWED_CAPABILITIES contains unknown capability {unknown[0]!r}. "
                f"Known capabilities: {', '.join(sorted(KNOWN_CAPABILITIES))}"
            )
        object.__setattr__(self, "allowed_capabilities", list(dict.fromkeys(capabilities)))
        return self

    @property
    def tool_temperature(self) -> float:
        """Temperature while the loop is still choosing a tool."""
        return self.temperature_tools

    @property
    def default_data_dir(self) -> Path:
        """OS-specific application-data directory (not the working directory)."""
        return Path(user_data_path(APP_NAME))

    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser() if self.data_dir else self.default_data_dir

    @property
    def database_path(self) -> Path:
        return self.resolved_data_dir / "db" / "claireai.sqlite3"

    @property
    def resolved_model_checkpoint(self) -> Path:
        return (
            Path(self.model_checkpoint).expanduser()
            if self.model_checkpoint
            else self.resolved_data_dir / "models" / "claire-scratch.pt"
        )

    @property
    def control_state_path(self) -> Path:
        return self.resolved_data_dir / "runtime-control.json"

    @property
    def log_dir(self) -> Path:
        return self.resolved_data_dir / "logs"

    @property
    def allowed_folder_roots(self) -> tuple[Path, ...]:
        """Allow-list entries as absolute paths, without requiring them to exist."""
        return tuple(folder.expanduser().resolve(strict=False) for folder in self.allowed_folders)

    def ensure_directories(self) -> None:
        for directory in (self.database_path.parent, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                try:
                    directory.chmod(0o700)
                except OSError:
                    pass
