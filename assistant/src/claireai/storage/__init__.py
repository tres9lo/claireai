"""Storage package: SQLite connection, migrations and repositories."""

from claireai.storage.database import Database, MigrationError, transaction
from claireai.storage.user_data import (
    ConversationMessage,
    ConversationRepository,
    DirectiveRepository,
    PreferenceRepository,
    apply_preferences,
)
from claireai.storage.day_plan import (
    DayPlanItem,
    DayPlanRepository,
    PLAN_STATUSES,
    render_day_plan,
)
from claireai.storage.tasks import (
    ALLOWED_TRANSITIONS,
    STATUSES,
    InvalidTransition,
    Task,
    TaskRepository,
    utc_now,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "Database",
    "ConversationMessage",
    "DayPlanItem",
    "DayPlanRepository",
    "PLAN_STATUSES",
    "render_day_plan",
    "ConversationRepository",
    "DirectiveRepository",
    "InvalidTransition",
    "MigrationError",
    "PreferenceRepository",
    "STATUSES",
    "Task",
    "TaskRepository",
    "transaction",
    "utc_now",
    "apply_preferences",
]
