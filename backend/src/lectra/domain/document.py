from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    """Represents an original source used by Lectra."""

    id: str
    name: str
    source_type: str
    metadata: dict[str, Any]
