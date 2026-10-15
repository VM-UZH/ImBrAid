"""Load project configuration and resolve paths relative to its JSON file."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    values: dict[str, Any]

    @property
    def project(self) -> str:
        return str(self.values["project"])

    def value(self, section: str, key: str) -> Any:
        return self.values[section][key]

    def path(self, section: str, key: str) -> Path:
        raw_value = self.value(section, key)
        expanded = Path(os.path.expandvars(raw_value)).expanduser()
        if not expanded.is_absolute():
            expanded = self.source_path.parent / expanded
        return expanded.resolve()


def load_project_config(path: str | Path) -> ProjectConfig:
    source_path = Path(path).expanduser().resolve()
    values = json.loads(source_path.read_text(encoding="utf-8"))
    return ProjectConfig(source_path=source_path, values=values)


__all__ = ["ProjectConfig", "load_project_config"]
