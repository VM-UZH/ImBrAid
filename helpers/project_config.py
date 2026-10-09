"""Load one tracked project configuration for notebooks and batch helpers.

All relative paths are resolved from the directory containing the JSON file.
External absolute paths remain unchanged.  Keeping resolution here avoids a
different ``data_root`` or output directory in every notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Project configuration with paths relative to its JSON file."""

    source_path: Path
    values: dict[str, Any]

    @property
    def project(self) -> str:
        return str(self.values["project"])

    def value(self, section: str, key: str) -> Any:
        return self.values[section][key]

    def path(self, section: str, key: str) -> Path:
        """Resolve a required path; relative values are config-directory based."""

        raw_value = self.value(section, key)
        expanded = Path(os.path.expandvars(raw_value)).expanduser()
        if not expanded.is_absolute():
            expanded = self.source_path.parent / expanded
        return expanded.resolve()


def load_project_config(path: str | Path) -> ProjectConfig:
    """Read a project JSON configuration."""

    source_path = Path(path).expanduser().resolve()
    values = json.loads(source_path.read_text(encoding="utf-8"))
    return ProjectConfig(source_path=source_path, values=values)


__all__ = ["ProjectConfig", "load_project_config"]
