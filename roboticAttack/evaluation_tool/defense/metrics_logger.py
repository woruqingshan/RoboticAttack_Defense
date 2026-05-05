"""JSONL writer for recovery metric instrumentation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .recovery_metrics import safe_json_value


class JsonlMetricsLogger:
    """Small fault-tolerant JSONL metrics writer."""

    def __init__(self, path: str, enabled: bool = True, strict: bool = False) -> None:
        self.path = str(path)
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self._fh: Optional[Any] = None
        if self.enabled:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            safe_row = safe_json_value(row)
            self._fh.write(json.dumps(safe_row, ensure_ascii=False, allow_nan=False) + "\n")
            self._fh.flush()
        except Exception:
            if self.strict:
                raise

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

