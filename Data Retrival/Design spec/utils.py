
"""
utils.py

Shared utilities for Design Spec pipeline.
"""

import json
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

def norm_text(x: Any) -> str:
    """
    Normalize text: handle None, HTML escapes, whitespace.
    """
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x)
    if not isinstance(x, str):
        return ""
    # registry often has HTML escapes like &gt;
    s = unescape(x)
    # normalize whitespace
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_bullets(text: str) -> List[str]:
    """
    Split common registry multiline fields into a list.
    """
    t = norm_text(text)
    if not t:
        return []
    # prefer newline splitting if present
    if "\n" in t:
        items = []
        for line in t.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-•*\u2022]\s*", "", line).strip()
            if line:
                items.append(line)
        return items

    # fallback: semicolon / comma separated
    parts = [p.strip() for p in re.split(r";\s*", t) if p.strip()]
    return parts if len(parts) > 1 else [t]

def parse_bool(text: Any) -> Optional[bool]:
    """
    Parse boolean-like text.
    """
    t = norm_text(text).lower()
    if not t:
        return None
    if t in ("yes", "y", "true", "1"):
        return True
    if t in ("no", "n", "false", "0"):
        return False
    return None

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
