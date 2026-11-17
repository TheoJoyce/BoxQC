"""Identifiers and timestamps shared by the dataset store and the inspection log."""

from __future__ import annotations

import re
import secrets
from datetime import datetime

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def now_iso() -> str:
    """Local time with UTC offset, to the second (e.g. ``2026-11-12T16:01:14-05:00``)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_id() -> str:
    """A sortable id: timestamp plus a short random suffix (``20261112-160114-a3f9``)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def check_id(item_id: str) -> str:
    """Return ``item_id`` if it is safe to use in a file name, else raise ``ValueError``."""
    if not _ID_RE.match(item_id):
        raise ValueError(f"Invalid id: {item_id!r}")
    return item_id
