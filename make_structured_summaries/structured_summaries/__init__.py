"""Structured book summary helpers."""

from .models import BookRecord
from .pipeline import SummaryConfig, summarize_book
from .preread import PrereadConfig, build_preread_brief
from .takeout import load_catalog, scan_takeout_root, write_catalog

__all__ = [
    "BookRecord",
    "PrereadConfig",
    "SummaryConfig",
    "build_preread_brief",
    "load_catalog",
    "scan_takeout_root",
    "summarize_book",
    "write_catalog",
]
