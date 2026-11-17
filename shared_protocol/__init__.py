"""Metadata-only foundations for the primary fixed experimental protocol.

No adapters, features, models, or legacy experiment modules are loaded here.
"""

from .manifest import load_manifest, validate_manifest
from .splits import create_split, development_records, load_split, save_split

__all__ = [
    "load_manifest", "validate_manifest", "create_split", "development_records",
    "load_split", "save_split",
]
