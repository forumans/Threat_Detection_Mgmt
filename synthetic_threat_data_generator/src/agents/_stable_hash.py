"""
Deterministic seed derivation, shared by every agent that seeds randomness
from an ID string (Persona Generator, Metadata Generator, TTS Engine). Not
itself an agent -- the leading underscore marks it as internal plumbing.

Python's builtin `hash()` on strings is randomized per process
(PYTHONHASHSEED, on by default, for security reasons) -- fine for dict
lookups, useless for a seed that's supposed to make the same
`Configuration.seed` reproduce the exact same dataset across separate runs
(the architecture doc's "version everything" principle, §3). This uses
SHA-256 instead, which is stable across processes, machines, and time.
"""

from __future__ import annotations

import hashlib


def stable_seed(*parts: str) -> int:
    """A deterministic 32-bit seed derived from one or more string parts."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
