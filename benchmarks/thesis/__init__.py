"""Thesis benchmark entry points and their shared CLI helpers."""

from __future__ import annotations

import argparse
import fnmatch
import re
from collections.abc import Mapping, Sequence
from typing import Any


def parse_memory_limit(value: str) -> int:
    """Parse a byte count or a human-readable size such as ``512M`` or ``13GiB``."""
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|b)?\s*", value, re.IGNORECASE
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected a size like 4096M, 32G, 16GiB, or raw bytes"
        )

    multipliers = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    bytes_value = int(number * multipliers[unit])
    if bytes_value <= 0:
        raise argparse.ArgumentTypeError("memory limit must be greater than zero")
    return bytes_value


def resolve_names(
    selectors: Sequence[str] | None, available: Mapping[str, Any]
) -> list[str]:
    """Resolve exact names, shell wildcards, or regular expressions."""
    names = sorted(available)
    if not selectors:
        return names

    selected: set[str] = set()
    invalid: list[str] = []
    for selector in selectors:
        if selector in available:
            selected.add(selector)
            continue
        matches = [name for name in names if fnmatch.fnmatchcase(name, selector)]
        if not matches:
            try:
                pattern = re.compile(selector)
            except re.error as exc:
                invalid.append(f"{selector!r} (invalid regex: {exc})")
                continue
            matches = [name for name in names if pattern.search(name)]
        if matches:
            selected.update(matches)
        else:
            invalid.append(f"{selector!r} (no matches)")

    if invalid:
        raise ValueError(
            f"Unknown selector(s): {', '.join(invalid)}. Available: {', '.join(names)}"
        )
    return sorted(selected)


def validate_common_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate resource limits and an optional inclusive n range."""
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.nr_seeds <= 0:
        parser.error("--nr-seeds must be greater than zero")
    if getattr(args, "nmin", None) is not None and args.nmin < 1:
        parser.error("--nmin must be at least one")
    if getattr(args, "nmax", None) is not None and args.nmax < 1:
        parser.error("--nmax must be at least one")
    if (
        getattr(args, "nmin", None) is not None
        and getattr(args, "nmax", None) is not None
        and args.nmin > args.nmax
    ):
        parser.error("--nmin cannot be greater than --nmax")
