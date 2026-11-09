"""Token prices, supplied by you. agent-otel ships no prices: they change, and they are yours to date.

A table maps model names (globs allowed) to prices per million tokens:

    {"currency": "USD",
     "models": {"llama3.1:*": {"input": 0, "output": 0},
                "my-hosted-model-*": {"input": 0.5, "output": 1.5, "cache_read": 0.05}}}

The first matching entry wins, in the order written.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PriceError(ValueError):
    pass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None = None


class Prices:
    def __init__(self, table: dict[str, Any]):
        if not isinstance(table, dict) or not isinstance(table.get("models"), dict):
            raise PriceError('a price table is {"currency": "...", "models": {"<model glob>": {"input": x, "output": y}}}')
        self.currency = str(table.get("currency", "USD"))
        self.entries: list[tuple[str, Price]] = []
        for pattern, p in table["models"].items():
            if not isinstance(p, dict) or not {"input", "output"} <= set(p):
                raise PriceError(f"price for {pattern!r} needs input and output (per million tokens)")
            try:
                price = Price(float(p["input"]), float(p["output"]),
                              float(p["cache_read"]) if p.get("cache_read") is not None else None)
            except (TypeError, ValueError):
                raise PriceError(f"price for {pattern!r} must be numbers") from None
            if min(price.input, price.output, price.cache_read or 0) < 0:
                raise PriceError(f"price for {pattern!r} cannot be negative")
            self.entries.append((str(pattern), price))

    @classmethod
    def load(cls, source: str | Path | dict | "Prices") -> "Prices":
        if isinstance(source, Prices):
            return source
        if isinstance(source, dict):
            return cls(source)
        try:
            return cls(json.loads(Path(source).read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            raise PriceError(f"{source}: {e}") from None

    def lookup(self, model: str) -> Price | None:
        for pattern, price in self.entries:
            if fnmatch.fnmatchcase(model, pattern):
                return price
        return None

    def cost(self, model: str, input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0) -> float | None:
        """Cost of one call, or None when the model is not in the table."""
        p = self.lookup(model)
        if p is None:
            return None
        cached = min(cache_read_tokens, input_tokens) if p.cache_read is not None else 0
        fresh = input_tokens - cached
        total = fresh * p.input + output_tokens * p.output + cached * (p.cache_read or 0)
        return round(total / 1_000_000, 10)
