import json

import pytest

from agent_otel.prices import PriceError, Prices


def test_first_match_and_globs():
    p = Prices({"models": {"a-mini": {"input": 1, "output": 2}, "a-*": {"input": 3, "output": 4}}})
    assert p.cost("a-mini", 1_000_000, 0) == 1.0
    assert p.cost("a-large", 0, 1_000_000) == 4.0
    assert p.cost("b", 10, 10) is None
    assert p.currency == "USD"


def test_cache_reads_never_exceed_input():
    p = Prices({"models": {"m": {"input": 2, "output": 0, "cache_read": 1}}})
    assert p.cost("m", 100, 0, cache_read_tokens=500) == pytest.approx(100 / 1e6)


@pytest.mark.parametrize("bad", [{}, {"models": []}, {"models": {"m": {"input": 1}}},
                                 {"models": {"m": {"input": "x", "output": 1}}},
                                 {"models": {"m": {"input": -1, "output": 1}}}])
def test_bad_tables(bad):
    with pytest.raises(PriceError):
        Prices(bad)


def test_load_from_file(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"currency": "GBP", "models": {"*": {"input": 0, "output": 0}}}))
    assert Prices.load(str(f)).currency == "GBP"
    with pytest.raises(PriceError):
        Prices.load(str(tmp_path / "missing.json"))
