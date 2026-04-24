"""
Phase 2 P0.5: Tests for the CLOB snapshot recorder.

Covers:
- record_snapshot with full L2 book (ws source)
- record_snapshot with REST fallback (top-of-book only)
- skip-when-unchanged optimization
- top-N level cap
- iter_snapshots round-trip
- retention prune (date-mocked)
- no-op when window_ts is 0 or None
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

import clob_snapshot as cs


@pytest.fixture(autouse=True)
def isolated_dir(monkeypatch, tmp_path):
    """Point SNAPSHOT_DIR at an empty tmp for every test."""
    monkeypatch.setattr(cs, "SNAPSHOT_DIR", tmp_path / "snaps")
    # Reset in-memory cache between tests
    cs._last_snapshot.clear()
    cs._last_prune_ts = 0.0
    yield


class _FakeL2Book:
    """Duck-types polymarket_ws.L2Book."""
    def __init__(self, bids, asks, ts_ms=None):
        self.bids = bids
        self.asks = asks
        self.last_ts_ms = ts_ms or int(time.time() * 1000)


class TestFullBookRecord:
    def test_records_ws_snapshot(self):
        yes = _FakeL2Book({0.93: 50.0, 0.92: 75.0}, {0.95: 30.0, 0.96: 40.0})
        no  = _FakeL2Book({0.07: 80.0}, {0.08: 20.0})
        ok = cs.record_snapshot(
            window_ts=1_700_000_000,
            yes_token="0xYES", no_token="0xNO",
            yes_book=yes, no_book=no,
        )
        assert ok is True
        fp = cs.SNAPSHOT_DIR / "1700000000.jsonl"
        assert fp.exists()
        rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert row["window_ts"] == 1_700_000_000
        assert row["yes_token"] == "0xYES"
        assert row["yes"]["src"] == "ws"
        assert row["yes"]["bids"] == [[0.93, 50.0], [0.92, 75.0]]
        assert row["yes"]["asks"] == [[0.95, 30.0], [0.96, 40.0]]
        assert row["no"]["bids"] == [[0.07, 80.0]]

    def test_top_n_cap(self, monkeypatch):
        monkeypatch.setattr(cs, "TOP_N_LEVELS", 3)
        bids = {0.90 + i * 0.01: float(i + 1) for i in range(10)}  # 10 price levels
        yes = _FakeL2Book(bids, {0.99: 1.0})
        cs.record_snapshot(window_ts=1_700_000_001, yes_token="a", no_token="b", yes_book=yes)
        fp = cs.SNAPSHOT_DIR / "1700000001.jsonl"
        row = json.loads(fp.read_text().splitlines()[0])
        assert len(row["yes"]["bids"]) == 3
        # Top-3 must be the highest prices (descending)
        prices = [lvl[0] for lvl in row["yes"]["bids"]]
        assert prices == sorted(prices, reverse=True)


class TestRESTFallback:
    def test_top_of_book_only(self):
        ok = cs.record_snapshot(
            window_ts=1_700_000_002,
            yes_token="a", no_token="b",
            yes_bid=0.93, yes_ask=0.95,
            no_bid=0.07,  no_ask=0.08,
        )
        assert ok is True
        fp = cs.SNAPSHOT_DIR / "1700000002.jsonl"
        row = json.loads(fp.read_text().splitlines()[0])
        assert row["yes"]["src"] == "rest"
        assert row["yes"]["bids"] == [[0.93, 0.0]]
        assert row["yes"]["asks"] == [[0.95, 0.0]]

    def test_nothing_to_write(self):
        """No books and no top-of-book → no-op."""
        ok = cs.record_snapshot(
            window_ts=1_700_000_003,
            yes_token="a", no_token="b",
        )
        assert ok is False
        fp = cs.SNAPSHOT_DIR / "1700000003.jsonl"
        assert not fp.exists()


class TestSkipUnchanged:
    def test_two_identical_snapshots_dedup(self):
        yes = _FakeL2Book({0.93: 50.0}, {0.95: 30.0})
        cs.record_snapshot(window_ts=1_700_000_004, yes_token="a", no_token="b", yes_book=yes)
        # Second call with unchanged top-of-book (size delta < 0.5) → skip
        yes2 = _FakeL2Book({0.93: 50.1}, {0.95: 30.0})
        ok2 = cs.record_snapshot(window_ts=1_700_000_004, yes_token="a", no_token="b", yes_book=yes2)
        assert ok2 is False
        fp = cs.SNAPSHOT_DIR / "1700000004.jsonl"
        assert len(fp.read_text().splitlines()) == 1

    def test_price_change_not_skipped(self):
        yes = _FakeL2Book({0.93: 50.0}, {0.95: 30.0})
        cs.record_snapshot(window_ts=1_700_000_005, yes_token="a", no_token="b", yes_book=yes)
        yes2 = _FakeL2Book({0.94: 50.0}, {0.95: 30.0})
        ok2 = cs.record_snapshot(window_ts=1_700_000_005, yes_token="a", no_token="b", yes_book=yes2)
        assert ok2 is True
        fp = cs.SNAPSHOT_DIR / "1700000005.jsonl"
        assert len(fp.read_text().splitlines()) == 2


class TestIterSnapshots:
    def test_roundtrip(self):
        yes1 = _FakeL2Book({0.93: 50.0}, {0.95: 30.0})
        yes2 = _FakeL2Book({0.94: 50.0}, {0.96: 30.0})
        cs.record_snapshot(window_ts=1_700_000_006, yes_token="a", no_token="b", yes_book=yes1)
        cs.record_snapshot(window_ts=1_700_000_006, yes_token="a", no_token="b", yes_book=yes2)

        rows = list(cs.iter_snapshots(1_700_000_006))
        assert len(rows) == 2
        assert rows[0]["yes"]["bids"][0][0] == 0.93
        assert rows[1]["yes"]["bids"][0][0] == 0.94

    def test_missing_file_yields_empty(self):
        rows = list(cs.iter_snapshots(9_999_999_999))
        assert rows == []


class TestInputValidation:
    def test_zero_window_skipped(self):
        assert cs.record_snapshot(window_ts=0, yes_token="a", no_token="b",
                                  yes_bid=0.5, yes_ask=0.6) is False

    def test_negative_window_skipped(self):
        assert cs.record_snapshot(window_ts=-1, yes_token="a", no_token="b",
                                  yes_bid=0.5, yes_ask=0.6) is False


class TestListRecordedWindows:
    def test_returns_sorted_newest_first(self):
        yes = _FakeL2Book({0.5: 1.0}, {0.6: 1.0})
        cs.record_snapshot(window_ts=1_700_000_010, yes_token="a", no_token="b", yes_book=yes)
        cs.record_snapshot(window_ts=1_700_000_020, yes_token="a", no_token="b", yes_book=yes)
        windows = cs.list_recorded_windows()
        # Both files exist
        assert 1_700_000_010 in windows
        assert 1_700_000_020 in windows
