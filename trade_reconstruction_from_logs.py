import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


ENTRY_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z).*ENTRY CONFIRMED: (?P<side>YES|NO) @ (?P<px>[0-9.]+) size=(?P<size>[0-9.]+)"
)
EXIT_PLACED_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z).*Exit order placed \((?P<oid>0x[a-fA-F0-9]+)\) for (?P<reason>[A-Z0-9_]+) — sell_size=(?P<sell_size>[0-9.]+) pos_size=(?P<pos_size>[0-9.]+)"
)
PENDING_FILLED_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z).*PENDING ORDER FILLED: (?P<oid>0x[a-fA-F0-9]+) \((?P<shares>[0-9.]+) shares\)"
)
FOK_RECON_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z).*FOK_IMMEDIATE_RECONCILE: order (?P<oidprefix>0x[a-fA-F0-9]+) filled (?P<shares>[0-9.]+)"
)
MARKET_URL_RE = re.compile(r"Polymarket: https://polymarket\.com/event/(?P<slug>[a-zA-Z0-9\-]+)")
WINDOW_RE = re.compile(r"window=(?P<start>\d+)-(?P<end>\d+)")
STRIKE_PRICE_RE = re.compile(r"Strike=(?P<strike>[0-9.]+) \| Price=(?P<price>[0-9.]+)")


def parse_ts(ts: str) -> datetime:
    # timestamps are already Zulu
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


@dataclass
class ExitAttempt:
    ts: datetime
    reason: str
    order_id: str
    sell_size: float
    pos_size: float
    filled_size: Optional[float] = None


@dataclass
class Trade:
    idx: int
    entry_ts: datetime
    slug: Optional[str]
    window_start: Optional[int]
    side: str
    entry_px: float
    entry_size: float
    entry_oid: Optional[str] = None
    exits: List[ExitAttempt] = field(default_factory=list)


def reconstruct(log_path: str) -> List[Trade]:
    trades: List[Trade] = []
    oid_to_trade: Dict[str, Trade] = {}
    oid_prefix_to_exit: Dict[str, ExitAttempt] = {}

    current_slug: Optional[str] = None
    current_window_start: Optional[int] = None

    trade_idx = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = MARKET_URL_RE.search(line)
            if m:
                current_slug = m.group("slug")

            m = WINDOW_RE.search(line)
            if m:
                current_window_start = int(m.group("start"))

            m = EXIT_PLACED_RE.search(line)
            if m:
                ts = parse_ts(m.group("ts"))
                ex = ExitAttempt(
                    ts=ts,
                    reason=m.group("reason"),
                    order_id=m.group("oid"),
                    sell_size=float(m.group("sell_size")),
                    pos_size=float(m.group("pos_size")),
                )
                # attach to last open trade
                if trades:
                    trades[-1].exits.append(ex)
                    oid_to_trade[ex.order_id] = trades[-1]
                    # also map prefix used by FOK reconcile line
                    oid_prefix_to_exit[ex.order_id[:10]] = ex
                continue

            m = ENTRY_RE.search(line)
            if m:
                trade_idx += 1
                ts = parse_ts(m.group("ts"))
                tr = Trade(
                    idx=trade_idx,
                    entry_ts=ts,
                    slug=current_slug,
                    window_start=current_window_start,
                    side=m.group("side"),
                    entry_px=float(m.group("px")),
                    entry_size=float(m.group("size")),
                )
                trades.append(tr)
                continue

            m = PENDING_FILLED_RE.search(line)
            if m:
                oid = m.group("oid")
                filled = float(m.group("shares"))
                tr = oid_to_trade.get(oid)
                if tr and tr.exits:
                    # match to most recent exit with same oid
                    for ex in reversed(tr.exits):
                        if ex.order_id == oid and ex.filled_size is None:
                            ex.filled_size = filled
                            break
                continue

            m = FOK_RECON_RE.search(line)
            if m:
                oidp = m.group("oidprefix")
                filled = float(m.group("shares"))
                ex = oid_prefix_to_exit.get(oidp)
                if ex and ex.filled_size is None:
                    ex.filled_size = filled
                continue

    return trades


def summarize(trades: List[Trade]) -> List[dict]:
    rows = []
    for tr in trades:
        first_exit = tr.exits[0] if tr.exits else None
        last_exit = tr.exits[-1] if tr.exits else None
        exit_ts = last_exit.ts if last_exit else None
        exit_reason = last_exit.reason if last_exit else None
        exit_filled = None
        if last_exit and last_exit.filled_size is not None:
            exit_filled = last_exit.filled_size

        hold_sec = (exit_ts - tr.entry_ts).total_seconds() if exit_ts else None

        rows.append(
            {
                "trade": tr.idx,
                "entry_ts": tr.entry_ts.isoformat(),
                "slug": tr.slug,
                "window_start": tr.window_start,
                "side": tr.side,
                "entry_px": tr.entry_px,
                "entry_size": tr.entry_size,
                "exit_ts": exit_ts.isoformat() if exit_ts else None,
                "exit_reason": exit_reason,
                "exit_sell_size": last_exit.sell_size if last_exit else None,
                "exit_filled_size": exit_filled,
                "hold_seconds": hold_sec,
                "exit_attempts": len(tr.exits),
            }
        )
    return rows


def write_csv(rows: List[dict], out_path: str):
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path")
    ap.add_argument("--out", default="trade_reconstruction.csv")
    args = ap.parse_args()

    trades = reconstruct(args.log_path)
    rows = summarize(trades)
    write_csv(rows, args.out)

    print(f"trades_found={len(trades)}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
