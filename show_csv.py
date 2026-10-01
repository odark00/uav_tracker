#!/usr/bin/env python3
"""
Pretty-print a CSV in the terminal (rich table). Numeric columns are right-aligned; with --best, the best value in each
numeric column is highlighted (higher is better, except the columns listed in LOWER_IS_BETTER).

Run:       python show_csv.py                                   (runs/csv_gt/metrics_summary.csv)
           python show_csv.py runs/csv_gt/metrics_per_sequence.csv --sort success_auc --head 20
           python show_csv.py runs/csv_gt/metrics_summary.csv --cols tracker,success_auc,sa,mota,fps --best
           python show_csv.py runs/csv_gt/metrics_summary.csv --transpose
Notebook:  !python3 show_csv.py runs/csv_gt/metrics_per_sequence.csv      (full width, colours kept)
           from show_csv import show; show("runs/csv_gt/metrics_summary.csv", best=True)
"""
import argparse
import csv
from pathlib import Path

from rich.console import Console
from rich.table import Table

LOWER_IS_BETTER = {"fp", "fn", "idsw", "center_err", "norm_center_err"}


def to_num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def fmt(s):
    v = to_num(s)
    if v is None:
        return s
    if v.is_integer() and "." not in s:
        return f"{int(v):,}"
    return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e4 else f"{v:,.1f}"


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def show(path="runs/csv_gt/metrics_summary.csv", cols=None, sort=None, asc=False, head=None, filter=None,
         best=False, transpose=False, auto=True, width=None):
    """Print the CSV as a table. Also usable from a notebook: from show_csv import show; show(path, best=True)."""
    all_cols, rows = read_csv(path)
    if isinstance(cols, str):
        cols = [c.strip() for c in cols.split(",")]
    if cols:
        missing = [c for c in cols if c not in all_cols]
        if missing:
            raise ValueError(f"unknown columns {missing}; available: {', '.join(all_cols)}")
    else:
        cols = all_cols
    if filter:
        key, _, val = filter.partition("=")
        rows = [r for r in rows if val in r.get(key, "")]
    if sort:
        if sort not in all_cols:
            raise ValueError(f"unknown sort column {sort!r}")
        numeric = all(to_num(r[sort]) is not None for r in rows)
        key = (lambda r: to_num(r[sort])) if numeric else (lambda r: r[sort])
        rows.sort(key=key, reverse=numeric != asc)
    total = len(rows)
    if head:
        rows = rows[:head]

    numeric_cols = {c for c in cols if rows and all(to_num(r[c]) is not None for r in rows)}
    best_vals = {}
    if best:
        for c in numeric_cols:
            vals = [to_num(r[c]) for r in rows]
            if len(set(vals)) > 1:
                best_vals[c] = min(vals) if c in LOWER_IS_BETTER else max(vals)

    title = f"{Path(path).name}  [dim]({len(rows)}/{total} rows)[/dim]"
    table = build(cols, rows, numeric_cols, best_vals, title, transpose=transpose)
    console = Console(width=width)
    if width is None and console.is_jupyter:
        # notebook output scrolls sideways, so print at full width instead of rich's fixed Jupyter width
        console = Console(width=natural_width(console, table))
    elif width is None and not console.is_terminal:
        # piped / notebook "!" cell: no terminal width to fit, so print at full width and keep the colours
        console = Console(force_terminal=True, width=natural_width(console, table))
    elif not transpose and auto and natural_width(console, table) > console.width:
        flipped = build(cols, rows, numeric_cols, best_vals, title, transpose=True)
        if natural_width(console, flipped) <= console.width:
            table = flipped
        else:
            console.print("[yellow]table is wider than the terminal; narrow it with --cols or --head[/]")
    console.print(table)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="runs/csv_gt/metrics_summary.csv")
    ap.add_argument("--cols", help="comma-separated columns to show (default: all)")
    ap.add_argument("--sort", help="column to sort by (numeric columns sort descending)")
    ap.add_argument("--asc", action="store_true", help="sort ascending")
    ap.add_argument("--head", type=int, help="show only the first N rows")
    ap.add_argument("--filter", help="COL=VALUE, keep rows whose COL contains VALUE")
    ap.add_argument("--best", action="store_true", help="highlight the best value in each numeric column")
    ap.add_argument("--transpose", action="store_true", help="one row per column (handy for wide CSVs)")
    ap.add_argument("--no-auto", action="store_true", help="don't auto-transpose tables wider than the terminal")
    ap.add_argument("--width", type=int, help="output width (default: terminal width; full table width when piped)")
    args = ap.parse_args()
    try:
        show(args.path, cols=args.cols, sort=args.sort, asc=args.asc, head=args.head, filter=args.filter,
             best=args.best, transpose=args.transpose, auto=not args.no_auto, width=args.width)
    except ValueError as e:
        ap.error(str(e))


def natural_width(console, table):
    """Width the table would take with no terminal limit (console.measure clamps to the terminal)."""
    return console.measure(table, options=console.options.update_width(10**6)).maximum


def build(cols, rows, numeric_cols, best, title, transpose):
    if transpose:
        first = cols[0]
        table = Table(title=title, header_style="bold cyan", row_styles=["", "dim"])
        table.add_column(first, style="bold")
        for r in rows:
            table.add_column(r[first], justify="right")
        for c in cols[1:]:
            cells = []
            for r in rows:
                cell = fmt(r[c])
                if c in best and to_num(r[c]) == best[c]:
                    cell = f"[bold green]{cell}[/]"
                cells.append(cell)
            table.add_row(c, *cells)
    else:
        table = Table(title=title, header_style="bold cyan", row_styles=["", "on grey11"])
        for i, c in enumerate(cols):
            table.add_column(c, justify="right" if c in numeric_cols else "left",
                             style="bold" if i == 0 else None, no_wrap=True)
        for r in rows:
            cells = []
            for c in cols:
                cell = fmt(r[c])
                if c in best and to_num(r[c]) == best[c]:
                    cell = f"[bold green]{cell}[/]"
                cells.append(cell)
            table.add_row(*cells)
    return table


if __name__ == "__main__":
    main()
