from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from stock_screener.config import Settings
from stock_screener.runner import run_screen
from stock_screener.update import update_daily

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def update(
    start: str = typer.Option(..., help="Start date YYYYMMDD"),
    end: str = typer.Option(..., help="End date YYYYMMDD"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
) -> None:
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    update_daily(settings=settings, start=start, end=end)


@app.command()
def run(
    date: str = typer.Option(..., help="Trade date YYYYMMDD"),
    combo: str = typer.Option("and", help="and|or"),
    out: Path = typer.Option(..., help="Output file path (.csv or .json)"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
    lookback_days: int = typer.Option(200, help="Calendar days lookback to compute indicators"),
    rules: Optional[str] = typer.Option(None, help="Comma-separated rule names (default: built-in)"),
) -> None:
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    results = run_screen(settings=settings, date=date, combo=combo, lookback_days=lookback_days, rules=rules)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        results.to_csv(out, index=False)
        return
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps(results.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    raise typer.BadParameter("out must end with .csv or .json")


if __name__ == "__main__":
    app()

