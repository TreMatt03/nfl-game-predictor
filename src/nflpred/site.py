"""Render the weekly slate to a self-contained page for GitHub Pages.

Deliberately templated with plain string formatting rather than a template
engine: the output is one page, and a dependency-free build is one less thing
that can break the scheduled job at 6am on a Tuesday.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import tracking
from .predict import Slate

SITE_DIR = Path(__file__).resolve().parents[2] / "site"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Week {week} Predictions</title>
<style>
:root {{
  --bg: #fbfaf9; --card: #ffffff; --ink: #1a1a18; --muted: #6b6b66;
  --line: #e6e4e0; --accent: #2d6cdf; --away: #c2410c; --bar: #eeece8;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14140f; --card: #1c1c18; --ink: #f0eee9; --muted: #9a978f;
    --line: #2e2e28; --accent: #6ea8ff; --away: #fb923c; --bar: #26261f;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
}}
.wrap {{ max-width: 860px; margin: 0 auto; padding: 40px 20px 72px; }}
header {{ border-bottom: 1px solid var(--line); padding-bottom: 24px; margin-bottom: 8px; }}
h1 {{ font-size: 1.9rem; margin: 0 0 6px; letter-spacing: -0.02em; }}
.sub {{ color: var(--muted); font-size: 0.95rem; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 26px; margin: 22px 0 0; }}
.metric .v {{ font-size: 1.35rem; font-weight: 640; font-variant-numeric: tabular-nums; }}
.metric .k {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.06em; }}
.game {{ border: 1px solid var(--line); background: var(--card); border-radius: 10px;
  padding: 18px 20px; margin-top: 14px; }}
.matchup {{ display: flex; justify-content: space-between; align-items: baseline;
  gap: 12px; }}
.teams {{ font-size: 1.12rem; font-weight: 620; }}
.kick {{ color: var(--muted); font-size: 0.83rem; white-space: nowrap; }}
.bar {{ height: 9px; background: var(--bar); border-radius: 5px; overflow: hidden;
  margin: 13px 0 9px; display: flex; }}
.bar .h {{ background: var(--accent); }}
.bar .a {{ background: var(--away); }}
.odds {{ display: flex; justify-content: space-between; font-size: 0.87rem;
  font-variant-numeric: tabular-nums; }}
.odds .home {{ color: var(--accent); font-weight: 600; }}
.odds .away {{ color: var(--away); font-weight: 600; }}
.why {{ margin-top: 13px; padding-top: 12px; border-top: 1px dashed var(--line);
  font-size: 0.85rem; color: var(--muted); }}
.why span {{ display: inline-block; margin-right: 14px; }}
.why b {{ color: var(--ink); font-weight: 600; }}
.qb {{ font-size: 0.82rem; color: var(--muted); margin-top: 4px; }}
h2 {{ font-size: 1.15rem; margin: 44px 0 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem;
  font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.05em; }}
tr.mine td {{ font-weight: 680; }}
.scroll {{ overflow-x: auto; }}
.hit {{ color: #15803d; font-weight: 650; }}
.miss {{ color: #b91c1c; font-weight: 650; }}
@media (prefers-color-scheme: dark) {{
  .hit {{ color: #4ade80; }}
  .miss {{ color: #f87171; }}
}}
.note {{ color: var(--muted); font-size: 0.85rem; margin: 6px 0 14px; }}
footer {{ margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 0.82rem; }}
a {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>NFL {season} &middot; Week {week}</h1>
  <div class="sub">Win probabilities from team efficiency, Elo, and quarterback
    form. Trained only on games played before each prediction.</div>
  <div class="metrics">{metrics}</div>
</header>

{games}

{track_record}

<h2>How the model was tested</h2>
<div class="sub" style="margin-bottom:12px">Walk-forward: every season is
predicted by a model trained only on earlier seasons. {backtest_games} games.</div>
<div class="scroll">{table}</div>

<footer>
  Built from public <a href="https://github.com/nflverse">nflverse</a> data.
  Updated {updated}. Predictions are for interest, not for wagering.
</footer>
</div>
</body>
</html>
"""


def _metric_block(label: str, value: str) -> str:
    return f'<div class="metric"><div class="v">{value}</div><div class="k">{label}</div></div>'


def _game_card(row: pd.Series) -> str:
    home, away = html.escape(row["home_team"]), html.escape(row["away_team"])
    hp = row["home_win_prob"] * 100
    ap = row["away_win_prob"] * 100

    kick = row["gameday"].strftime("%a %d %b") if pd.notna(row["gameday"]) else ""

    drivers = "".join(
        f'<span><b>{html.escape(d["factor"])}</b> &rarr; '
        f'{html.escape(row[d["favors"] + "_team"])}</span>'
        for d in row["drivers"]
    )

    qb_line = ""
    if pd.notna(row.get("home_qb")) and pd.notna(row.get("away_qb")):
        qb_line = (
            f'<div class="qb">QB: {html.escape(str(row["away_qb"]))} '
            f'&middot; {html.escape(str(row["home_qb"]))}</div>'
        )

    return f"""<div class="game">
  <div class="matchup">
    <div class="teams">{away} <span style="color:var(--muted)">at</span> {home}</div>
    <div class="kick">{kick}</div>
  </div>
  {qb_line}
  <div class="bar">
    <div class="a" style="width:{ap:.1f}%"></div>
    <div class="h" style="width:{hp:.1f}%"></div>
  </div>
  <div class="odds">
    <span class="away">{away} {ap:.0f}%</span>
    <span class="home">{hp:.0f}% {home}</span>
  </div>
  <div class="why">{drivers}</div>
</div>"""


def _track_record(scored: pd.DataFrame) -> str:
    """Live results section, or a placeholder before anything has settled."""
    summary = tracking.summarise(scored)

    if not summary["games"]:
        return (
            '<h2>Track record</h2>\n<div class="note">No predictions have been '
            "settled yet. Every forecast is logged when published and scored "
            "here once the game is played.</div>"
        )

    tiles = "".join(
        [
            _metric_block("Record", f"{summary['wins']}&ndash;{summary['losses']}"),
            _metric_block("Accuracy", f"{summary['accuracy'] * 100:.1f}%"),
            _metric_block("Log loss", f"{summary['log_loss']:.3f}"),
            _metric_block("Brier", f"{summary['brier']:.3f}"),
        ]
    )

    recent = tracking.latest_week(scored)
    rows = []
    for _, r in recent.iterrows():
        picked = r["favorite"]
        hit = r["correct"] == 1
        confidence = r["confidence"] * 100
        rows.append(
            f"<tr><td>{html.escape(r['away_team'])} at "
            f"{html.escape(r['home_team'])}</td>"
            f"<td>{html.escape(str(picked))} {confidence:.0f}%</td>"
            f"<td>{html.escape(str(r['winner']))}</td>"
            f'<td class="{"hit" if hit else "miss"}">'
            f'{"correct" if hit else "wrong"}</td></tr>'
        )

    week_label = ""
    if not recent.empty:
        week_label = (
            f"Week {int(recent.iloc[0]['week'])}, "
            f"{int(recent.iloc[0]['season'])}"
        )

    return f"""<h2>Track record</h2>
<div class="note">Live results. Each prediction is logged when it is published
and never revised, so this is what was actually called beforehand.</div>
<div class="metrics" style="margin-bottom:18px">{tiles}</div>
<h2 style="margin-top:28px;font-size:1rem">{week_label}</h2>
<div class="scroll"><table><thead><tr>
<th>Game</th><th>Pick</th><th>Winner</th><th>Result</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""


def _backtest_table(backtest: pd.DataFrame) -> str:
    header = "".join(
        f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in backtest.columns
    )
    rows = []
    for _, r in backtest.iterrows():
        cls = ' class="mine"' if "Logistic (full)" == r["model"] else ""
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in r)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render(
    slate: Slate,
    backtest: pd.DataFrame,
    out_dir: Path = SITE_DIR,
    scored: pd.DataFrame | None = None,
) -> Path:
    """Write index.html and predictions.json into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    mine = backtest[backtest["model"] == "Logistic (full)"].iloc[0]
    vegas = backtest[backtest["model"] == "Vegas moneyline"]

    metrics = [
        _metric_block("Accuracy", f"{mine['accuracy'] * 100:.1f}%"),
        _metric_block("Log loss", f"{mine['log_loss']:.3f}"),
        _metric_block("Brier", f"{mine['brier']:.3f}"),
    ]
    if not vegas.empty:
        metrics.append(
            _metric_block("Vegas log loss", f"{vegas.iloc[0]['log_loss']:.3f}")
        )
    metrics.append(_metric_block("Games", str(len(slate.games))))

    page = PAGE.format(
        season=slate.season,
        week=slate.week,
        metrics="".join(metrics),
        games="".join(_game_card(r) for _, r in slate.games.iterrows()),
        table=_backtest_table(backtest),
        track_record=_track_record(
            scored if scored is not None else pd.DataFrame()
        ),
        backtest_games=int(mine["games"]),
        updated=datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
    )

    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")

    payload = {
        "season": slate.season,
        "week": slate.week,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "games": [
            {
                "game_id": r["game_id"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "kickoff": r["gameday"].strftime("%Y-%m-%d")
                if pd.notna(r["gameday"])
                else None,
                "home_win_prob": round(float(r["home_win_prob"]), 4),
                "away_win_prob": round(float(r["away_win_prob"]), 4),
                "favorite": r["favorite"],
                "drivers": r["drivers"],
            }
            for _, r in slate.games.iterrows()
        ],
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    return index
