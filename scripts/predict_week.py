"""Predict the upcoming week and rebuild the published page.

This is what the scheduled GitHub Action runs. By default it loads the model
committed by scripts/train.py and pulls only the last few seasons of
play-by-play -- enough to refresh team and quarterback form, which is all a
forecast needs. Pass --retrain to rebuild the model from full history instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflpred import data, pipeline, site, tracking
from nflpred.model import fit_final
from nflpred.predict import predict_slate

MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"

# Team form would be happy with three seasons -- an 8-game halflife forgets
# faster than that. Quarterback ratings drive the window instead: a backup may
# have only a handful of starts spread over several years, and a short window
# keeps whichever of them happen to fall inside it. Cutting to three seasons
# left one quarterback with 7 career starts rated the second best passer in the
# league, because it kept his good season and dropped his bad one. Six seasons
# reproduces full-history ratings for every active starter.
FORM_SEASONS = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    parser.add_argument("--retrain", action="store_true", help="refit on all history")
    parser.add_argument("--season", type=int, help="override season")
    parser.add_argument("--week", type=int, help="override week")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    schedules = data.load_schedules(refresh=args.refresh)
    last_played = int(schedules[schedules["result"].notna()]["season"].max())

    if args.retrain:
        games, team_games, elo_now, starters = pipeline.build(refresh=args.refresh)
        model = fit_final(games)
    else:
        if not MODEL_PATH.exists():
            raise SystemExit("no saved model; run scripts/train.py first")
        # Elo is rebuilt from the full schedule either way -- it needs only
        # scores, which are cheap. Only the play-by-play window is trimmed.
        _, team_games, elo_now, starters = pipeline.build(
            start_season=last_played - FORM_SEASONS + 1, refresh=args.refresh
        )
        model = joblib.load(MODEL_PATH)

    slate = predict_slate(
        model, schedules, team_games, elo_now, starters, args.season, args.week
    )

    backtest_path = REPORTS / "backtest.csv"
    if not backtest_path.exists():
        raise SystemExit("run scripts/backtest.py first to produce reports/backtest.csv")

    # Log first, so this week's slate is on the record before it is published.
    tracking.log_predictions(slate)

    scored = tracking.score_history(tracking.load_log(), schedules)
    out = site.render(slate, pd.read_csv(backtest_path), scored=scored)

    print(f"\n{slate.season} Week {slate.week} -- {len(slate.games)} games\n")
    for _, row in slate.games.iterrows():
        top = row["drivers"][0]["factor"] if row["drivers"] else ""
        print(
            f"  {row['away_team']:>3} at {row['home_team']:<3}  "
            f"{row['home_win_prob'] * 100:5.1f}% home   "
            f"pick {row['favorite']:<3}  ({top})"
        )
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
