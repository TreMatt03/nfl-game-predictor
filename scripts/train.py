"""Fit the final model on all available history and save it.

Separated from prediction on purpose. Training needs twenty seasons of
play-by-play; predicting one week needs the model plus a couple of recent
seasons to refresh form. Keeping them apart is what lets the scheduled job run
in under a minute instead of re-downloading the archive every Tuesday.
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

from nflpred import pipeline, quarterback
from nflpred.explain import global_importance
from nflpred.model import fit_final

MODEL_PATH = ROOT / "models" / "model.joblib"
CAREER_PATH = ROOT / "models" / "qb_career.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    games, team_games, _, starters = pipeline.build(refresh=args.refresh)
    pipeline.save(games, team_games)

    model = fit_final(games)
    joblib.dump(model, MODEL_PATH)

    # Forecasting reads only recent seasons, so it cannot count career length
    # for itself. Record it here, where the whole archive is loaded.
    careers = quarterback.player_ratings(starters, career_starts=pd.Series(dtype=float))
    careers[["career_starts"]].to_csv(CAREER_PATH)

    importance = global_importance(model)
    importance.to_csv(REPORTS / "importance.csv", index=False)

    print(f"\ntrained on {len(games)} games, seasons "
          f"{games.season.min()}-{games.season.max()}")
    print(f"saved {MODEL_PATH.relative_to(ROOT)}\n")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
