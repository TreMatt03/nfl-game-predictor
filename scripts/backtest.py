"""Walk-forward evaluation against the baselines that matter.

Run with ``python scripts/backtest.py``. Writes a metrics table and a
calibration plot into ``reports/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflpred import pipeline
from nflpred.model import score, walk_forward

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIRST_TEST_SEASON = 2012


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REPORTS.mkdir(exist_ok=True)

    games, team_games, _, _ = pipeline.build(refresh=args.refresh)
    pipeline.save(games, team_games)

    logistic = walk_forward(games, "logistic", FIRST_TEST_SEASON)
    gbm = walk_forward(games, "gbm", FIRST_TEST_SEASON)
    elo_only = walk_forward(games, "logistic", FIRST_TEST_SEASON, features=["elo_diff"])

    truth = logistic["home_win"].to_numpy()

    rows = [
        # The naive benchmark: pick the home team every single week.
        score(truth, np.full(len(truth), truth.mean())).as_row("Always home"),
        score(truth, elo_only["pred"].to_numpy()).as_row("Elo only"),
        score(truth, gbm["pred"].to_numpy()).as_row("Gradient boosting"),
        score(truth, logistic["pred"].to_numpy()).as_row("Logistic (full)"),
    ]

    # Vegas is only scored on the subset where a line exists, so it gets its own
    # row with its own game count rather than being compared row-for-row.
    vegas = logistic["vegas_home_prob"].to_numpy()
    if np.isfinite(vegas).any():
        rows.append(score(truth, vegas).as_row("Vegas moneyline"))
        overlap = np.isfinite(vegas)
        rows.append(
            score(truth[overlap], logistic["pred"].to_numpy()[overlap]).as_row(
                "Logistic (Vegas games only)"
            )
        )

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "backtest.csv", index=False)
    print("\nWalk-forward, seasons %d-%d\n" % (FIRST_TEST_SEASON, games.season.max()))
    print(table.to_string(index=False))

    by_season = (
        logistic.groupby("season")
        .apply(
            lambda g: pd.Series(
                score(g["home_win"].to_numpy(), g["pred"].to_numpy()).as_row("season")
            ),
            include_groups=False,
        )
        .drop(columns=["model"])
    )
    by_season.to_csv(REPORTS / "by_season.csv")
    print("\nPer season\n")
    print(by_season.to_string())

    _calibration_plot(truth, logistic["pred"].to_numpy())


def _calibration_plot(truth: np.ndarray, pred: np.ndarray) -> None:
    """Do games we call 70% actually happen 70% of the time?"""
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(pred, bins) - 1

    centres, actual, counts = [], [], []
    for b in range(10):
        mask = idx == b
        if mask.sum() < 20:
            continue
        centres.append(pred[mask].mean())
        actual.append(truth[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="Perfect calibration")
    ax.scatter(centres, actual, s=np.array(counts) / 3, color="#1f77b4", zorder=3)
    ax.plot(centres, actual, color="#1f77b4", label="Model")
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Observed home win rate")
    ax.set_title("Calibration, walk-forward %d-present" % FIRST_TEST_SEASON)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=150)
    print("\nwrote reports/calibration.png")


if __name__ == "__main__":
    main()
