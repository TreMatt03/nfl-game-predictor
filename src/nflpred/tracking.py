"""Record predictions when they are made, and score them once games are played.

The point of this file is to make the model's public record impossible to
flatter. A prediction is written to the log the first time it is published and
never rewritten, so the tracked history is what was actually claimed
beforehand -- not what the current model, trained on more data, would say about
a game it has now seen the result of.

That distinction is the whole reason the log is a committed file rather than
something regenerated on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .model import score

log = logging.getLogger(__name__)

LOG_PATH = Path(__file__).resolve().parents[2] / "predictions_log.csv"

LOG_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "away_qb",
    "home_qb",
    "home_win_prob",
    "favorite",
    "predicted_at",
    # Which build made the call. Without it a model change is invisible in the
    # record, and a track record spanning two different models reads as one.
    "model_version",
]


def load_log(path: Path | None = None) -> pd.DataFrame:
    """Read the prediction log, or an empty frame shaped like it."""
    path = path or LOG_PATH
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path)


def log_predictions(slate, path: Path | None = None) -> int:
    """Append predictions for games not already logged. Returns how many.

    Existing entries are left untouched. If a slate is regenerated -- a rerun,
    a corrected starter, a model change -- the original call stands, because
    revising a prediction after the fact and still calling it a prediction is
    how a track record becomes fiction.
    """
    path = path or LOG_PATH
    existing = load_log(path)

    frame = slate.games.copy()
    frame["predicted_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    frame["model_version"] = __version__
    frame["gameday"] = frame["gameday"].dt.strftime("%Y-%m-%d")

    for column in ("away_qb", "home_qb"):
        if column not in frame:
            frame[column] = None

    fresh = frame[~frame["game_id"].isin(set(existing["game_id"]))][LOG_COLUMNS]
    if fresh.empty:
        log.info("no new predictions to log")
        return 0

    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.sort_values(["season", "week", "gameday", "game_id"])
    combined.to_csv(path, index=False)

    log.info("logged %d new predictions", len(fresh))
    return len(fresh)


def score_history(
    predictions: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Join logged predictions to final scores, keeping only settled games."""
    results = schedules[schedules["result"].notna()][["game_id", "result"]]
    merged = predictions.merge(results, on="game_id", how="inner")

    if merged.empty:
        return merged.assign(home_win=[], correct=[], predicted_home=[])

    # Ties count as neither right nor wrong, so they are dropped rather than
    # scored as a loss for whichever side was favoured.
    merged = merged[merged["result"] != 0].copy()

    merged["home_win"] = (merged["result"] > 0).astype(int)
    merged["predicted_home"] = (merged["home_win_prob"] >= 0.5).astype(int)
    merged["correct"] = (merged["home_win"] == merged["predicted_home"]).astype(int)
    merged["winner"] = np.where(
        merged["home_win"] == 1, merged["home_team"], merged["away_team"]
    )
    merged["confidence"] = np.maximum(
        merged["home_win_prob"], 1.0 - merged["home_win_prob"]
    )
    return merged


def summarise(scored: pd.DataFrame) -> dict:
    """Headline record over every settled prediction."""
    if scored.empty:
        return {"games": 0}

    metrics = score(
        scored["home_win"].to_numpy(), scored["home_win_prob"].to_numpy()
    )
    return {
        "games": len(scored),
        "wins": int(scored["correct"].sum()),
        "losses": int((1 - scored["correct"]).sum()),
        "accuracy": metrics.accuracy,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
    }


def by_week(scored: pd.DataFrame) -> pd.DataFrame:
    """Record week by week, most recent first."""
    if scored.empty:
        return pd.DataFrame(columns=["season", "week", "games", "correct", "accuracy"])

    grouped = (
        scored.groupby(["season", "week"])
        .agg(games=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["games"]
    return grouped.sort_values(["season", "week"], ascending=False)


def latest_week(scored: pd.DataFrame) -> pd.DataFrame:
    """Every settled game from the most recently completed week."""
    if scored.empty:
        return scored

    season = scored["season"].max()
    week = scored[scored["season"] == season]["week"].max()
    recent = scored[(scored["season"] == season) & (scored["week"] == week)]
    return recent.sort_values("confidence", ascending=False)
