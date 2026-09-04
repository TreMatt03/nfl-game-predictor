"""Forecast games that have not been played.

The awkward part of forecasting, as opposed to backtesting, is that the inputs
come from a different place. A historical row knows the pre-game Elo and who
actually took the snaps; an upcoming row has to be assembled from each team's
current state and an assumption about who will start. Everything in this module
exists to build that row so it lines up exactly with what the model trained on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import quarterback
from .dataset import FEATURE_COLUMNS
from .explain import top_drivers
from .features import OFFENSE_METRICS, latest_form, regress_ratings

log = logging.getLogger(__name__)


@dataclass
class Slate:
    """A week of upcoming games with predictions attached."""

    season: int
    week: int
    games: pd.DataFrame


def next_unplayed_week(schedules: pd.DataFrame) -> tuple[int, int]:
    """The earliest season and week that still has an unplayed game."""
    pending = schedules[schedules["result"].isna()]
    if pending.empty:
        raise RuntimeError("every scheduled game has a result")

    season = int(pending["season"].min())
    week = int(pending[pending["season"] == season]["week"].min())
    return season, week


def _net_now(team_games: pd.DataFrame) -> pd.DataFrame:
    """Each team's current offence-minus-defence form, one row per team."""
    form = latest_form(team_games)
    net = pd.DataFrame(index=form.index)
    for metric in OFFENSE_METRICS:
        net[metric] = form[f"off_{metric}_form"] - form[f"def_{metric}_form"]
    return net


def build_upcoming(
    schedules: pd.DataFrame,
    team_games: pd.DataFrame,
    elo_now: dict[str, float],
    starters: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Feature rows for one week's fixtures, matching the training layout."""
    fixtures = schedules[
        (schedules["season"] == season) & (schedules["week"] == week)
    ].copy()

    net = _net_now(team_games)

    # The season matters here: it selects the confirmed-starter overrides that
    # correct for offseason moves the play-by-play cannot yet show.
    qbs = quarterback.current_starters(starters, season=season)

    # Ratings come back current as of the last completed game. If we are
    # forecasting a later season, take the off-season regression for each
    # boundary in between before using them.
    last_played = int(schedules[schedules["result"].notna()]["season"].max())
    if season > last_played:
        elo_now = regress_ratings(elo_now, seasons=season - last_played)

    for metric in OFFENSE_METRICS:
        home_val = fixtures["home_team"].map(net[metric])
        away_val = fixtures["away_team"].map(net[metric])
        fixtures[f"net_{metric}"] = home_val - away_val

    fixtures["home_elo_pre"] = fixtures["home_team"].map(elo_now).fillna(1500.0)
    fixtures["away_elo_pre"] = fixtures["away_team"].map(elo_now).fillna(1500.0)
    fixtures["elo_diff"] = fixtures["home_elo_pre"] - fixtures["away_elo_pre"]

    fixtures["rest_diff"] = fixtures["home_rest"] - fixtures["away_rest"]
    fixtures["neutral_site"] = (fixtures["location"] != "Home").astype(int)
    fixtures["div_game"] = fixtures["div_game"].fillna(0).astype(int)

    # Last week's starter is assumed to start again -- the same assumption an
    # opening betting line makes, and wrong far less often than it is right.
    for side in ("home", "away"):
        fixtures[f"{side}_qb"] = fixtures[f"{side}_team"].map(qbs["qb_name"])
        fixtures[f"{side}_qb_form"] = (
            fixtures[f"{side}_team"]
            .map(qbs["qb_form_now"])
            .fillna(quarterback.REPLACEMENT_LEVEL)
        )
    fixtures["qb_diff"] = fixtures["home_qb_form"] - fixtures["away_qb_form"]

    return fixtures


def predict_slate(
    model,
    schedules: pd.DataFrame,
    team_games: pd.DataFrame,
    elo_now: dict[str, float],
    starters: pd.DataFrame,
    season: int | None = None,
    week: int | None = None,
) -> Slate:
    """Predict the next unplayed week, or a specific one if asked."""
    if season is None or week is None:
        season, week = next_unplayed_week(schedules)

    fixtures = build_upcoming(
        schedules, team_games, elo_now, starters, season, week
    )

    missing = fixtures[FEATURE_COLUMNS].isna().any(axis=1)
    if missing.any():
        # Happens in Week 1 when a team's form has not been rebuilt yet. A
        # neutral value keeps the game on the slate instead of dropping it.
        log.warning("%d fixtures missing features, filling neutral", missing.sum())
        fixtures[FEATURE_COLUMNS] = fixtures[FEATURE_COLUMNS].fillna(0.0)

    fixtures["home_win_prob"] = model.predict_proba(fixtures[FEATURE_COLUMNS])[:, 1]
    fixtures["away_win_prob"] = 1.0 - fixtures["home_win_prob"]
    fixtures["favorite"] = np.where(
        fixtures["home_win_prob"] >= 0.5,
        fixtures["home_team"],
        fixtures["away_team"],
    )
    fixtures["confidence"] = fixtures[["home_win_prob", "away_win_prob"]].max(axis=1)
    fixtures["drivers"] = top_drivers(model, fixtures, n=3)

    fixtures = fixtures.sort_values("confidence", ascending=False)
    return Slate(season=season, week=week, games=fixtures)
