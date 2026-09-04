"""One call that goes from public URLs to a modelling table.

Both the training scripts and the weekly forecaster enter through here, so the
features a prediction is made from are built by exactly the same code that
built the features the model was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import data, dataset, features, injuries, quarterback

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Play-by-play only goes back usefully to 1999, but betting lines -- needed for
# the Vegas benchmark -- start in 2006, so that is where training begins.
# Injury reports begin in 2009 and are now a model feature, so training starts
# there rather than 2006. Three seasons of history are worth less than a
# feature that is populated on every row.
DEFAULT_START_SEASON = 2009


def build(
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
    refresh: bool = False,
    qb_priors=None,
    draft_prior_upto: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Return ``(games, team_games, elo_now, starters)`` ready for modelling.

    ``games`` is one row per game with features and labels, ``team_games`` is
    the per-team form history behind them, ``elo_now`` is every team's rating
    after the most recent completed game, and ``starters`` is the per-game
    quarterback history.
    """
    schedules = data.load_schedules(refresh=refresh)

    # Schedules run a season ahead of results: next year's fixtures are
    # published before a single snap of it is played. Play-by-play only exists
    # for seasons that have actually started, so the last *played* season is
    # the ceiling. This advances on its own once Week 1 kicks off.
    played = schedules[schedules["result"].notna()]
    end_season = end_season or int(played["season"].max())

    log.info("loading play-by-play %s-%s", start_season, end_season)
    seasons = list(range(start_season, end_season + 1))
    pbp = data.load_pbp(seasons, refresh=refresh)

    stats = features.team_game_stats(pbp)
    team_games = features.build_team_games(stats, schedules)
    team_games = features.add_form(team_games)

    qb_stats = quarterback.qb_game_stats(pbp)
    starters = quarterback.identify_starters(qb_stats, schedules)

    if qb_priors is None:
        # A passer with almost no record is shrunk toward a prior set by his
        # draft slot rather than a flat replacement level. Estimating it needs
        # start counts, so a provisional pass comes first.
        provisional = quarterback.add_qb_form(starters)
        qb_priors = quarterback.draft_priors(
            provisional,
            data.load_draft_picks(refresh=refresh),
            upto_season=draft_prior_upto or end_season,
        )

    starters = quarterback.add_qb_form(starters, priors=qb_priors)

    elo_pre, elo_now = features.elo_ratings(schedules)
    games = dataset.build_dataset(team_games, schedules, elo_pre)
    games = quarterback.attach_to_games(games, starters)

    reports = data.load_injuries(seasons, refresh=refresh)
    games = injuries.attach(games, injuries.burden(reports))

    games = dataset.modelling_frame(games)

    log.info("built %d labelled games", len(games))
    return games, team_games, elo_now, starters


def save(games: pd.DataFrame, team_games: pd.DataFrame) -> None:
    """Cache the built tables so downstream steps do not re-download anything."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    games.to_parquet(PROCESSED_DIR / "games.parquet", index=False)
    team_games.to_parquet(PROCESSED_DIR / "team_games.parquet", index=False)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back what ``save`` wrote."""
    return (
        pd.read_parquet(PROCESSED_DIR / "games.parquet"),
        pd.read_parquet(PROCESSED_DIR / "team_games.parquet"),
    )
