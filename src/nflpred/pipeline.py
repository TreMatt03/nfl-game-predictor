"""One call that goes from public URLs to a modelling table.

Both the training scripts and the weekly forecaster enter through here, so the
features a prediction is made from are built by exactly the same code that
built the features the model was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import data, dataset, features, quarterback

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Play-by-play only goes back usefully to 1999, but betting lines -- needed for
# the Vegas benchmark -- start in 2006, so that is where training begins.
DEFAULT_START_SEASON = 2006


def build(
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
    refresh: bool = False,
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
    starters = quarterback.add_qb_form(starters)

    elo_pre, elo_now = features.elo_ratings(schedules)
    games = dataset.build_dataset(team_games, schedules, elo_pre)
    games = quarterback.attach_to_games(games, starters)
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
