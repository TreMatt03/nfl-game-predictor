"""Loading and caching of public nflverse data.

Two sources are used, deliberately kept separate because they have very
different costs:

* ``games.csv`` (~2 MB) has one row per game back to 1999 with final scores,
  rest days, and closing betting lines. Cheap enough to re-download often.
* ``play_by_play_{season}.parquet`` (~20 MB each) has every play. Expensive, so
  seasons are cached on disk and only the aggregate is ever kept in memory.

Nothing here needs an API key. All of it is public.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)

# Repo-root/data. Gitignored; rebuilt on demand.
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Teams that relocated or rebranded. nflverse uses the historical code in old
# seasons, so Elo ratings would otherwise treat a moved franchise as brand new.
TEAM_ALIASES = {
    "OAK": "LV",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
}


def normalize_team(series: pd.Series) -> pd.Series:
    """Collapse relocated franchises onto their current abbreviation."""
    return series.replace(TEAM_ALIASES)


def load_schedules(refresh: bool = False) -> pd.DataFrame:
    """Return every scheduled game 1999-present, including future ones.

    Rows with a null ``result`` have not been played yet -- that is what makes
    this table usable both for training labels and for the upcoming slate.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "games.csv"

    if refresh or not path.exists():
        log.info("downloading schedules")
        resp = requests.get(SCHEDULE_URL, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)

    games = pd.read_csv(path, low_memory=False)
    games["gameday"] = pd.to_datetime(games["gameday"])
    for col in ("home_team", "away_team"):
        games[col] = normalize_team(games[col])
    return games


def load_pbp(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Return play-by-play for ``seasons``, caching each season's parquet.

    Only the columns needed downstream are read back off disk; the full file has
    372 of them and we use about twenty.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for season in seasons:
        path = CACHE_DIR / f"pbp_{season}.parquet"
        if refresh or not path.exists():
            log.info("downloading play-by-play for %s", season)
            resp = requests.get(PBP_URL.format(season=season), timeout=600)

            # nflverse publishes a season's file once it kicks off. Asking for
            # one that does not exist yet is expected in the off-season, not an
            # error worth failing the whole run over.
            if resp.status_code == 404:
                log.warning("no play-by-play published for %s yet, skipping", season)
                continue
            resp.raise_for_status()
            path.write_bytes(resp.content)

        frames.append(pd.read_parquet(path, columns=PBP_COLUMNS))

    if not frames:
        raise RuntimeError(f"no play-by-play available for seasons {seasons}")

    pbp = pd.concat(frames, ignore_index=True)
    for col in ("posteam", "defteam"):
        pbp[col] = normalize_team(pbp[col])
    return pbp


PBP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "pass",
    "rush",
    "yards_gained",
    "interception",
    "fumble_lost",
    "sack",
    "cpoe",
    "third_down_converted",
    "third_down_failed",
    "penalty_yards",
    "qb_dropback",
    "qb_epa",
    "passer_player_id",
    "passer_player_name",
]
