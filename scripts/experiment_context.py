"""Test whether coaching, weather and travel context add anything.

Written to answer a specific question honestly: coaching changes obviously
matter to a football fan, so does a model that already knows team strength and
quarterback quality get any *additional* signal from them?

The confound to beat is severe. Teams fire coaches because they were bad, and
"was bad" is already in Elo. A naive new-coach flag can look predictive while
carrying no information the model did not already have, so every candidate here
is judged on whether it improves walk-forward log loss *on top of* the existing
features -- not on whether it correlates with winning.

Run with ``python scripts/experiment_context.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflpred import data, pipeline
from nflpred.dataset import FEATURE_COLUMNS
from nflpred.model import score, walk_forward

FIRST_TEST_SEASON = 2012


def coach_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-game coach tenure and first-season flags for both teams.

    Tenure counts games this coach has already coached this team, so it is
    zero on a debut and rises from there. Both are strictly pre-game.
    """
    played = schedules.sort_values("gameday")
    tenure: dict[tuple[str, str], int] = {}
    first_season: dict[tuple[str, str], int] = {}
    rows = []

    for game in played.itertuples():
        record = {"game_id": game.game_id}

        for side in ("home", "away"):
            team = getattr(game, f"{side}_team")
            coach = getattr(game, f"{side}_coach")
            key = (team, coach)

            record[f"{side}_coach_tenure"] = tenure.get(key, 0)
            first_season.setdefault(key, game.season)
            record[f"{side}_coach_new"] = int(tenure.get(key, 0) == 0)
            record[f"{side}_coach_first_season"] = int(
                first_season[key] == game.season
            )
            tenure[key] = tenure.get(key, 0) + 1

        rows.append(record)

    frame = pd.DataFrame(rows)
    frame["coach_tenure_diff"] = (
        frame["home_coach_tenure"] - frame["away_coach_tenure"]
    )
    frame["coach_new_diff"] = frame["home_coach_new"] - frame["away_coach_new"]
    frame["coach_first_season_diff"] = (
        frame["home_coach_first_season"] - frame["away_coach_first_season"]
    )
    return frame


def weather_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Wind, temperature and whether the game is played indoors.

    Weather hits both teams equally, so it should not move a *win* probability
    much on its own. It is tested anyway because that reasoning deserves
    checking rather than assuming.
    """
    frame = schedules[["game_id", "temp", "wind", "roof"]].copy()
    indoors = frame["roof"].isin(["dome", "closed"])

    frame["is_indoors"] = indoors.astype(int)
    # Indoor games have no reading because there is no weather to read.
    frame["wind"] = np.where(indoors, 0.0, frame["wind"]).astype(float)
    frame["temp"] = np.where(indoors, 70.0, frame["temp"]).astype(float)

    frame["wind_speed"] = frame["wind"].fillna(frame["wind"].median())
    frame["temp"] = frame["temp"].fillna(frame["temp"].median())
    frame["cold"] = np.maximum(0.0, 50.0 - frame["temp"])

    # Renamed so they do not collide with the raw schedule columns already
    # carried on the game table.
    return frame[["game_id", "is_indoors", "wind_speed", "cold"]]


CANDIDATES = {
    "coach tenure": ["coach_tenure_diff"],
    "new coach": ["coach_new_diff"],
    "first season": ["coach_first_season_diff"],
    "all coaching": [
        "coach_tenure_diff",
        "coach_new_diff",
        "coach_first_season_diff",
    ],
    "weather": ["is_indoors", "wind_speed", "cold"],
    "coaching + weather": [
        "coach_tenure_diff",
        "coach_new_diff",
        "coach_first_season_diff",
        "is_indoors",
        "wind_speed",
        "cold",
    ],
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    games, team_games, _, _ = pipeline.build()
    schedules = data.load_schedules()

    games = games.merge(coach_context(schedules), on="game_id", how="left")
    games = games.merge(weather_context(schedules), on="game_id", how="left")

    extra = sorted({c for cols in CANDIDATES.values() for c in cols})
    games = games.dropna(subset=extra).reset_index(drop=True)

    baseline = walk_forward(games, "logistic", FIRST_TEST_SEASON)
    truth = baseline["home_win"].to_numpy()
    base = score(truth, baseline["pred"].to_numpy())

    rows = [{"added": "nothing (current model)", **base.as_row("")}]

    for name, columns in CANDIDATES.items():
        result = walk_forward(
            games, "logistic", FIRST_TEST_SEASON, features=FEATURE_COLUMNS + columns
        )
        scored = score(truth, result["pred"].to_numpy())
        row = scored.as_row("")
        row["delta"] = round(scored.log_loss - base.log_loss, 5)
        rows.append({"added": name, **row})

    table = pd.DataFrame(rows).drop(columns=["model"])
    print("\nDoes context help, on top of the existing features?")
    print("Negative delta = better. Seasons %d-%d.\n"
          % (FIRST_TEST_SEASON, games.season.max()))
    print(table.to_string(index=False))

    _raw_effects(games)


def _raw_effects(games: pd.DataFrame) -> None:
    """The uncontrolled picture, to show why controlling matters."""
    first = games[games["home_coach_first_season"] == 1]
    rest = games[games["home_coach_first_season"] == 0]

    print("\nUncontrolled: how home teams do under a first-season coach\n")
    print(f"  first-season coach   {first['home_win'].mean():.3f} win rate "
          f"({len(first)} games), mean Elo {first['home_elo_pre'].mean():.0f}")
    print(f"  established coach    {rest['home_win'].mean():.3f} win rate "
          f"({len(rest)} games), mean Elo {rest['home_elo_pre'].mean():.0f}")


if __name__ == "__main__":
    main()
