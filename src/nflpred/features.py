"""Turn raw plays and scores into one modelling row per game.

The whole file exists to answer a single question without cheating: what did we
know about these two teams before kickoff? Every team metric is an
exponentially weighted average that is shifted one game back, so a result never
leaks into the prediction of the game that produced it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A halflife of 8 games means roughly half the signal comes from the last half
# season. Short enough to track a team that changed, long enough to survive one
# weird Sunday.
FORM_HALFLIFE = 8.0

# Metrics aggregated per team per game, then averaged over recent history.
OFFENSE_METRICS = [
    "epa_play",
    "pass_epa",
    "rush_epa",
    "success_rate",
    "explosive_rate",
    "turnover_rate",
    "sack_rate",
    "third_down_rate",
]

FORM_COLUMNS = [f"off_{m}_form" for m in OFFENSE_METRICS] + [
    f"def_{m}_form" for m in OFFENSE_METRICS
]

_RAW_METRIC_COLUMNS = [f"off_{m}" for m in OFFENSE_METRICS] + [
    f"def_{m}" for m in OFFENSE_METRICS
]


def team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate plays into one offensive stat line per (game, team).

    Restricted to scrimmage plays with a defined EPA, which drops kneels,
    spikes, special teams, and aborted plays that would otherwise drag the
    per-play averages around.
    """
    plays = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
    ].copy()

    plays["is_explosive"] = (plays["yards_gained"] >= 20).astype(float)
    plays["is_turnover"] = (
        (plays["interception"] == 1) | (plays["fumble_lost"] == 1)
    ).astype(float)

    keys = ["game_id", "season", "week", "posteam", "defteam"]
    stats = (
        plays.groupby(keys, observed=True)
        .agg(
            epa_play=("epa", "mean"),
            success_rate=("success", "mean"),
            explosive_rate=("is_explosive", "mean"),
            turnover_rate=("is_turnover", "mean"),
            sack_rate=("sack", "mean"),
            third_downs_won=("third_down_converted", "sum"),
            third_downs_lost=("third_down_failed", "sum"),
            plays=("epa", "size"),
        )
        .reset_index()
    )

    # Split EPA by play type; a team that can only throw is a different team.
    # After the filter above pass == 0 implies a run, so one unstack does it.
    splits = (
        plays.groupby(["game_id", "posteam", "pass"], observed=True)["epa"]
        .mean()
        .unstack("pass")
        .rename(columns={1.0: "pass_epa", 0.0: "rush_epa", 1: "pass_epa", 0: "rush_epa"})
        .reset_index()
    )
    stats = stats.merge(splits, on=["game_id", "posteam"], how="left")

    attempts = stats["third_downs_won"] + stats["third_downs_lost"]
    stats["third_down_rate"] = np.where(
        attempts > 0, stats["third_downs_won"] / attempts.replace(0, np.nan), np.nan
    )

    stats = stats.rename(columns={"posteam": "team", "defteam": "opponent"})
    return stats.drop(columns=["third_downs_won", "third_downs_lost"])


def build_team_games(stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game holding both sides of the ball.

    A team's defensive stat line for a game is its opponent's offensive stat
    line from that same game, which is how defensive efficiency is measured in
    practice.
    """
    offense = stats.rename(columns={m: f"off_{m}" for m in OFFENSE_METRICS})

    defense = stats.rename(columns={"team": "opponent", "opponent": "team"})
    defense = defense.rename(columns={m: f"def_{m}" for m in OFFENSE_METRICS})
    defense = defense[["game_id", "team"] + [f"def_{m}" for m in OFFENSE_METRICS]]

    team_games = offense.merge(defense, on=["game_id", "team"], how="inner")

    dates = schedules[["game_id", "gameday"]]
    team_games = team_games.merge(dates, on="game_id", how="left")
    return team_games.sort_values(["team", "gameday"]).reset_index(drop=True)


def add_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """Attach lagged exponentially weighted form for every metric.

    The shift(1) before the ewm is the entire anti-leakage guarantee: the value
    on row n is built only from rows 0..n-1.
    """
    out = team_games.sort_values(["team", "gameday"]).copy()
    grouped = out.groupby("team", observed=True)

    for col in _RAW_METRIC_COLUMNS:
        out[f"{col}_form"] = grouped[col].transform(
            lambda s: s.shift(1).ewm(halflife=FORM_HALFLIFE, min_periods=3).mean()
        )

    # How much history backs this row. Used to drop unreliable early rows.
    out["games_seen"] = grouped.cumcount()
    return out


def latest_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """Form for each team including its most recent game.

    add_form deliberately lags by one game, which is right for training but
    wrong for forecasting a game that has not been played. Re-running the EWMA
    without the shift and taking the last row per team gives the current state
    to predict from.
    """
    out = team_games.sort_values(["team", "gameday"]).copy()
    grouped = out.groupby("team", observed=True)

    for col in _RAW_METRIC_COLUMNS:
        out[f"{col}_form"] = grouped[col].transform(
            lambda s: s.ewm(halflife=FORM_HALFLIFE, min_periods=1).mean()
        )

    return out.groupby("team", observed=True).tail(1).set_index("team")[FORM_COLUMNS]


def regress_ratings(
    ratings: dict[str, float], seasons: int = 1, regress: float = 0.25
) -> dict[str, float]:
    """Apply the off-season pull toward the mean to a set of final ratings.

    elo_ratings only regresses when it crosses a season boundary *in the played
    data*, so ratings handed back at the end of a season have not yet taken the
    off-season haircut. Forecasting the next season without applying it here
    would treat last January's gaps as still fully valid and produce Week 1
    probabilities well past anything the betting market would offer.
    """
    factor = (1.0 - regress) ** seasons
    return {team: 1500.0 + (r - 1500.0) * factor for team, r in ratings.items()}


def elo_ratings(
    schedules: pd.DataFrame,
    k: float = 20.0,
    home_edge: float = 55.0,
    regress: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Classic Elo with a margin-of-victory multiplier and off-season decay.

    Returns (pre_game_ratings, final_ratings). The first holds the rating each
    team carried into every played game, so it is safe to use as a feature
    directly. The second is the state after the last completed game, which is
    what upcoming fixtures need.

    Ratings start at 1500 and are pulled a quarter of the way back to the mean
    between seasons, roughly matching how much NFL team strength reverts year
    to year.
    """
    played = schedules[schedules["result"].notna()].sort_values("gameday")

    ratings: dict[str, float] = {}
    current_season = None
    rows = []

    for game in played.itertuples():
        if current_season is not None and game.season != current_season:
            for team in ratings:
                ratings[team] = 1500.0 + (ratings[team] - 1500.0) * (1 - regress)
        current_season = game.season

        home = ratings.setdefault(game.home_team, 1500.0)
        away = ratings.setdefault(game.away_team, 1500.0)

        # Neutral-site games get no home bump.
        edge = home_edge if game.location == "Home" else 0.0
        expected_home = 1.0 / (1.0 + 10 ** (-(home + edge - away) / 400.0))

        rows.append(
            {
                "game_id": game.game_id,
                "home_elo_pre": home,
                "away_elo_pre": away,
            }
        )

        margin = float(game.result)
        actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        # Blowouts move ratings more, but the log damps runaway updates and the
        # denominator stops favourites from gaining much by beating bad teams.
        signed_diff = (home + edge - away) * (1 if margin > 0 else -1)
        mov_multiplier = np.log(abs(margin) + 1) * (2.2 / (signed_diff * 0.001 + 2.2))

        shift = k * mov_multiplier * (actual_home - expected_home)
        ratings[game.home_team] = home + shift
        ratings[game.away_team] = away - shift

    return pd.DataFrame(rows), ratings
