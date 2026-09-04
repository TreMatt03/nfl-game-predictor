"""Quarterback form, tracked per player rather than per team.

Team-level metrics quietly assume the roster is the same one that produced
them. That assumption breaks hardest at quarterback: a team whose starter is
injured is a materially different team, and Elo has no way to know it. Rating
the passer separately and carrying that rating with the player is what lets the
model react in the week a backup takes over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Halflife in starts. Longer than the team halflife because quarterback quality
# is a more stable trait than team form.
QB_HALFLIFE = 10.0

# EPA per dropback assigned to a passer with no usable history -- an untested
# rookie or a backup off the bench. Set to the 25th percentile of established
# starters (measured at +0.002 over 2006-2025, rounded), so an unknown is
# treated as below the +0.079 starter average but not hopeless. Recheck with
# quarterback.replacement_percentile().
REPLACEMENT_LEVEL = 0.0

# A start needs enough snaps to mean anything; below this the game was a relief
# appearance or a wildcat gimmick, not a start.
MIN_DROPBACKS = 10

# Starts of regression toward replacement level. A quarterback with this many
# starts is trusted halfway; one with none is trusted not at all. Without it a
# single ugly afternoon defines a career.
SHRINKAGE_STARTS = 6.0

# How far back to look when deciding who a team's real starter is. Long enough
# that one rested Week 18 does not overturn a season, short enough that a
# genuine midseason change is picked up.
STARTER_LOOKBACK = 10


def qb_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """EPA per dropback for every passer in every game.

    Dropbacks rather than pass attempts, so sacks and scrambles -- both of which
    are quarterback outcomes -- count against or for the passer.
    """
    plays = pbp[
        (pbp["qb_dropback"] == 1)
        & pbp["passer_player_id"].notna()
        & pbp["qb_epa"].notna()
        & pbp["posteam"].notna()
    ]

    stats = (
        plays.groupby(
            ["game_id", "season", "week", "posteam", "passer_player_id"],
            observed=True,
        )
        .agg(
            qb_epa=("qb_epa", "mean"),
            dropbacks=("qb_epa", "size"),
            qb_name=("passer_player_name", "first"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    return stats[stats["dropbacks"] >= MIN_DROPBACKS]


def identify_starters(qb_stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """The passer with the most dropbacks for each team in each game.

    Derived from what actually happened on the field rather than from a depth
    chart, so a quarterback pulled in the first quarter is not credited with the
    start.
    """
    starters = qb_stats.sort_values("dropbacks", ascending=False).drop_duplicates(
        ["game_id", "team"]
    )

    dates = schedules[["game_id", "gameday"]]
    starters = starters.merge(dates, on="game_id", how="left")
    return starters.sort_values(["passer_player_id", "gameday"]).reset_index(drop=True)


def _shrink(form: pd.Series, starts: pd.Series) -> pd.Series:
    """Pull a rating toward replacement level in proportion to its thinness.

    A passer with two starts and a dreadful average is far more likely to be
    unlucky than genuinely that bad. Weighting the observed form by sample size
    against a replacement-level prior is the standard correction, and it stops
    one Week 18 cameo from being treated as a career.
    """
    observed = form.fillna(REPLACEMENT_LEVEL)
    weight = starts / (starts + SHRINKAGE_STARTS)
    return weight * observed + (1.0 - weight) * REPLACEMENT_LEVEL


def add_qb_form(starters: pd.DataFrame) -> pd.DataFrame:
    """Lagged, shrunk EWMA of each passer's EPA per dropback.

    Keyed on the player, not the team, so the rating follows a quarterback who
    is traded or who takes over mid-season. As elsewhere, the shift(1) is what
    keeps a game out of its own prediction.
    """
    out = starters.sort_values(["passer_player_id", "gameday"]).copy()
    grouped = out.groupby("passer_player_id", observed=True)

    raw = grouped["qb_epa"].transform(
        lambda s: s.shift(1).ewm(halflife=QB_HALFLIFE, min_periods=1).mean()
    )
    out["qb_starts"] = grouped.cumcount()
    out["qb_form"] = _shrink(raw, out["qb_starts"])
    return out


def current_starters(starters_with_form: pd.DataFrame) -> pd.DataFrame:
    """Each team's established starter and their up-to-date rating.

    Naively taking the most recent start gets this badly wrong. Teams out of
    contention rest their starter in Week 18, so the last man to take a snap is
    often a backup who will not see the field again -- which is how a model ends
    up projecting Kansas City with its third-string quarterback. Taking whoever
    started most of the team's recent games instead survives a rested week while
    still catching a real midseason change.
    """
    out = starters_with_form.sort_values(["passer_player_id", "gameday"]).copy()

    # Unshifted this time: for a future game the most recent start is legitimate
    # information, not leakage.
    grouped = out.groupby("passer_player_id", observed=True)
    raw_now = grouped["qb_epa"].transform(
        lambda s: s.ewm(halflife=QB_HALFLIFE, min_periods=1).mean()
    )
    career_starts = grouped.cumcount() + 1
    out["qb_form_now"] = _shrink(raw_now, career_starts)
    out["career_starts"] = career_starts

    recent = (
        out.sort_values("gameday")
        .groupby("team", observed=True)
        .tail(STARTER_LOOKBACK)
    )

    # Most starts in the window, ties broken by whoever started most recently.
    counts = (
        recent.groupby(["team", "passer_player_id"], observed=True)
        .agg(starts=("game_id", "size"), last=("gameday", "max"))
        .reset_index()
        .sort_values(["starts", "last"], ascending=[False, False])
        .drop_duplicates("team")
    )

    latest_row = (
        out.sort_values("gameday")
        .groupby(["team", "passer_player_id"], observed=True)
        .tail(1)
    )
    chosen = counts.merge(latest_row, on=["team", "passer_player_id"], how="left")

    return chosen.set_index("team")[
        ["passer_player_id", "qb_name", "qb_form_now", "career_starts"]
    ]


def attach_to_games(games: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Add home/away quarterback ratings and their difference to the game table."""
    lookup = starters[["game_id", "team", "qb_form", "qb_name"]]

    home = lookup.rename(
        columns={
            "team": "home_team",
            "qb_form": "home_qb_form",
            "qb_name": "home_qb",
        }
    )
    away = lookup.rename(
        columns={
            "team": "away_team",
            "qb_form": "away_qb_form",
            "qb_name": "away_qb",
        }
    )

    games = games.merge(home, on=["game_id", "home_team"], how="left")
    games = games.merge(away, on=["game_id", "away_team"], how="left")

    for side in ("home", "away"):
        games[f"{side}_qb_form"] = games[f"{side}_qb_form"].fillna(REPLACEMENT_LEVEL)

    games["qb_diff"] = games["home_qb_form"] - games["away_qb_form"]
    return games


def replacement_percentile(starters_with_form: pd.DataFrame, q: float = 0.25) -> float:
    """The quantile used to set REPLACEMENT_LEVEL, over established starters."""
    established = starters_with_form[starters_with_form["qb_starts"] >= 10]
    return float(np.nanquantile(established["qb_form"], q))
