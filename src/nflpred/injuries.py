"""Who is unavailable this week, from the official injury reports.

This is the one input that tells the model something results cannot. Elo and
efficiency describe the team that played; the injury report describes the team
about to play. Everywhere else in this project, roster change is invisible until
it shows up in outcomes -- here it arrives before kickoff.

Reports are published from Wednesday of game week, which is why the scheduled
job runs when it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rough importance by position. A starting tackle matters more than a third
# safety, and counting every absence equally drowns the signal in inactive
# special-teamers. Deliberately coarse: these are judgement calls, and tuning
# them against the test set would be fitting noise.
POSITION_WEIGHT = {
    "QB": 4.0,
    "T": 1.6,
    "LT": 1.6,
    "WR": 1.3,
    "CB": 1.3,
    "DE": 1.3,
    "EDGE": 1.3,
    "G": 1.1,
    "C": 1.1,
    "S": 1.0,
    "FS": 1.0,
    "SS": 1.0,
    "LB": 1.0,
    "OLB": 1.0,
    "DT": 1.0,
    "TE": 0.9,
    "RB": 0.9,
    "ILB": 0.8,
    "NT": 0.8,
    "FB": 0.3,
    "K": 0.3,
    "P": 0.2,
    "LS": 0.1,
}
DEFAULT_POSITION_WEIGHT = 0.9

# Doubtful players usually sit; questionable ones usually play. Anything not
# listed carries no weight.
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}


def burden(injuries: pd.DataFrame) -> pd.DataFrame:
    """Per team-week: how many players are missing, and how much they matter."""
    if injuries.empty:
        return pd.DataFrame(
            columns=["season", "week", "team", "out_count", "weighted_out"]
        )

    frame = injuries.copy()
    frame["status_weight"] = frame["report_status"].map(STATUS_WEIGHT).fillna(0.0)
    frame["position_weight"] = (
        frame["position"].map(POSITION_WEIGHT).fillna(DEFAULT_POSITION_WEIGHT)
    )
    frame["weighted"] = frame["status_weight"] * frame["position_weight"]

    return (
        frame.groupby(["season", "week", "team"], observed=True)
        .agg(
            out_count=("status_weight", lambda s: float((s == 1.0).sum())),
            weighted_out=("weighted", "sum"),
        )
        .reset_index()
    )


def attach(games: pd.DataFrame, team_week_burden: pd.DataFrame) -> pd.DataFrame:
    """Add the home-minus-away injury differential to a game table.

    Teams with no report for a week are treated as fully healthy. That is the
    right default: a missing report means nothing was filed, and guessing at an
    unknown injury load would be worse than assuming none.
    """
    out = games.copy()

    if team_week_burden.empty:
        out["home_weighted_out"] = 0.0
        out["away_weighted_out"] = 0.0
        out["injury_diff"] = 0.0
        return out

    for side in ("home", "away"):
        renamed = team_week_burden.rename(
            columns={
                "team": f"{side}_team",
                "out_count": f"{side}_out_count",
                "weighted_out": f"{side}_weighted_out",
            }
        )
        out = out.merge(renamed, on=["season", "week", f"{side}_team"], how="left")
        out[f"{side}_weighted_out"] = out[f"{side}_weighted_out"].fillna(0.0)
        out[f"{side}_out_count"] = out[f"{side}_out_count"].fillna(0.0)

    # Positive means the away team is the more banged-up side, favouring home.
    out["injury_diff"] = out["away_weighted_out"] - out["home_weighted_out"]
    return out


def describe(row: pd.Series) -> str:
    """A short human-readable note about the injury gap in one game."""
    diff = row.get("injury_diff", 0.0)
    if not np.isfinite(diff) or abs(diff) < 1.0:
        return ""

    favoured = row["home_team"] if diff > 0 else row["away_team"]
    return f"{favoured} the healthier side"
