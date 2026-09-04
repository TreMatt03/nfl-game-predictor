"""Who is unavailable this week, weighted by how much they actually play.

This is the one input that tells the model something results cannot. Elo and
efficiency describe the team that played; the injury report describes the team
about to play. Everywhere else in this project, roster change is invisible until
it shows up in outcomes -- here it arrives before kickoff.

An absence is weighted by the player's recent share of his unit's snaps, so
losing a every-down left tackle counts for far more than losing a rotational
one. Crucially that share is taken from games *before* the week in question: a
player who sat has no snaps in the week he was hurt, and using the current week
would quietly turn the feature into a readout of who did not play.

Reports are published from Wednesday of game week, which is why the scheduled
job runs when it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# How quickly a player's usage estimate follows a change in role. Four games is
# short enough to notice someone taking over a starting job mid-season.
SHARE_HALFLIFE = 4.0

# Fallback usage when a player has no snap history: an injured rookie, or any
# season before snap counts begin in 2012. These are measured means over
# 2012-2025, not guesses, so they sit on the same scale as a real share.
POSITION_SHARE = {
    "QB": 0.811,
    "G": 0.710,
    "T": 0.692,
    "C": 0.690,
    "OL": 0.619,
    "FS": 0.555,
    "CB": 0.550,
    "S": 0.543,
    "SS": 0.535,
    "WR": 0.505,
    "DE": 0.483,
    "DT": 0.478,
    "LB": 0.439,
    "DL": 0.439,
    "TE": 0.437,
    "NT": 0.435,
    "DB": 0.353,
    "RB": 0.325,
    "FB": 0.222,
    "K": 0.0,
    "P": 0.0,
    "LS": 0.0,
}
DEFAULT_SHARE = 0.45

# Doubtful players usually sit; questionable ones usually play. Anything not
# listed carries no weight.
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}


def snap_shares(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """Each player's usage entering each week, from earlier games only.

    Returns one row per player per game with the share he carried *into* it,
    which is what an injury in that week should be weighted by.
    """
    if snap_counts.empty:
        return pd.DataFrame(columns=["pfr_player_id", "order", "share_before"])

    frame = snap_counts.copy()
    frame["share"] = frame[["offense_pct", "defense_pct"]].max(axis=1)

    # A single monotonic key so weeks order correctly across seasons. Cast
    # explicitly: the injury and snap tables disagree on int vs float, and
    # merge_asof refuses to join keys of different dtype.
    frame["order"] = (frame["season"].astype("int64") * 100
                      + frame["week"].astype("int64"))
    frame = frame.sort_values(["pfr_player_id", "order"])

    frame["share_before"] = frame.groupby("pfr_player_id", observed=True)[
        "share"
    ].transform(lambda s: s.shift(1).ewm(halflife=SHARE_HALFLIFE, min_periods=1).mean())

    return frame[["pfr_player_id", "order", "share_before"]].dropna(
        subset=["share_before"]
    )


def _attach_shares(
    reports: pd.DataFrame, shares: pd.DataFrame, bridge: pd.DataFrame
) -> pd.Series:
    """Look up each injured player's usage as of the week he was listed."""
    frame = reports.copy()
    frame["order"] = (frame["season"].astype("int64") * 100
                      + frame["week"].astype("int64"))

    if bridge is not None and not bridge.empty:
        frame = frame.merge(bridge, on="gsis_id", how="left")
    else:
        frame["pfr_id"] = np.nan

    if shares.empty or frame["pfr_id"].isna().all():
        return pd.Series(np.nan, index=frame.index)

    left = frame[["pfr_id", "order"]].rename(columns={"pfr_id": "pfr_player_id"})
    left["_row"] = np.arange(len(left))
    left = left.dropna(subset=["pfr_player_id"]).sort_values("order")

    right = shares.sort_values("order")

    # merge_asof takes the most recent snap history strictly at or before this
    # week; share_before is already lagged, so the same week is safe to match.
    matched = pd.merge_asof(
        left,
        right,
        on="order",
        by="pfr_player_id",
        direction="backward",
    )

    out = pd.Series(np.nan, index=frame.index)
    out.iloc[matched["_row"].to_numpy()] = matched["share_before"].to_numpy()
    return out


def burden(
    reports: pd.DataFrame,
    shares: pd.DataFrame | None = None,
    bridge: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per team-week: how much of the team's usual playing time is missing.

    With ``shares`` supplied the weight is the player's own recent snap share.
    Without it, every player falls back to the average share for his position,
    which is what happens for seasons before snap counts exist.
    """
    if reports.empty:
        return pd.DataFrame(
            columns=["season", "week", "team", "out_count", "weighted_out"]
        )

    frame = reports.copy()
    frame["status_weight"] = frame["report_status"].map(STATUS_WEIGHT).fillna(0.0)

    fallback = frame["position"].map(POSITION_SHARE).fillna(DEFAULT_SHARE)
    if shares is not None and not shares.empty:
        measured = _attach_shares(frame, shares, bridge)
        frame["usage"] = measured.fillna(fallback)
    else:
        frame["usage"] = fallback

    frame["weighted"] = frame["status_weight"] * frame["usage"]

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
