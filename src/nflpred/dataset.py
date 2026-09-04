"""Assemble the modelling table: one row per game, labelled with the result.

Features are expressed as home-minus-away differentials so the table is
symmetric -- swapping the two teams flips every feature's sign and the target.
That stops the model from learning anything about a team's identity, only about
the gap between the two sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FORM_COLUMNS, OFFENSE_METRICS

# Two different views of the same form numbers.
#   net_*      -- overall team quality gap (home net minus away net)
#   homeoff_*  -- home offence against away defence
#   awayoff_*  -- away offence against home defence
NET_FEATURES = [f"net_{m}" for m in OFFENSE_METRICS]
MATCHUP_FEATURES = [f"homeoff_{m}" for m in OFFENSE_METRICS] + [
    f"awayoff_{m}" for m in OFFENSE_METRICS
]
CONTEXT_FEATURES = ["elo_diff", "rest_diff", "div_game", "neutral_site"]
QB_FEATURES = ["qb_diff"]
INJURY_FEATURES = ["injury_diff"]

# Candidate feature sets, compared honestly in scripts/experiment.py.
# EPA per play, passing EPA and success rate all measure close to the same
# thing, and all of them are already baked into Elo. Including the lot makes
# the coefficients unstable -- signs flip and a factor that helped a team gets
# reported as having hurt it -- which matters here because those coefficients
# are published as the per-game explanation. This set keeps the metrics that
# carry distinct information.
LEAN_METRICS = ["success_rate", "explosive_rate", "turnover_rate"]
LEAN_FEATURES = [f"net_{m}" for m in LEAN_METRICS]

FEATURE_SETS = {
    "elo_only": ["elo_diff"],
    "elo_qb": ["elo_diff"] + QB_FEATURES,
    "net": NET_FEATURES + CONTEXT_FEATURES,
    "net_qb": NET_FEATURES + CONTEXT_FEATURES + QB_FEATURES,
    "lean_qb": LEAN_FEATURES + CONTEXT_FEATURES + QB_FEATURES,
    "lean_qb_injury": LEAN_FEATURES + CONTEXT_FEATURES + QB_FEATURES + INJURY_FEATURES,
    "matchup_qb": MATCHUP_FEATURES + CONTEXT_FEATURES + QB_FEATURES,
    "full": NET_FEATURES + MATCHUP_FEATURES + CONTEXT_FEATURES + QB_FEATURES,
}

FEATURE_COLUMNS = FEATURE_SETS["lean_qb_injury"]

_METRIC_LABELS = {
    "epa_play": "EPA per play",
    "pass_epa": "Passing efficiency",
    "rush_epa": "Rushing efficiency",
    "success_rate": "Success rate",
    "explosive_rate": "Explosive plays",
    "turnover_rate": "Turnovers",
    "sack_rate": "Pressure game",
    "third_down_rate": "Third downs",
}

FEATURE_LABELS = {
    **{f"net_{m}": f"{label} (overall)" for m, label in _METRIC_LABELS.items()},
    **{f"homeoff_{m}": f"{label}, home offence" for m, label in _METRIC_LABELS.items()},
    **{f"awayoff_{m}": f"{label}, away offence" for m, label in _METRIC_LABELS.items()},
    "elo_diff": "Elo rating gap",
    "rest_diff": "Rest advantage",
    "div_game": "Divisional matchup",
    "neutral_site": "Neutral site",
    "qb_diff": "Quarterback play",
    "injury_diff": "Injury report",
}


def _net_form(team_form: pd.DataFrame) -> pd.DataFrame:
    """Collapse each team's offensive and defensive form into one net number.

    For most metrics a high offensive value and a low value allowed are both
    good, so offence-minus-defence is a sensible single summary. Turnovers and
    sacks run the other way; the model learns that from the coefficient sign
    rather than us hard-coding it here.
    """
    net = pd.DataFrame(index=team_form.index)
    for metric in OFFENSE_METRICS:
        net[metric] = team_form[f"off_{metric}_form"] - team_form[f"def_{metric}_form"]
    return net


def _side_view(combined: pd.DataFrame, side: str) -> pd.DataFrame:
    """Prefix every form column with ``home_`` or ``away_`` for merging."""
    value_cols = [c for c in combined.columns if c not in ("game_id", "team")]
    renamed = combined.rename(columns={"team": f"{side}_team"})
    return renamed.rename(columns={c: f"{side}_{c}" for c in value_cols})


def build_dataset(
    team_games: pd.DataFrame,
    schedules: pd.DataFrame,
    elo_pre: pd.DataFrame,
) -> pd.DataFrame:
    """Join home and away form onto the schedule and label completed games."""
    per_team = team_games.set_index(["game_id", "team"])[FORM_COLUMNS]
    net = _net_form(per_team).add_prefix("net_")
    combined = per_team.join(net).reset_index()

    games = schedules.merge(elo_pre, on="game_id", how="inner")
    games = games.merge(_side_view(combined, "home"), on=["game_id", "home_team"])
    games = games.merge(_side_view(combined, "away"), on=["game_id", "away_team"])

    for metric in OFFENSE_METRICS:
        games[f"net_{metric}"] = (
            games[f"home_net_{metric}"] - games[f"away_net_{metric}"]
        )
        games[f"homeoff_{metric}"] = (
            games[f"home_off_{metric}_form"] - games[f"away_def_{metric}_form"]
        )
        games[f"awayoff_{metric}"] = (
            games[f"away_off_{metric}_form"] - games[f"home_def_{metric}_form"]
        )

    games["elo_diff"] = games["home_elo_pre"] - games["away_elo_pre"]
    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    games["neutral_site"] = (games["location"] != "Home").astype(int)
    games["div_game"] = games["div_game"].fillna(0).astype(int)

    games["home_win"] = np.where(
        games["result"] > 0, 1, np.where(games["result"] < 0, 0, np.nan)
    )
    games["vegas_home_prob"] = _devigged_home_prob(games)

    return games


def _american_to_prob(odds: pd.Series) -> np.ndarray:
    """Convert American moneyline odds to a raw implied probability."""
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def _devigged_home_prob(games: pd.DataFrame) -> pd.Series:
    """Vegas home win probability with the bookmaker's margin divided out.

    Raw implied probabilities sum to more than one because that overround is the
    house edge. Normalising by the sum is the standard first-order correction
    and gives an honest benchmark to measure the model against.
    """
    home_raw = _american_to_prob(games["home_moneyline"])
    away_raw = _american_to_prob(games["away_moneyline"])
    total = home_raw + away_raw
    return pd.Series(np.where(total > 0, home_raw / total, np.nan), index=games.index)


def modelling_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Rows complete enough to train or evaluate on.

    Early-season rows where a team has almost no history produce unstable form
    values, so they are dropped rather than imputed.
    """
    frame = games.dropna(subset=FEATURE_COLUMNS + ["home_win"]).copy()
    return frame.reset_index(drop=True)
