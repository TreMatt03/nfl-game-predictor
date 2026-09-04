"""Test whether injuries and quarterback draft capital add anything.

Two candidates, both aimed at the same blind spot: the model rates a team from
its results, so it cannot see that the roster producing those results is not
the roster about to play.

    injuries    -- who is unavailable this week, from the official reports
    draft slot  -- a better prior than flat replacement level for a passer
                   with almost no professional record

As in scripts/experiment_context.py, each is judged on whether it improves
walk-forward log loss on top of the existing features, and the draft prior is
estimated only from seasons before the evaluation window.

Run with ``python scripts/experiment_roster.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflpred import data, pipeline, quarterback
from nflpred.dataset import FEATURE_COLUMNS
from nflpred.model import score, walk_forward

FIRST_TEST_SEASON = 2012

# Injury reports begin in 2009, so evaluation starts once a few seasons exist.
FIRST_INJURY_TEST_SEASON = 2012

# Rough importance by position, used to weight who is missing. A starting
# tackle matters more than a third safety, and treating every absence as equal
# would drown the signal in inactive special-teamers.
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
    "LB": 1.0,
    "DT": 1.0,
    "TE": 0.9,
    "RB": 0.9,
    "OLB": 1.0,
    "ILB": 0.8,
    "FS": 1.0,
    "SS": 1.0,
    "NT": 0.8,
    "FB": 0.3,
    "K": 0.3,
    "P": 0.2,
    "LS": 0.1,
}

# Doubtful players usually sit; questionable ones usually play.
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}


def injury_burden(injuries: pd.DataFrame) -> pd.DataFrame:
    """Per team-week: how many players are missing, and how much they matter."""
    frame = injuries.copy()
    frame["status_weight"] = frame["report_status"].map(STATUS_WEIGHT).fillna(0.0)
    frame["position_weight"] = frame["position"].map(POSITION_WEIGHT).fillna(0.9)
    frame["weighted"] = frame["status_weight"] * frame["position_weight"]

    grouped = (
        frame.groupby(["season", "week", "team"], observed=True)
        .agg(
            out_count=("status_weight", lambda s: (s == 1.0).sum()),
            weighted_out=("weighted", "sum"),
        )
        .reset_index()
    )
    return grouped


def attach_injuries(games: pd.DataFrame, burden: pd.DataFrame) -> pd.DataFrame:
    """Add home-minus-away injury differentials to the game table."""
    out = games.copy()

    for side in ("home", "away"):
        renamed = burden.rename(
            columns={
                "team": f"{side}_team",
                "out_count": f"{side}_out",
                "weighted_out": f"{side}_weighted_out",
            }
        )
        out = out.merge(
            renamed, on=["season", "week", f"{side}_team"], how="left"
        )

    for column in ("out", "weighted_out"):
        for side in ("home", "away"):
            out[f"{side}_{column}"] = out[f"{side}_{column}"].fillna(0.0)
        # Positive means the away team is more banged up, favouring the home side.
        out[f"injury_{column}_diff"] = out[f"away_{column}"] - out[f"home_{column}"]

    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    schedules = data.load_schedules()
    draft = data.load_draft_picks()

    print("\n=== 1. Quarterback draft capital as a prior ===\n")
    baseline_games, _, _, starters = pipeline.build()
    base_wf = walk_forward(baseline_games, "logistic", FIRST_TEST_SEASON)
    truth = base_wf["home_win"].to_numpy()
    base = score(truth, base_wf["pred"].to_numpy())

    priors = quarterback.draft_priors(
        starters, draft, upto_season=FIRST_TEST_SEASON - 1
    )
    print("Priors estimated on 2006-%d only:" % (FIRST_TEST_SEASON - 1))
    qbs = draft[draft["position"] == "QB"][["gsis_id", "pick"]].dropna(
        subset=["gsis_id"]
    )
    buckets = (
        starters.merge(qbs, left_on="passer_player_id", right_on="gsis_id", how="left")
        .assign(bucket=lambda d: d["pick"].map(quarterback._draft_bucket))
        .drop_duplicates("passer_player_id")
        .set_index("passer_player_id")["bucket"]
    )
    summary = (
        pd.DataFrame({"bucket": buckets, "prior": priors})
        .groupby("bucket")
        .agg(passers=("prior", "size"), prior=("prior", "first"))
        .round(4)
    )
    print(summary.to_string())

    draft_games, _, _, _ = pipeline.build(qb_priors=priors)
    draft_wf = walk_forward(draft_games, "logistic", FIRST_TEST_SEASON)
    draft_score = score(truth, draft_wf["pred"].to_numpy())

    rows = [
        {"model": "flat replacement prior", **base.as_row("")},
        {"model": "draft-slot prior", **draft_score.as_row("")},
    ]
    table = pd.DataFrame(rows).drop(columns=["model"], errors="ignore")
    table.insert(0, "prior", ["flat replacement", "draft slot"])
    print("\n" + table.to_string(index=False))
    print("delta: %+.5f log loss" % (draft_score.log_loss - base.log_loss))

    _inexperienced_only(baseline_games, base_wf, draft_wf, starters)

    print("\n\n=== 2. Injury reports ===\n")
    injuries = data.load_injuries(list(range(2009, 2026)))
    burden = injury_burden(injuries)
    games = attach_injuries(baseline_games, burden)
    games = games[games["season"] >= FIRST_INJURY_TEST_SEASON].reset_index(drop=True)

    inj_base = walk_forward(games, "logistic", FIRST_INJURY_TEST_SEASON)
    inj_truth = inj_base["home_win"].to_numpy()
    inj_reference = score(inj_truth, inj_base["pred"].to_numpy())

    candidates = {
        "count of players Out": ["injury_out_diff"],
        "importance-weighted": ["injury_weighted_out_diff"],
        "both": ["injury_out_diff", "injury_weighted_out_diff"],
    }

    rows = [{"added": "nothing", **inj_reference.as_row("")}]
    for name, columns in candidates.items():
        result = walk_forward(
            games,
            "logistic",
            FIRST_INJURY_TEST_SEASON,
            features=FEATURE_COLUMNS + columns,
        )
        scored = score(inj_truth, result["pred"].to_numpy())
        row = scored.as_row("")
        row["delta"] = round(scored.log_loss - inj_reference.log_loss, 5)
        rows.append({"added": name, **row})

    print(pd.DataFrame(rows).drop(columns=["model"]).to_string(index=False))

    print("\nHow lopsided do injuries actually get?")
    diff = games["injury_weighted_out_diff"]
    print("  weighted differential: mean %+.2f, sd %.2f, 5th/95th %+.1f/%+.1f"
          % (diff.mean(), diff.std(), diff.quantile(0.05), diff.quantile(0.95)))
    lopsided = games[diff.abs() >= diff.abs().quantile(0.9)]
    print("  in the 10%% most lopsided games, the healthier side won %.1f%%"
          % (100 * np.where(
              lopsided["injury_weighted_out_diff"] > 0,
              lopsided["home_win"], 1 - lopsided["home_win"]).mean()))


def _inexperienced_only(games, base_wf, draft_wf, starters) -> None:
    """The draft prior can only matter where a passer is inexperienced."""
    lookup = starters.set_index(["game_id", "team"])["qb_starts"]
    keys_home = list(zip(base_wf["game_id"], base_wf["home_team"]))
    keys_away = list(zip(base_wf["game_id"], base_wf["away_team"]))

    min_starts = np.minimum(
        pd.Series(keys_home).map(lookup).fillna(99).to_numpy(),
        pd.Series(keys_away).map(lookup).fillna(99).to_numpy(),
    )
    mask = min_starts < 6
    truth = base_wf["home_win"].to_numpy()

    print("\nRestricted to the %d games with a quarterback under 6 starts:"
          % mask.sum())
    flat = score(truth[mask], base_wf["pred"].to_numpy()[mask])
    smart = score(truth[mask], draft_wf["pred"].to_numpy()[mask])
    print("  flat replacement prior : log loss %.4f" % flat.log_loss)
    print("  draft-slot prior       : log loss %.4f  (%+.5f)"
          % (smart.log_loss, smart.log_loss - flat.log_loss))


if __name__ == "__main__":
    main()
