"""Compare feature sets and regularisation strengths, walk-forward.

Exists so the choices baked into the final model are recorded as evidence
rather than asserted. Run with ``python scripts/experiment.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflpred import pipeline
from nflpred.dataset import FEATURE_SETS
from nflpred.model import score

FIRST_TEST_SEASON = 2012


def walk_forward_probs(games: pd.DataFrame, features: list[str], C: float):
    """Out-of-sample probabilities for every season after the burn-in."""
    seasons = sorted(s for s in games["season"].unique() if s >= FIRST_TEST_SEASON)
    out = []

    for season in seasons:
        train = games[games["season"] < season]
        test = games[games["season"] == season]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=5000)),
            ]
        )
        model.fit(train[features], train["home_win"])
        block = test[["home_win"]].copy()
        block["pred"] = model.predict_proba(test[features])[:, 1]
        out.append(block)

    return pd.concat(out, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    games, team_games, _, _ = pipeline.build()
    pipeline.save(games, team_games)

    rows = []
    for name, features in FEATURE_SETS.items():
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            result = walk_forward_probs(games, features, C)
            scores = score(result["home_win"].to_numpy(), result["pred"].to_numpy())
            rows.append({"features": name, "C": C, **scores.as_row("")})

    table = pd.DataFrame(rows).drop(columns=["model"])
    table = table.sort_values("log_loss")

    print("\nWalk-forward %d-%d, sorted by log loss\n" % (
        FIRST_TEST_SEASON, games.season.max()))
    print(table.to_string(index=False))

    best = table.iloc[0]
    print(f"\nbest: features={best.features} C={best.C} log_loss={best.log_loss}")


if __name__ == "__main__":
    main()
