"""Model definitions and honest, time-respecting evaluation.

The evaluation here is walk-forward: to score season S the model is trained only
on seasons before S. A plain random split would shuffle future games into the
training set and inflate every number on this page, so it is never used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import FEATURE_COLUMNS


def make_model(kind: str = "logistic"):
    """Build one of the candidate classifiers.

    Logistic regression is the default because it is naturally well calibrated
    and its coefficients give a per-game explanation for free -- which is the
    part of this project that is actually interesting.
    """
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                # C chosen by the walk-forward sweep in scripts/experiment.py.
                ("clf", LogisticRegression(C=0.01, max_iter=2000)),
            ]
        )
    if kind == "gbm":
        base = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=0,
        )
        # Trees are confident in a way they have not earned; isotonic
        # calibration pulls the probabilities back to reality.
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    raise ValueError(f"unknown model kind: {kind}")


@dataclass
class Scores:
    """Standard classification metrics for a set of probability forecasts."""

    n: int
    accuracy: float
    log_loss: float
    brier: float
    auc: float

    def as_row(self, name: str) -> dict:
        return {
            "model": name,
            "games": self.n,
            "accuracy": round(self.accuracy, 4),
            "log_loss": round(self.log_loss, 4),
            "brier": round(self.brier, 4),
            "auc": round(self.auc, 4),
        }


def score(y_true: np.ndarray, prob: np.ndarray) -> Scores:
    """Score forecasts, ignoring rows where the forecast is undefined."""
    mask = ~np.isnan(prob)
    y_true, prob = np.asarray(y_true)[mask], np.asarray(prob)[mask]
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    return Scores(
        n=len(y_true),
        accuracy=accuracy_score(y_true, prob > 0.5),
        log_loss=log_loss(y_true, prob, labels=[0, 1]),
        brier=brier_score_loss(y_true, prob),
        auc=roc_auc_score(y_true, prob),
    )


def walk_forward(
    games: pd.DataFrame,
    kind: str = "logistic",
    first_test_season: int = 2012,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Predict every season from a model that has only seen earlier seasons.

    Returns the input rows for the tested seasons with a ``pred`` column added.
    """
    features = features or FEATURE_COLUMNS
    seasons = sorted(s for s in games["season"].unique() if s >= first_test_season)
    out = []

    for season in seasons:
        train = games[games["season"] < season]
        test = games[games["season"] == season]
        if train.empty or test.empty:
            continue

        model = make_model(kind)
        model.fit(train[features], train["home_win"])

        predicted = test.copy()
        predicted["pred"] = model.predict_proba(test[features])[:, 1]
        out.append(predicted)

    return pd.concat(out, ignore_index=True)


def fit_final(games: pd.DataFrame, kind: str = "logistic"):
    """Train on everything available, for use on genuinely unplayed games."""
    model = make_model(kind)
    model.fit(games[FEATURE_COLUMNS], games["home_win"])
    return model
