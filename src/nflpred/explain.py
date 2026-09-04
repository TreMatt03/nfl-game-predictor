"""Explain why the model favoured one side.

A bare win probability is not much of a product. For a linear model the
decomposition is exact rather than approximated: the log-odds is a sum of
per-feature terms, so each term is honestly "how many log-odds this factor
contributed", and they add up to the prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import FEATURE_COLUMNS, FEATURE_LABELS


def contributions(model, rows: pd.DataFrame) -> pd.DataFrame:
    """Per-feature log-odds contribution for each row.

    Expects the logistic pipeline from ``make_model("logistic")``: a scaler
    followed by a linear classifier. Tree models have no such exact
    decomposition and are rejected rather than silently approximated.
    """
    try:
        scaler = model.named_steps["scale"]
        clf = model.named_steps["clf"]
    except (AttributeError, KeyError) as exc:
        raise TypeError(
            "contributions() needs the linear pipeline; "
            "tree models have no exact decomposition"
        ) from exc

    scaled = scaler.transform(rows[FEATURE_COLUMNS])
    terms = scaled * clf.coef_[0]

    return pd.DataFrame(terms, columns=FEATURE_COLUMNS, index=rows.index)


def top_drivers(model, rows: pd.DataFrame, n: int = 3) -> list[list[dict]]:
    """The ``n`` factors that moved each prediction most, largest first.

    Sign is relative to the home team: a positive contribution pushed the
    forecast toward a home win, negative toward the away team.
    """
    terms = contributions(model, rows)
    out = []

    for _, row in terms.iterrows():
        ranked = row.reindex(row.abs().sort_values(ascending=False).index)[:n]
        out.append(
            [
                {
                    "factor": FEATURE_LABELS.get(name, name),
                    "effect": float(value),
                    "favors": "home" if value > 0 else "away",
                }
                for name, value in ranked.items()
            ]
        )
    return out


def global_importance(model) -> pd.DataFrame:
    """Overall feature weight, as absolute standardised coefficients.

    Because the inputs were standardised, coefficients are directly comparable
    across features with different units.
    """
    clf = model.named_steps["clf"]
    return (
        pd.DataFrame(
            {
                "feature": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLUMNS],
                "coefficient": clf.coef_[0],
                "abs_coefficient": np.abs(clf.coef_[0]),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
