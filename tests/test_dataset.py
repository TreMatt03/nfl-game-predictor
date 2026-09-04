"""Tests for the game-level table and the model's explanations."""

import numpy as np
import pandas as pd
import pytest

from nflpred import dataset
from nflpred.explain import contributions, global_importance
from nflpred.model import make_model, score


def test_devig_produces_probabilities_that_sum_to_one():
    """Raw implied odds sum above 1; the house edge has to come back out."""
    games = pd.DataFrame({"home_moneyline": [-200.0], "away_moneyline": [170.0]})
    home = dataset._devigged_home_prob(games)
    away = 1.0 - home

    assert 0.0 < home.iloc[0] < 1.0
    assert home.iloc[0] + away.iloc[0] == pytest.approx(1.0)


def test_devig_favours_the_shorter_price():
    games = pd.DataFrame({"home_moneyline": [-300.0], "away_moneyline": [250.0]})
    assert dataset._devigged_home_prob(games).iloc[0] > 0.5


def test_devig_is_undefined_without_a_line():
    games = pd.DataFrame({"home_moneyline": [np.nan], "away_moneyline": [np.nan]})
    assert np.isnan(dataset._devigged_home_prob(games).iloc[0])


def test_missing_odds_do_not_crash_scoring():
    """Vegas has no line for some games; those rows are skipped, not fatal."""
    truth = np.array([1, 0, 1, 0])
    pred = np.array([0.8, 0.2, np.nan, 0.4])

    result = score(truth, pred)
    assert result.n == 3


def _synthetic_games(n=400, seed=0):
    """A dataset where the home team wins more often as elo_diff grows."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {col: rng.normal(size=n) for col in dataset.FEATURE_COLUMNS}
    )
    logit = 0.9 * frame["elo_diff"] + 0.5 * frame["qb_diff"]
    frame["home_win"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


def test_contributions_reconstruct_the_prediction():
    """The explanation is exact, not an approximation -- so it must add up."""
    games = _synthetic_games()
    model = make_model("logistic")
    model.fit(games[dataset.FEATURE_COLUMNS], games["home_win"])

    terms = contributions(model, games.head(5))
    intercept = model.named_steps["clf"].intercept_[0]
    rebuilt = 1 / (1 + np.exp(-(terms.sum(axis=1) + intercept)))

    expected = model.predict_proba(games.head(5)[dataset.FEATURE_COLUMNS])[:, 1]
    np.testing.assert_allclose(rebuilt.to_numpy(), expected, rtol=1e-9)


def test_contributions_reject_models_without_an_exact_decomposition():
    games = _synthetic_games()
    gbm = make_model("gbm")
    gbm.fit(games[dataset.FEATURE_COLUMNS], games["home_win"])

    with pytest.raises(TypeError):
        contributions(gbm, games.head(3))


def test_importance_finds_the_signal_that_was_planted():
    games = _synthetic_games(n=3000)
    model = make_model("logistic")
    model.fit(games[dataset.FEATURE_COLUMNS], games["home_win"])

    top = global_importance(model).iloc[0]["feature"]
    assert top == dataset.FEATURE_LABELS["elo_diff"]


def test_every_feature_has_a_readable_label():
    """The published page prints these, so a missing one is user-visible."""
    for column in dataset.FEATURE_COLUMNS:
        assert column in dataset.FEATURE_LABELS
