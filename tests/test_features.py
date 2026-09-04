"""Tests for the feature layer.

The leakage tests are the important ones. Everything else in this project can
be wrong and the result is a worse model; leakage makes the reported numbers
meaningless while still looking good, so it gets checked directly.
"""

import numpy as np
import pandas as pd
import pytest

from nflpred import features


def _team_games(values):
    """A single team's game log with a controllable offensive EPA sequence."""
    n = len(values)
    frame = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "team": ["AAA"] * n,
            "opponent": ["BBB"] * n,
            "gameday": pd.date_range("2020-09-01", periods=n, freq="7D"),
        }
    )
    for metric in features.OFFENSE_METRICS:
        frame[f"off_{metric}"] = values
        frame[f"def_{metric}"] = 0.0
    return frame


def test_form_never_sees_the_current_game():
    """The whole model rests on this: row n must not depend on row n."""
    baseline = features.add_form(_team_games([1.0, 2.0, 3.0, 4.0, 5.0]))

    # Rewrite only the last game's result to something absurd.
    tampered = features.add_form(_team_games([1.0, 2.0, 3.0, 4.0, 999.0]))

    assert baseline["off_epa_play_form"].equals(tampered["off_epa_play_form"])


def test_form_is_undefined_before_enough_history():
    form = features.add_form(_team_games([1.0, 2.0, 3.0, 4.0]))["off_epa_play_form"]
    assert np.isnan(form.iloc[0])


def test_form_tracks_recent_games_more_heavily():
    """A team that improves should show rising form, below its latest game."""
    form = features.add_form(_team_games([0.0, 0.0, 0.0, 1.0, 1.0]))
    values = form["off_epa_play_form"].to_numpy()
    assert values[-1] > values[-2]
    assert values[-1] < 1.0


def _schedule(rows):
    frame = pd.DataFrame(rows)
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    return frame


def test_elo_is_zero_sum_within_a_season():
    """Points taken off the loser are exactly the points given to the winner."""
    schedules = _schedule(
        [
            {
                "game_id": "g1",
                "season": 2020,
                "gameday": "2020-09-13",
                "home_team": "AAA",
                "away_team": "BBB",
                "result": 7.0,
                "location": "Home",
            },
            {
                "game_id": "g2",
                "season": 2020,
                "gameday": "2020-09-20",
                "home_team": "BBB",
                "away_team": "AAA",
                "result": 3.0,
                "location": "Home",
            },
        ]
    )
    _, final = features.elo_ratings(schedules)
    assert final["AAA"] + final["BBB"] == pytest.approx(3000.0)


def test_elo_pre_game_rating_excludes_that_game():
    schedules = _schedule(
        [
            {
                "game_id": "g1",
                "season": 2020,
                "gameday": "2020-09-13",
                "home_team": "AAA",
                "away_team": "BBB",
                "result": 21.0,
                "location": "Home",
            }
        ]
    )
    pre, final = features.elo_ratings(schedules)
    assert pre.iloc[0]["home_elo_pre"] == 1500.0
    assert final["AAA"] > 1500.0


def test_winning_raises_rating_and_losing_lowers_it():
    schedules = _schedule(
        [
            {
                "game_id": "g1",
                "season": 2020,
                "gameday": "2020-09-13",
                "home_team": "AAA",
                "away_team": "BBB",
                "result": 14.0,
                "location": "Home",
            }
        ]
    )
    _, final = features.elo_ratings(schedules)
    assert final["AAA"] > final["BBB"]


def test_regression_pulls_ratings_toward_the_mean():
    regressed = features.regress_ratings({"AAA": 1700.0, "BBB": 1300.0}, regress=0.25)
    assert regressed["AAA"] == pytest.approx(1650.0)
    assert regressed["BBB"] == pytest.approx(1350.0)


def test_regression_compounds_over_multiple_seasons():
    once = features.regress_ratings({"AAA": 1700.0}, seasons=1, regress=0.25)
    twice = features.regress_ratings({"AAA": 1700.0}, seasons=2, regress=0.25)
    assert twice["AAA"] < once["AAA"] < 1700.0
