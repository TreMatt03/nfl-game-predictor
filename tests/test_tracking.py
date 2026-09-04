"""Tests for the prediction log and its scoring.

The central guarantee is that a logged prediction is never rewritten. If a
rerun could overwrite an earlier call with a better-informed one, the published
track record would slowly become fiction, and it would look like an improving
model rather than a bug.
"""

import pandas as pd
import pytest

from nflpred import tracking
from nflpred.predict import Slate


def _slate(rows, season=2026, week=1):
    frame = pd.DataFrame(rows)
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    frame["season"] = season
    frame["week"] = week
    return Slate(season=season, week=week, games=frame)


def _game(game_id, home, away, prob, gameday="2026-09-13"):
    return {
        "game_id": game_id,
        "home_team": home,
        "away_team": away,
        "home_win_prob": prob,
        "favorite": home if prob >= 0.5 else away,
        "gameday": gameday,
        "home_qb": "H.Qb",
        "away_qb": "A.Qb",
    }


def test_predictions_are_written_once(tmp_path):
    path = tmp_path / "log.csv"
    slate = _slate([_game("g1", "AAA", "BBB", 0.7)])

    assert tracking.log_predictions(slate, path) == 1
    assert tracking.log_predictions(slate, path) == 0
    assert len(tracking.load_log(path)) == 1


def test_a_rerun_cannot_revise_an_earlier_call(tmp_path):
    """The original prediction stands, even if the model has since changed."""
    path = tmp_path / "log.csv"
    tracking.log_predictions(_slate([_game("g1", "AAA", "BBB", 0.70)]), path)

    # Same game, different probability -- e.g. after a corrected starter.
    tracking.log_predictions(_slate([_game("g1", "AAA", "BBB", 0.95)]), path)

    logged = tracking.load_log(path)
    assert len(logged) == 1
    assert logged.iloc[0]["home_win_prob"] == pytest.approx(0.70)


def test_new_games_append_alongside_existing_ones(tmp_path):
    path = tmp_path / "log.csv"
    tracking.log_predictions(_slate([_game("g1", "AAA", "BBB", 0.7)]), path)
    tracking.log_predictions(
        _slate([_game("g1", "AAA", "BBB", 0.7), _game("g2", "CCC", "DDD", 0.4)]), path
    )

    assert set(tracking.load_log(path)["game_id"]) == {"g1", "g2"}


def _schedule(rows):
    """A schedule frame that keeps its columns even when it has no rows."""
    return pd.DataFrame(rows, columns=["game_id", "result"])


def test_scoring_marks_hits_and_misses():
    predictions = pd.DataFrame(
        [
            _game("g1", "AAA", "BBB", 0.7) | {"season": 2026, "week": 1},
            _game("g2", "CCC", "DDD", 0.3) | {"season": 2026, "week": 1},
        ]
    )
    # g1: home won, predicted home -> correct. g2: home won, predicted away.
    schedules = _schedule(
        [{"game_id": "g1", "result": 7.0}, {"game_id": "g2", "result": 3.0}]
    )

    scored = tracking.score_history(predictions, schedules)

    assert dict(zip(scored["game_id"], scored["correct"])) == {"g1": 1, "g2": 0}
    assert dict(zip(scored["game_id"], scored["winner"])) == {"g1": "AAA", "g2": "CCC"}


def test_unplayed_games_are_not_scored():
    predictions = pd.DataFrame(
        [_game("g1", "AAA", "BBB", 0.7) | {"season": 2026, "week": 1}]
    )
    scored = tracking.score_history(predictions, _schedule([]))
    assert scored.empty


def test_ties_are_excluded_rather_than_counted_as_losses():
    predictions = pd.DataFrame(
        [_game("g1", "AAA", "BBB", 0.7) | {"season": 2026, "week": 1}]
    )
    schedules = _schedule([{"game_id": "g1", "result": 0.0}])

    assert tracking.score_history(predictions, schedules).empty


def test_summary_counts_the_record():
    predictions = pd.DataFrame(
        [
            _game("g1", "AAA", "BBB", 0.7) | {"season": 2026, "week": 1},
            _game("g2", "CCC", "DDD", 0.6) | {"season": 2026, "week": 1},
            _game("g3", "EEE", "FFF", 0.8) | {"season": 2026, "week": 1},
        ]
    )
    schedules = _schedule(
        [
            {"game_id": "g1", "result": 7.0},
            {"game_id": "g2", "result": -3.0},
            {"game_id": "g3", "result": 10.0},
        ]
    )

    summary = tracking.summarise(tracking.score_history(predictions, schedules))

    assert (summary["games"], summary["wins"], summary["losses"]) == (3, 2, 1)
    assert summary["accuracy"] == pytest.approx(2 / 3)


def test_summary_of_nothing_is_empty_not_an_error():
    assert tracking.summarise(pd.DataFrame())["games"] == 0


def test_latest_week_returns_only_the_most_recent():
    predictions = pd.DataFrame(
        [
            _game("g1", "AAA", "BBB", 0.7) | {"season": 2026, "week": 1},
            _game("g2", "CCC", "DDD", 0.6) | {"season": 2026, "week": 2},
        ]
    )
    schedules = _schedule(
        [{"game_id": "g1", "result": 7.0}, {"game_id": "g2", "result": 3.0}]
    )

    recent = tracking.latest_week(tracking.score_history(predictions, schedules))

    assert list(recent["game_id"]) == ["g2"]


def test_missing_log_reads_as_empty(tmp_path):
    assert tracking.load_log(tmp_path / "absent.csv").empty
