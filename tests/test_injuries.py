"""Tests for injury burden and its effect on the game table."""

import pandas as pd
import pytest

from nflpred import injuries


def _reports(rows):
    return pd.DataFrame(
        rows, columns=["season", "week", "team", "position", "report_status"]
    )


def test_out_players_count_more_than_questionable_ones():
    reports = _reports(
        [
            (2025, 1, "AAA", "WR", "Out"),
            (2025, 1, "BBB", "WR", "Questionable"),
        ]
    )
    burden = injuries.burden(reports).set_index("team")

    assert burden.loc["AAA", "weighted_out"] > burden.loc["BBB", "weighted_out"]


def test_position_importance_is_reflected():
    """A missing quarterback outweighs a missing long snapper."""
    reports = _reports(
        [(2025, 1, "AAA", "QB", "Out"), (2025, 1, "BBB", "LS", "Out")]
    )
    burden = injuries.burden(reports).set_index("team")

    assert burden.loc["AAA", "weighted_out"] > burden.loc["BBB", "weighted_out"]


def test_out_count_ignores_questionable():
    reports = _reports(
        [
            (2025, 1, "AAA", "WR", "Out"),
            (2025, 1, "AAA", "CB", "Out"),
            (2025, 1, "AAA", "TE", "Questionable"),
        ]
    )
    assert injuries.burden(reports).iloc[0]["out_count"] == 2


def test_unlisted_status_carries_no_weight():
    reports = _reports([(2025, 1, "AAA", "WR", None)])
    assert injuries.burden(reports).iloc[0]["weighted_out"] == pytest.approx(0.0)


def test_unknown_position_gets_the_default_weight():
    reports = _reports([(2025, 1, "AAA", "ATHLETE", "Out")])
    burden = injuries.burden(reports)
    assert burden.iloc[0]["weighted_out"] == pytest.approx(
        injuries.DEFAULT_POSITION_WEIGHT
    )


def _games():
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2025,
                "week": 1,
                "home_team": "AAA",
                "away_team": "BBB",
            }
        ]
    )


def test_differential_favours_the_healthier_side():
    """Positive means the away team is more banged up, favouring home."""
    reports = _reports(
        [(2025, 1, "BBB", "QB", "Out"), (2025, 1, "BBB", "T", "Out")]
    )
    attached = injuries.attach(_games(), injuries.burden(reports))

    assert attached.iloc[0]["injury_diff"] > 0


def test_differential_is_signed_the_other_way_too():
    reports = _reports([(2025, 1, "AAA", "QB", "Out")])
    attached = injuries.attach(_games(), injuries.burden(reports))

    assert attached.iloc[0]["injury_diff"] < 0


def test_teams_with_no_report_are_treated_as_healthy():
    """A missing report means nothing was filed, not an unknown injury load."""
    attached = injuries.attach(_games(), injuries.burden(_reports([])))

    assert attached.iloc[0]["injury_diff"] == pytest.approx(0.0)


def test_attach_survives_empty_reports_without_losing_games():
    attached = injuries.attach(_games(), injuries.burden(_reports([])))
    assert len(attached) == 1
    assert "injury_diff" in attached


def test_evenly_injured_teams_cancel_out():
    reports = _reports(
        [(2025, 1, "AAA", "WR", "Out"), (2025, 1, "BBB", "WR", "Out")]
    )
    attached = injuries.attach(_games(), injuries.burden(reports))

    assert attached.iloc[0]["injury_diff"] == pytest.approx(0.0)


def test_reports_from_another_week_do_not_leak_in():
    reports = _reports([(2025, 2, "BBB", "QB", "Out")])
    attached = injuries.attach(_games(), injuries.burden(reports))

    assert attached.iloc[0]["injury_diff"] == pytest.approx(0.0)
