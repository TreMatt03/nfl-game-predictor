"""Tests for injury burden and its effect on the game table."""

import pandas as pd
import pytest

from nflpred import injuries


def _reports(rows):
    frame = pd.DataFrame(
        rows, columns=["season", "week", "team", "position", "report_status"]
    )
    frame["gsis_id"] = [f"p{i}" for i in range(len(frame))]
    return frame


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
        injuries.DEFAULT_SHARE
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


def _snaps(rows):
    """Snap-count rows: (season, week, pfr_id, offense_pct, defense_pct)."""
    return pd.DataFrame(
        rows,
        columns=["season", "week", "pfr_player_id", "offense_pct", "defense_pct"],
    )


def _bridge(pairs):
    return pd.DataFrame(pairs, columns=["gsis_id", "pfr_id"])


def test_snap_share_excludes_the_current_week():
    """A player who sat has no snaps that week; using them would be circular."""
    snaps = _snaps(
        [
            (2025, 1, "aa", 1.0, 0.0),
            (2025, 2, "aa", 1.0, 0.0),
            (2025, 3, "aa", 0.0, 0.0),
        ]
    )
    shares = injuries.snap_shares(snaps).set_index("order")

    # Entering week 3 he was a full-time player, despite playing no snaps in it.
    assert shares.loc[202503, "share_before"] == pytest.approx(1.0)


def test_snap_share_has_no_value_before_a_players_first_game():
    snaps = _snaps([(2025, 1, "aa", 1.0, 0.0)])
    assert injuries.snap_shares(snaps).empty


def test_a_starter_outweighs_a_rotational_player_at_the_same_position():
    """The whole point of snap share over position weights."""
    snaps = _snaps(
        [
            (2025, 1, "starter", 1.0, 0.0),
            (2025, 2, "starter", 1.0, 0.0),
            (2025, 1, "backup", 0.1, 0.0),
            (2025, 2, "backup", 0.1, 0.0),
        ]
    )
    shares = injuries.snap_shares(snaps)
    bridge = _bridge([("p0", "starter"), ("p1", "backup")])

    reports = _reports(
        [(2025, 2, "AAA", "WR", "Out"), (2025, 2, "BBB", "WR", "Out")]
    )
    burden = injuries.burden(reports, shares, bridge).set_index("team")

    assert burden.loc["AAA", "weighted_out"] > burden.loc["BBB", "weighted_out"]


def test_players_without_snap_history_fall_back_to_position():
    snaps = _snaps([(2025, 1, "other", 1.0, 0.0), (2025, 2, "other", 1.0, 0.0)])
    shares = injuries.snap_shares(snaps)

    reports = _reports([(2025, 2, "AAA", "QB", "Out")])
    burden = injuries.burden(reports, shares, _bridge([("zz", "zz")]))

    assert burden.iloc[0]["weighted_out"] == pytest.approx(
        injuries.POSITION_SHARE["QB"]
    )


def test_share_carries_over_from_the_most_recent_prior_week():
    """A player hurt in week 5 is weighted by his week 1-4 usage."""
    snaps = _snaps(
        [(2025, w, "aa", 1.0, 0.0) for w in (1, 2, 3, 4)]
    )
    shares = injuries.snap_shares(snaps)
    reports = _reports([(2025, 8, "AAA", "WR", "Out")])

    burden = injuries.burden(reports, shares, _bridge([("p0", "aa")]))
    assert burden.iloc[0]["weighted_out"] == pytest.approx(1.0)


def test_snap_share_falls_back_when_no_snap_data_exists():
    """Seasons before 2012 have no snap counts and must still work."""
    reports = _reports([(2010, 1, "AAA", "QB", "Out")])
    burden = injuries.burden(reports, injuries.snap_shares(_snaps([])), None)

    assert burden.iloc[0]["weighted_out"] == pytest.approx(
        injuries.POSITION_SHARE["QB"]
    )
