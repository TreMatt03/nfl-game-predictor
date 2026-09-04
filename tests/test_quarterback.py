"""Tests for quarterback rating and, especially, starter selection.

The starter-selection test encodes a bug that was live in this project: taking
each team's most recent start picked the Week 18 backup that rested teams play,
and projected those teams with a third-stringer all of the following season.
"""

import pandas as pd
import pytest

from nflpred import quarterback


def _starts(rows):
    frame = pd.DataFrame(rows)
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    if "game_id" not in frame:
        frame["game_id"] = [f"g{i}" for i in range(len(frame))]
    return frame


def test_shrinkage_pulls_thin_records_toward_replacement():
    """One dreadful afternoon should not define a career."""
    form = pd.Series([-0.5, -0.5])
    starts = pd.Series([1, 100])

    shrunk = quarterback._shrink(form, starts)

    assert abs(shrunk.iloc[0] - quarterback.REPLACEMENT_LEVEL) < abs(
        shrunk.iloc[1] - quarterback.REPLACEMENT_LEVEL
    )


def test_shrinkage_barely_moves_an_established_starter():
    shrunk = quarterback._shrink(pd.Series([0.2]), pd.Series([200]))
    assert shrunk.iloc[0] == pytest.approx(0.2, abs=0.01)


def test_unknown_passer_gets_replacement_level():
    shrunk = quarterback._shrink(pd.Series([float("nan")]), pd.Series([0]))
    assert shrunk.iloc[0] == pytest.approx(quarterback.REPLACEMENT_LEVEL)


def test_qb_form_does_not_include_the_current_game():
    starts = _starts(
        [
            {
                "passer_player_id": "qb1",
                "team": "AAA",
                "qb_name": "A.One",
                "gameday": f"2020-09-{d:02d}",
                "qb_epa": epa,
            }
            for d, epa in zip((6, 13, 20, 27), (0.1, 0.1, 0.1, 5.0))
        ]
    )
    form = quarterback.add_qb_form(starts)["qb_form"]

    # The blowout in the final game must not raise that game's own rating.
    assert form.iloc[-1] < 1.0


def test_rested_week_18_backup_does_not_become_the_starter():
    """A team resting its starter in the finale still starts him next season."""
    rows = [
        {
            "passer_player_id": "starter",
            "team": "AAA",
            "qb_name": "A.Starter",
            "gameday": f"2020-09-{d:02d}",
            "qb_epa": 0.2,
        }
        for d in (6, 13, 20, 27)
    ]
    rows += [
        {
            "passer_player_id": "starter",
            "team": "AAA",
            "qb_name": "A.Starter",
            "gameday": f"2020-10-{d:02d}",
            "qb_epa": 0.2,
        }
        for d in (4, 11, 18, 25)
    ]
    # One relief appearance, most recent of all.
    rows.append(
        {
            "passer_player_id": "backup",
            "team": "AAA",
            "qb_name": "B.Backup",
            "gameday": "2020-11-01",
            "qb_epa": -0.6,
        }
    )

    starters = quarterback.add_qb_form(_starts(rows))
    chosen = quarterback.current_starters(starters)

    assert chosen.loc["AAA", "qb_name"] == "A.Starter"


def test_genuine_midseason_change_is_picked_up():
    """A quarterback who has taken over for good should be the starter."""
    rows = [
        {
            "passer_player_id": "old",
            "team": "AAA",
            "qb_name": "O.Ld",
            "gameday": f"2020-09-{d:02d}",
            "qb_epa": 0.0,
        }
        for d in (6, 13, 20)
    ]
    rows += [
        {
            "passer_player_id": "new",
            "team": "AAA",
            "qb_name": "N.Ew",
            "gameday": f"2020-1{m}-{d:02d}",
            "qb_epa": 0.2,
        }
        for m, d in ((0, 4), (0, 11), (0, 18), (0, 25), (1, 1), (1, 8), (1, 15))
    ]

    starters = quarterback.add_qb_form(_starts(rows))
    chosen = quarterback.current_starters(starters)

    assert chosen.loc["AAA", "qb_name"] == "N.Ew"


def test_relief_appearances_are_not_counted_as_starts():
    pbp = pd.DataFrame(
        {
            "game_id": ["g1"] * 12,
            "season": [2020] * 12,
            "week": [1] * 12,
            "posteam": ["AAA"] * 12,
            "qb_dropback": [1.0] * 12,
            "qb_epa": [0.1] * 12,
            "passer_player_id": ["starter"] * 10 + ["backup"] * 2,
            "passer_player_name": ["A.Starter"] * 10 + ["B.Backup"] * 2,
        }
    )
    stats = quarterback.qb_game_stats(pbp)

    assert list(stats["passer_player_id"]) == ["starter"]
