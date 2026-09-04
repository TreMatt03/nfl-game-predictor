"""Print the starting quarterback assumed for every team.

Worth eyeballing before trusting a slate, especially early in a season: the
inferred names come from last season's snaps and go stale the moment a
quarterback changes team. Anything wrong here should be corrected in
config/starters.json rather than in code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflpred import data, pipeline, quarterback

FORM_SEASONS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="season to resolve starters for")
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    schedules = data.load_schedules(refresh=args.refresh)
    last_played = int(schedules[schedules["result"].notna()]["season"].max())
    season = args.season or int(schedules[schedules["result"].isna()]["season"].min())

    _, _, _, starters = pipeline.build(start_season=last_played - FORM_SEASONS + 1)
    qbs = quarterback.current_starters(starters, season=season)

    confirmed = (qbs["source"] == "confirmed").sum()
    print(f"\nAssumed starters for {season} "
          f"({confirmed} confirmed, {len(qbs) - confirmed} inferred)\n")
    print(f"  {'TEAM':<5} {'QUARTERBACK':<16} {'RATING':>7}  {'STARTS':>6}  SOURCE")

    for team, row in qbs.sort_index().iterrows():
        flag = "confirmed" if row["source"] == "confirmed" else ""
        print(
            f"  {team:<5} {row['qb_name']:<16} {row['qb_form_now']:>+7.3f}  "
            f"{int(row['career_starts']):>6}  {flag}"
        )

    print("\nRating is shrunk EPA per dropback; 0.000 is replacement level.")
    print("Fix anything wrong in config/starters.json.\n")


if __name__ == "__main__":
    main()
