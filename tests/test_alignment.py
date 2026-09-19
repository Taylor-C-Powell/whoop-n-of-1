"""Tests for the alignment rules the article depends on.

Synthetic cycles pin each rule down; the last block checks the real export,
including the two rows where rule 1 (wake date) and rule 2 (waking day) differ.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze as az  # noqa: E402


def cycles(rows):
    cols = ["Cycle start time", "Cycle end time", "Sleep onset", "Wake onset", "recovery", "asleep_min", "strain"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[:4]:
        df[c] = pd.to_datetime(df[c])
    return df


def day(s):
    return pd.Timestamp(s)


# ---- rule 1: recovery and sleep belong to the wake-onset date
def test_recovery_is_keyed_to_the_morning_it_was_computed():
    cyc = cycles([
        ("2026-05-05 22:05", "2026-05-06 23:55", "2026-05-05 22:05", "2026-05-06 05:30", 91, 415, 5.8),   # asleep before midnight
        ("2026-05-09 00:37", "2026-05-09 22:07", "2026-05-09 00:37", "2026-05-09 06:57", 70, 348, 18.3),  # asleep after midnight
    ])
    by_wake = az.recovery_by_wake_date(cyc)
    assert list(by_wake.index) == [day("2026-05-06"), day("2026-05-09")]
    assert by_wake.loc["2026-05-06", "recovery"] == 91


def test_cycle_start_key_moves_only_the_nights_that_began_before_midnight():
    cyc = cycles([
        ("2026-05-05 22:05", "2026-05-06 23:55", "2026-05-05 22:05", "2026-05-06 05:30", 91, 415, 5.8),
        ("2026-05-09 00:37", "2026-05-09 22:07", "2026-05-09 00:37", "2026-05-09 06:57", 70, 348, 18.3),
    ])
    by_start = az.recovery_by_cycle_start(cyc)
    assert list(by_start.index) == [day("2026-05-05"), day("2026-05-09")]   # first night lands a day early
    assert list(by_start["unshifted"]) == [False, True]                      # second night is left in place


# ---- rule 2: strain belongs to the waking day the cycle covers
def test_strain_and_recovery_share_a_date_on_an_ordinary_night():
    cyc = cycles([("2026-05-05 22:05", "2026-05-06 23:55", "2026-05-05 22:05", "2026-05-06 05:30", 91, 415, 5.8)])
    assert az.strain_by_waking_day(cyc).index[0] == az.recovery_by_wake_date(cyc).index[0] == day("2026-05-06")


def test_strain_stays_with_its_own_day_when_the_night_at_cycle_start_was_not_recorded():
    # The cycle opens late on May 31, but the only sleep WHOOP recorded in it ended early on Jun 2.
    # Its recovery belongs to Jun 2; its strain was accumulated on Jun 1.
    cyc = cycles([("2026-05-31 23:24", "2026-06-02 06:27", "2026-06-01 23:28", "2026-06-02 02:57", 21, 200, 2.9)])
    assert az.recovery_by_wake_date(cyc).index[0] == day("2026-06-02")
    assert az.strain_by_waking_day(cyc).index[0] == day("2026-06-01")


def test_a_strap_off_placeholder_is_not_a_rest_day():
    cyc = cycles([
        ("2026-05-05 00:00", "2026-05-05 22:05", None, None, np.nan, np.nan, 0.0),
        ("2026-05-05 22:05", "2026-05-06 23:55", "2026-05-05 22:05", "2026-05-06 05:30", 91, 415, 5.8),
    ])
    assert list(az.strain_by_waking_day(cyc).index) == [day("2026-05-06")]


def test_two_cycles_on_one_waking_day_fail_loudly():
    cyc = cycles([
        ("2026-05-06 01:00", "2026-05-06 11:00", "2026-05-06 01:00", "2026-05-06 05:00", 50, 240, 3.0),
        ("2026-05-06 11:00", "2026-05-06 23:00", "2026-05-06 11:00", "2026-05-06 12:00", 40, 60, 9.0),
    ])
    with pytest.raises(ValueError):
        az.strain_by_waking_day(cyc)


# ---- rule 3: one scored sleep per wake date, fragmented mornings flagged
def test_longest_sleep_wins_and_the_morning_is_flagged_as_fragmented():
    cyc = cycles([
        ("2026-05-13 00:01", "2026-05-14 04:56", "2026-05-14 01:14", "2026-05-14 02:30", 6, 51, 4.1),
        ("2026-05-14 04:56", "2026-05-14 23:57", "2026-05-14 04:56", "2026-05-14 08:51", 53, 218, 16.1),
        ("2026-05-14 23:57", "2026-05-16 03:06", "2026-05-14 23:57", "2026-05-15 08:03", 67, 473, 10.9),
    ])
    by_wake = az.recovery_by_wake_date(cyc)
    assert by_wake.loc["2026-05-14", "recovery"] == 53
    assert bool(by_wake.loc["2026-05-14", "fragmented"]) and not bool(by_wake.loc["2026-05-15", "fragmented"])


# ---- rule 4: "prior day" is a calendar day
def test_prior_day_leaves_a_gap_empty_instead_of_borrowing_the_previous_row():
    strain = pd.Series([10.0, 12.0], index=pd.to_datetime(["2026-05-10", "2026-05-13"]))
    prev = az.prior_day(strain).reindex(pd.to_datetime(["2026-05-11", "2026-05-13", "2026-05-14"]))
    assert prev.loc["2026-05-11"] == 10.0
    assert np.isnan(prev.loc["2026-05-13"])          # May 12 was never logged
    assert prev.loc["2026-05-14"] == 12.0


# ---- rule 5: nothing measured after the score may predict it
def test_every_logged_behaviour_enters_the_models_with_a_lag():
    for feature in az.OLS_FEATURES + az.RF_FEATURES:
        assert feature not in az.LAGGED_BEHAVIOURS, f"{feature} is logged after the recovery score is computed"


# ---- the real export
@pytest.fixture(scope="module")
def real():
    cyc = az.load_cycles(az.DATA / "whoop" / "physiological_cycles.csv")
    log = az.load_daily_log(az.DATA / "daily_master.csv")
    return cyc, log


def test_real_export_counts(real):
    cyc, _ = real
    scored = az.scored(cyc)
    moved = scored["Cycle start time"].dt.normalize() != scored["Wake onset"].dt.normalize()
    assert (len(cyc), len(scored), int(moved.sum()), int((~moved).sum())) == (31, 29, 18, 11)
    by_wake = az.recovery_by_wake_date(cyc)
    assert len(by_wake) == 27
    assert [d.strftime("%m-%d") for d in by_wake.index[by_wake["fragmented"]]] == ["05-14", "06-02"]


def test_real_export_edge_rows_carry_the_right_strain(real):
    cyc, _ = real
    strain = az.strain_by_waking_day(cyc)
    assert strain.loc["2026-05-13"] == 4.1      # row whose wake date is May 14
    assert strain.loc["2026-06-01"] == 2.9      # row whose wake date is Jun 2
    assert strain.loc["2026-06-02"] == 13.5     # the day Jun 3's recovery should be paired with
    assert day("2026-05-05") not in strain.index


def test_a_session_without_an_rpe_is_missing_not_zero(real):
    _, log = real
    for d in ("2026-05-11", "2026-05-27", "2026-05-30"):
        assert np.isnan(log.loc[d, "srpe"])
    assert log.loc["2026-05-10", "srpe"] == 0       # a genuine rest day stays a zero
