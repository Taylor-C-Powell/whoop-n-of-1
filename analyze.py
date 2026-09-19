"""What a month of my WHOOP data actually predicted: an n-of-1 analysis.

Reproduces every number and figure in ARTICLE.md from the files in ./data.

    python analyze.py        # writes figures/*.png, results.md, results.json
    python -m pytest -q      # checks the alignment rules the article depends on

Alignment rules (each has a test in tests/test_alignment.py)
1. Recovery, HRV, RHR and the sleep fields in a WHOOP cycle row describe the
   sleep that ended at `Wake onset`, so they are keyed to the wake-onset date.
   Keying them to cycle start moves a night one day early whenever sleep began
   before midnight, and leaves it in place when sleep began after midnight.
2. `Day Strain` in the same row accrues from cycle start to cycle end, so it is
   keyed to the waking day the cycle covers (the date of the cycle midpoint).
   Rules 1 and 2 usually give the same date. They differ when the night at the
   start of a cycle was not recorded: that row's strain belongs to the day
   *before* its wake date.
3. One scored sleep per wake date: the longest. Mornings with more than one
   scored sleep are flagged as fragmented, and the lag analysis is reported
   with and without them.
4. "Prior day" is the previous calendar day, found by shifting dates, never by
   taking the previous row.
5. A predictor of the recovery score has to exist before the score does. The
   score is computed at wake, so logged behaviours enter with a one-day lag.

Everything here is exploratory: one subject, about four weeks, many tests,
autocorrelated days. p-values are reported for calibration, not for belief.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"

RENAME = {
    "Recovery score %": "recovery", "Resting heart rate (bpm)": "rhr",
    "Heart rate variability (ms)": "hrv", "Skin temp (celsius)": "skin_temp",
    "Blood oxygen %": "spo2", "Day Strain": "strain",
    "Sleep performance %": "sleep_perf", "Respiratory rate (rpm)": "resp_rate",
    "Asleep duration (min)": "asleep_min", "Deep (SWS) duration (min)": "deep_min",
    "REM duration (min)": "rem_min", "Sleep debt (min)": "sleep_debt",
    "Sleep efficiency %": "sleep_eff",
}
TIME_COLS = ["Cycle start time", "Cycle end time", "Sleep onset", "Wake onset"]
# WHOOP documents these as inputs to the recovery score, so their correlation
# with the score is partly built in.
SCORE_INPUTS = ["hrv", "rhr", "resp_rate", "sleep_perf", "skin_temp", "spo2"]
NIGHT_SLEEP = ["sleep_debt", "deep_min", "rem_min", "sleep_h", "sleep_eff"]
MORNING_SELF = ["zg_soreness", "zg_energy", "weight_kg"]
LAGGED_BEHAVIOURS = ["strain", "srpe", "caffeine_mg", "alcohol_g", "water_ml"]
OLS_FEATURES = ["sleep_h", "sleep_debt", "prev_strain", "prev_caffeine_mg", "prev_alcohol_g"]
RF_FEATURES = ["sleep_h", "deep_min", "rem_min", "sleep_debt", "prev_strain", "prev_srpe",
               "prev_caffeine_mg", "prev_alcohol_g", "prev_water_ml", "weight_kg", "zg_soreness"]
RF_SEEDS, RF_TREES = 25, 200
NICE = {
    "recovery": "Recovery %", "hrv": "HRV (ms)", "rhr": "Resting HR (bpm)",
    "resp_rate": "Respiratory rate", "sleep_perf": "Sleep performance %",
    "skin_temp": "Skin temp (°C)", "spo2": "Blood oxygen %", "sleep_debt": "Sleep debt (min)",
    "deep_min": "Deep sleep (min)", "rem_min": "REM (min)", "sleep_h": "Sleep hours",
    "sleep_eff": "Sleep efficiency %", "zg_soreness": "Soreness (self)",
    "zg_energy": "Energy (self)", "weight_kg": "Bodyweight (kg)", "strain": "Day strain",
    "srpe": "Training load (sRPE)", "caffeine_mg": "Caffeine (mg)", "alcohol_g": "Alcohol (g)",
    "water_ml": "Water (ml)", "prev_strain": "Strain, prior day",
    "prev_srpe": "Training load, prior day", "prev_caffeine_mg": "Caffeine, prior day",
    "prev_alcohol_g": "Alcohol, prior day", "prev_water_ml": "Water, prior day",
}


# ------------------------------------------------------------------ loading
def load_cycles(path: Path) -> pd.DataFrame:
    cyc = pd.read_csv(path)
    for col in TIME_COLS:
        cyc[col] = pd.to_datetime(cyc[col], errors="coerce")
    cyc = cyc.rename(columns=RENAME)
    cyc["sleep_h"] = cyc["asleep_min"] / 60.0
    return cyc


def load_daily_log(path: Path) -> pd.DataFrame:
    log = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    # A session with no RPE was stored as a training load of zero. That is a
    # missing value, not a rest day; a rest day has n_sessions == 0.
    log.loc[(log["n_sessions"] > 0) & (log["srpe"] == 0), "srpe"] = np.nan
    return log


# ---------------------------------------------------------------- alignment
def scored(cyc: pd.DataFrame) -> pd.DataFrame:
    return cyc.dropna(subset=["recovery"])


def recovery_by_wake_date(cyc: pd.DataFrame) -> pd.DataFrame:
    """Rules 1 and 3: one row per wake date (the longest scored sleep)."""
    s = scored(cyc).copy()
    s["date"] = s["Wake onset"].dt.normalize()
    s["fragmented"] = s.groupby("date")["date"].transform("size") > 1
    s = s.sort_values("asleep_min", ascending=False).drop_duplicates("date")
    return s.set_index("date").sort_index()


def recovery_by_cycle_start(cyc: pd.DataFrame) -> pd.DataFrame:
    """The original, wrong keying. Kept only to show what it did (Section A)."""
    s = scored(cyc).copy()
    s["date"] = s["Cycle start time"].dt.normalize()
    s["unshifted"] = s["date"] == s["Wake onset"].dt.normalize()
    s = s.sort_values("asleep_min", ascending=False).drop_duplicates("date")
    return s.set_index("date").sort_index()


def strain_by_waking_day(cyc: pd.DataFrame) -> pd.Series:
    """Rule 2: strain keyed to the date of the cycle midpoint.

    Cycles with no recorded sleep and zero strain are strap-off placeholders
    (the activation day), not rest days, and are dropped.
    """
    c = cyc.dropna(subset=["strain", "Cycle end time"]).copy()
    placeholder = c["asleep_min"].isna() & (c["strain"] == 0)
    c = c[~placeholder]
    mid = c["Cycle start time"] + (c["Cycle end time"] - c["Cycle start time"]) / 2
    out = pd.Series(c["strain"].to_numpy(), index=pd.DatetimeIndex(mid.dt.normalize()), name="strain")
    if out.index.has_duplicates:
        raise ValueError("two cycles map to one waking day; strain needs an explicit rule")
    return out.sort_index()


def prior_day(values: pd.Series) -> pd.Series:
    """Rule 4: the value logged on the previous calendar day."""
    out = values.copy()
    out.index = out.index + pd.Timedelta(days=1)
    return out


def build_frame(recovery_rows: pd.DataFrame, strain: pd.Series, log: pd.DataFrame) -> pd.DataFrame:
    device = recovery_rows[[c for c in RENAME.values() if c != "strain"] + ["sleep_h"]]
    extra = [c for c in ("fragmented", "unshifted") if c in recovery_rows]
    frame = pd.concat([device, recovery_rows[extra], strain, log], axis=1, sort=True)
    for col in LAGGED_BEHAVIOURS:
        frame[f"prev_{col}"] = prior_day(frame[col]).reindex(frame.index)
    return frame


# -------------------------------------------------------------------- stats
def corr(df: pd.DataFrame, a: str, b: str, method: str = "spearman") -> dict:
    s = df[[a, b]].dropna()
    if len(s) < 6:
        return {"r": None, "p": None, "n": int(len(s))}
    f = stats.pearsonr if method == "pearson" else stats.spearmanr
    r, p = f(s[a], s[b])
    return {"r": float(r), "p": float(p), "n": int(len(s))}


def boot_ci(df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """Percentile bootstrap for Spearman's rho. Days are resampled as if
    independent, which they are not, so read the interval as a lower bound on
    the uncertainty."""
    s = df[[a, b]].dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        t = s[rng.integers(0, len(s), len(s))]
        if len(np.unique(t[:, 0])) > 2 and len(np.unique(t[:, 1])) > 2:
            rhos.append(np.corrcoef(stats.rankdata(t[:, 0]), stats.rankdata(t[:, 1]))[0, 1])
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return [float(lo), float(hi)]


def ols(df: pd.DataFrame, y: str, features: list[str], standardize: bool = True) -> dict:
    """Ordinary least squares with 95% intervals; predictors in SD units."""
    sub = df[[y] + features].dropna().astype(float)
    X = sub[features]
    if standardize:
        X = (X - X.mean()) / X.std()
    A = np.column_stack([np.ones(len(sub)), X.to_numpy()])
    yv = sub[y].to_numpy()
    beta = np.linalg.solve(A.T @ A, A.T @ yv)
    resid = yv - A @ beta
    dof = len(sub) - A.shape[1]
    se = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(A.T @ A)))
    half = stats.t.ppf(0.975, dof) * se
    pvals = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - resid @ resid / ((yv - yv.mean()) @ (yv - yv.mean()))
    return {"n": int(len(sub)), "r2": float(r2),
            "adj_r2": float(1 - (1 - r2) * (len(sub) - 1) / dof),
            "beta": {f: {"b": float(beta[i + 1]), "lo": float(beta[i + 1] - half[i + 1]),
                         "hi": float(beta[i + 1] + half[i + 1]), "p": float(pvals[i + 1])}
                     for i, f in enumerate(features)}}


def analyse(M0: pd.DataFrame, M: pd.DataFrame, cyc: pd.DataFrame) -> dict:
    R: dict = {}
    sc = scored(cyc)
    shifted = sc["Cycle start time"].dt.normalize() != sc["Wake onset"].dt.normalize()
    fragmented = M["fragmented"].fillna(False).astype(bool)
    both = M[["zg_hrv", "hrv"]].dropna()
    R["counts"] = {
        "cycles": int(len(cyc)), "scored_cycles": int(len(sc)),
        "wake_dates": int(M["recovery"].notna().sum()),
        "fragmented_mornings": [d.strftime("%Y-%m-%d") for d in M.index[fragmented]],
        "moved_earlier_by_cycle_start_key": int(shifted.sum()),
        "left_in_place_by_cycle_start_key": int((~shifted).sum()),
        "overlap_wake": int((M["recovery"].notna() & M["zg_readiness"].notna()).sum()),
        "overlap_start": int((M0["recovery"].notna() & M0["zg_readiness"].notna()).sum()),
        "alcohol_days_in_log": int((M["alcohol_g"] > 0).sum()),
        "alcohol_days_in_whoop_window": int(((M["alcohol_g"] > 0) & M["recovery"].notna()).sum()),
        "self_hrv_identical_days": int((both["zg_hrv"] == both["hrv"]).sum()),
        "self_hrv_compared_days": int(len(both)),
        "self_hrv_max_abs_diff_ms": float((both["zg_hrv"] - both["hrv"]).abs().max()),
    }
    # A) alignment: self-logged vs device under both keyings
    R["alignment"] = {
        label: {"hrv_self_vs_device": corr(df, "zg_hrv", "hrv", "pearson"),
                "sleep_self_vs_device": corr(df, "zg_sleep_h", "sleep_h", "pearson"),
                "readiness_vs_recovery": corr(df, "zg_readiness", "recovery", "pearson")}
        for label, df in (("cycle_start", M0), ("wake_onset", M))}
    # B) same-morning correlates of the score, grouped by what they are
    R["same_morning"] = {
        "score_inputs": {c: corr(M, c, "recovery") for c in SCORE_INPUTS},
        "night_sleep": {c: corr(M, c, "recovery") for c in NIGHT_SLEEP},
        "morning_self_report": {c: corr(M, c, "recovery") for c in MORNING_SELF},
    }
    R["inputs_explain_score"] = ols(M, "recovery", ["hrv", "rhr", "resp_rate", "sleep_perf"], standardize=False)
    R["skin_vs_hrv"] = corr(M, "skin_temp", "hrv", "pearson")
    # C) prior-day strain and training load, against the raw signals and the score
    steady = M[~fragmented]
    lag = {}
    for target in ("hrv", "rhr", "recovery"):
        lag[target] = {
            "prior_day_strain": {**corr(M, "prev_strain", target), "ci": boot_ci(M, "prev_strain", target)},
            "prior_day_strain_no_fragmented": {**corr(steady, "prev_strain", target),
                                               "ci": boot_ci(steady, "prev_strain", target)},
            "same_day_strain": corr(M, "strain", target),
            "prior_day_srpe": corr(M, "prev_srpe", target),
        }
    lag["recovery"]["prior_day_caffeine"] = corr(M, "prev_caffeine_mg", "recovery")
    lag["recovery"]["prior_day_alcohol"] = corr(M, "prev_alcohol_g", "recovery")
    R["lagged"] = lag
    # D) regression on predictors that precede the score
    R["ols"] = ols(M, "recovery", OLS_FEATURES)
    # E) random forest on the same principle, repeated over seeds
    R["rf"] = forest_importance(M)
    return R


def forest_importance(M: pd.DataFrame, seeds: int = RF_SEEDS) -> dict:
    """Impurity importance averaged over seeds, plus how often each feature
    ranks first. With about 20 rows a single seed's ordering means little."""
    sub = M[["recovery"] + RF_FEATURES].dropna()
    imps = np.zeros((seeds, len(RF_FEATURES)))
    for seed in range(seeds):
        rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed, min_samples_leaf=2)
        imps[seed] = rf.fit(sub[RF_FEATURES], sub["recovery"]).feature_importances_
    first = np.bincount(imps.argmax(axis=1), minlength=len(RF_FEATURES)) / seeds
    order = np.argsort(-imps.mean(axis=0))
    return {"n": int(len(sub)), "seeds": seeds, "trees": RF_TREES,
            "importance": {RF_FEATURES[i]: float(imps[:, i].mean()) for i in order},
            "ranked_first_share": {RF_FEATURES[i]: float(first[i]) for i in order if first[i] > 0}}


# ------------------------------------------------------------------ figures
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fmt(c: dict, sym: str = "ρ") -> str:
    if c["r"] is None:
        return f"n = {c['n']} (too few)"
    return f"{sym} = {c['r']:+.2f}, {fmt_p(c['p'])}, n = {c['n']}"


def scatter(ax, x, y, color=BLUE, hollow=False, label=None):
    face = SURFACE if hollow else color
    edge = color if hollow else SURFACE
    ax.scatter(x, y, s=26 if hollow else 42, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3, label=label)


def fitline(ax, x, y, color=BLUE, style="-"):
    s = pd.concat([x, y], axis=1).dropna().astype(float)
    if len(s) >= 3:
        m, b = np.polyfit(s.iloc[:, 0], s.iloc[:, 1], 1)
        xs = np.linspace(s.iloc[:, 0].min(), s.iloc[:, 0].max(), 50)
        ax.plot(xs, m * xs + b, color=color, linewidth=2, linestyle=style, zorder=2)


def finish(fig, ax_list, title, name, rect=(0, 0, 1, 1)):
    for ax in ax_list:
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG / name)
    plt.close(fig)


def two_panel(M, cols, labels, y, ylabel, title, name, xlabel=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, sharey=True)
    for ax, col, lab in zip(axes, cols, labels):
        s = M[[col, y]].dropna()
        scatter(ax, s[col], s[y])
        fitline(ax, s[col], s[y])
        ax.set_title(f"{lab}\n{fmt(corr(M, col, y))}")
        ax.set_xlabel(xlabel or lab)
    axes[0].set_ylabel(ylabel)
    finish(fig, axes, title, name)


def make_figures(M0: pd.DataFrame, M: pd.DataFrame, R: dict) -> None:
    FIG.mkdir(exist_ok=True)
    set_style()

    # Fig 1: random-forest importance
    top = list(R["rf"]["importance"].items())[:8][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    best = top[-1][1]
    colors = [BLUE if best - v < 0.01 else "#b9cfee" for _, v in top]  # a tie is drawn as a tie
    bars = ax.barh([NICE[k] for k, _ in top], [v for _, v in top], color=colors, height=0.62)
    for b, (_, v) in zip(bars, top):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=MUTED, fontsize=9)
    ax.set_xlim(0, top[-1][1] * 1.18)
    ax.set_xlabel(f"Random-forest impurity importance (sums to 1 across all {len(RF_FEATURES)} features)")
    ax.grid(axis="y", visible=False)
    finish(fig, [ax], f"What came before my recovery score  ·  {R['rf']['n']} days, mean of {R['rf']['seeds']} forests",
           "fig1_importance.png")

    # Fig 2: sleep debt and sleep performance vs recovery
    two_panel(M, ["sleep_debt", "sleep_perf"], ["Sleep debt (min)", "Sleep performance % (a score input)"],
              "recovery", "WHOOP recovery %", "Sleep debt and sleep performance vs. recovery",
              "fig2_sleep_vs_recovery.png")

    # Fig 3: same-day vs prior-day strain, against the raw signal
    two_panel(M, ["strain", "prev_strain"], ["Same-day strain", "Prior-day strain"], "hrv",
              "Morning HRV (ms)", "Training fatigue shows up a day late, in the raw signal",
              "fig3_strain_lag.png", xlabel="WHOOP day strain")

    # Fig 4: skin temperature vs HRV
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    s = M[["skin_temp", "hrv"]].dropna()
    scatter(ax, s["skin_temp"], s["hrv"])
    fitline(ax, s["skin_temp"], s["hrv"])
    ax.set_title(fmt(R["skin_vs_hrv"], "r"))
    ax.set_xlabel("Skin temperature during sleep (°C)")
    ax.set_ylabel("HRV (ms)")
    finish(fig, [ax], "Skin temperature vs. HRV", "fig4_skintemp_hrv.png")

    # Fig 5: the alignment bug
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), dpi=200, sharey=True)
    s = M0[["zg_hrv", "hrv", "unshifted"]].dropna()
    stayed_mask = s["unshifted"].astype(bool)
    moved, stayed = s[~stayed_mask], s[stayed_mask]
    scatter(axes[0], moved["hrv"], moved["zg_hrv"], ORANGE, label="key landed on an earlier date")
    scatter(axes[0], stayed["hrv"], stayed["zg_hrv"], ORANGE, hollow=True, label="cycle began after midnight: key landed on the right date")
    fitline(axes[0], s["hrv"], s["zg_hrv"], ORANGE)
    axes[0].set_title(f"Keyed to cycle start (wrong)\n{fmt(R['alignment']['cycle_start']['hrv_self_vs_device'], 'r')}")
    s = M[["zg_hrv", "hrv"]].dropna()
    scatter(axes[1], s["hrv"], s["zg_hrv"])
    fitline(axes[1], s["hrv"], s["zg_hrv"])
    axes[1].set_title(f"Keyed to wake onset (right)\n{fmt(R['alignment']['wake_onset']['hrv_self_vs_device'], 'r')}")
    for ax in axes:
        ax.set_xlabel("WHOOP HRV (ms)")
    axes[0].set_ylabel("HRV I logged in the app (ms)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=2,
               handletextpad=0.2, columnspacing=1.2)
    finish(fig, axes, "One join key, two conclusions", "fig5_alignment_bug.png", rect=(0, 0.07, 1, 1))

    # Fig 6: the same lag against the composite score, fragmented mornings marked
    fig, ax = plt.subplots(figsize=(5.6, 3.85), dpi=200)
    s = M[["prev_strain", "recovery", "fragmented"]].dropna()
    frag = s["fragmented"].astype(bool)
    scatter(ax, s.loc[~frag, "prev_strain"], s.loc[~frag, "recovery"], label="ordinary mornings")
    scatter(ax, s.loc[frag, "prev_strain"], s.loc[frag, "recovery"], ORANGE, label="after a fragmented night")
    fitline(ax, s.loc[~frag, "prev_strain"], s.loc[~frag, "recovery"])
    fitline(ax, s["prev_strain"], s["recovery"], ORANGE, style="--")
    lag = R["lagged"]["recovery"]
    ax.set_title(f"without them: {fmt(lag['prior_day_strain_no_fragmented'])}\n"
                 f"all mornings:  {fmt(lag['prior_day_strain'])}", fontsize=9.5)
    ax.set_xlabel("Prior-day WHOOP strain")
    ax.set_ylabel("WHOOP recovery %")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.1, 0.0), ncol=2,
               handletextpad=0.2, columnspacing=1.2)
    finish(fig, [ax], "Two mornings decide how strong the score version looks", "fig6_score_vs_signal.png",
           rect=(0, 0.07, 1, 1))


# ------------------------------------------------------------------ results
def table(rows: list[tuple[str, dict]], sym: str = "ρ") -> list[str]:
    out = [f"| Variable | {sym} | p | n |", "|---|---|---|---|"]
    for name, c in rows:
        if c["r"] is None:
            out.append(f"| {name} | n/a | n/a | {c['n']} |")
        else:
            p = "< 0.001" if c["p"] < 0.001 else f"{c['p']:.3f}"
            out.append(f"| {name} | {c['r']:+.2f} | {p} | {c['n']} |")
    return out


def by_strength(block: dict) -> list[tuple[str, dict]]:
    return sorted(((NICE[k], v) for k, v in block.items()), key=lambda kv: -abs(kv[1]["r"] or 0))


def results_markdown(R: dict) -> str:
    n, A, lag = R["counts"], R["alignment"], R["lagged"]
    L = ["# Results (regenerated by analyze.py)", "",
         f"Cycles in the export: {n['cycles']} · with a recovery score: {n['scored_cycles']} · "
         f"distinct wake dates: {n['wake_dates']} · overlap with the daily log: {n['overlap_wake']} "
         f"(cycle-start keyed: {n['overlap_start']})", "",
         f"Fragmented mornings (more than one scored sleep): {', '.join(n['fragmented_mornings'])}", "",
         "## A. Alignment check: self-logged vs device", "",
         f"The cycle-start key put {n['moved_earlier_by_cycle_start_key']} of {n['scored_cycles']} scored cycles on an "
         f"earlier date than the morning they describe and left {n['left_in_place_by_cycle_start_key']} in place "
         "(the cycle began after midnight).", "",
         "| Pair | Cycle-start keyed | Wake-onset keyed |", "|---|---|---|"]
    for key, name in (("hrv_self_vs_device", "HRV, self-logged vs WHOOP"),
                      ("sleep_self_vs_device", "Sleep hours, self vs WHOOP"),
                      ("readiness_vs_recovery", "Readiness (self) vs WHOOP recovery")):
        L.append(f"| {name} | {fmt(A['cycle_start'][key], 'r')} | {fmt(A['wake_onset'][key], 'r')} |")
    L += ["", f"Self-logged HRV equals the device value on {n['self_hrv_identical_days']} of "
              f"{n['self_hrv_compared_days']} days; the largest difference is {n['self_hrv_max_abs_diff_ms']:.0f} ms.", "",
          "## B. Same-morning correlates of WHOOP recovery (Spearman, wake-onset keyed)", "",
          "### B1. Documented inputs to the score (correlation is partly built in)", ""]
    L += table(by_strength(R["same_morning"]["score_inputs"]))
    ie = R["inputs_explain_score"]
    L += ["", f"Recovery regressed on HRV, RHR, respiratory rate and sleep performance: R² = {ie['r2']:.2f} (n = {ie['n']}).", "",
          "### B2. Other measures of the night's sleep", ""]
    L += table(by_strength(R["same_morning"]["night_sleep"]))
    L += ["", "### B3. Morning self-reports", ""]
    L += table(by_strength(R["same_morning"]["morning_self_report"]))
    L += ["", f"Skin temp vs HRV (Pearson): {fmt(R['skin_vs_hrv'], 'r')}", "",
          "## C. Prior-day strain and training load (Spearman; strain keyed to the waking day)", "",
          "| Outcome | Prior-day strain, all mornings | 95% bootstrap CI | Without fragmented mornings | 95% bootstrap CI | Same-day strain | Prior-day sRPE |",
          "|---|---|---|---|---|---|---|"]
    for target in ("hrv", "rhr", "recovery"):
        t = lag[target]
        L.append(f"| {NICE[target]} | {fmt(t['prior_day_strain'])} | [{t['prior_day_strain']['ci'][0]:+.2f}, {t['prior_day_strain']['ci'][1]:+.2f}] "
                 f"| {fmt(t['prior_day_strain_no_fragmented'])} | [{t['prior_day_strain_no_fragmented']['ci'][0]:+.2f}, "
                 f"{t['prior_day_strain_no_fragmented']['ci'][1]:+.2f}] | {fmt(t['same_day_strain'])} | {fmt(t['prior_day_srpe'])} |")
    L += ["", f"Prior-day caffeine vs recovery: {fmt(lag['recovery']['prior_day_caffeine'])}",
          f"Prior-day alcohol vs recovery: {fmt(lag['recovery']['prior_day_alcohol'])} "
          f"({n['alcohol_days_in_whoop_window']} alcohol days inside the WHOOP window)", "",
          f"## D. Standardized OLS, recovery ~ predictors that precede the score "
          f"(n = {R['ols']['n']}, R² = {R['ols']['r2']:.2f}, adjusted R² = {R['ols']['adj_r2']:.2f})", "",
          "| Feature | β (recovery pts per SD) | 95% CI | p |", "|---|---|---|---|"]
    for k, b in sorted(R["ols"]["beta"].items(), key=lambda kv: -abs(kv[1]["b"])):
        L.append(f"| {NICE[k]} | {b['b']:+.1f} | [{b['lo']:+.1f}, {b['hi']:+.1f}] | {b['p']:.3f} |")
    rf = R["rf"]
    L += ["", f"## E. Random-forest importance (n = {rf['n']}; mean of {rf['seeds']} seeds, {rf['trees']} trees each; "
              "nothing measured after the score)", "",
          "| Feature | mean importance | ranked first in |", "|---|---|---|"]
    L += [f"| {NICE[k]} | {v:.3f} | {rf['ranked_first_share'].get(k, 0):.0%} of seeds |" for k, v in rf["importance"].items()]
    return "\n".join(L) + "\n"


def rounded(obj, places: int = 6):
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, dict):
        return {k: rounded(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, places) for v in obj]
    return obj


def main() -> None:
    cyc = load_cycles(DATA / "whoop" / "physiological_cycles.csv")
    log = load_daily_log(DATA / "daily_master.csv")
    strain = strain_by_waking_day(cyc)
    M0 = build_frame(recovery_by_cycle_start(cyc), strain, log)
    M = build_frame(recovery_by_wake_date(cyc), strain, log)
    R = analyse(M0, M, cyc)
    make_figures(M0, M, R)
    text = results_markdown(R)
    # newline="\n" keeps the outputs byte-identical across operating systems
    (HERE / "results.md").write_text(text, encoding="utf-8", newline="\n")
    (HERE / "results.json").write_text(json.dumps(rounded(R), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
