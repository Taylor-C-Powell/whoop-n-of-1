"""What a month of my WHOOP data actually predicted — n-of-1 analysis.

Reproduces every number and figure in the article from the files in ./data.
Run:  python analyze.py      (writes figures/*.png and results.md)

Design notes
- WHOOP cycles are keyed to the *wake-onset* date (the morning a recovery score
  applies to). Keying to cycle-start time silently shifts every record one day
  earlier relative to same-day self-logs; Section A shows how much that matters.
- Everything here is exploratory: one subject, 21–31 usable days, many tests,
  autocorrelated days. p-values are reported for calibration, not for belief.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)

# ---- palette (light surface; 3 categorical slots validated all-pairs) ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def D(s):
    return pd.to_datetime(s, errors="coerce").dt.date


# ---------------------------------------------------------------- load
master = pd.read_csv(DATA / "daily_master.csv", parse_dates=["date"])
cyc = pd.read_csv(DATA / "whoop" / "physiological_cycles.csv")

ren = {
    "Recovery score %": "whoop_recovery", "Resting heart rate (bpm)": "whoop_rhr",
    "Heart rate variability (ms)": "whoop_hrv", "Skin temp (celsius)": "whoop_skintemp",
    "Day Strain": "whoop_strain", "Sleep performance %": "whoop_sleepperf",
    "Respiratory rate (rpm)": "whoop_resp", "Asleep duration (min)": "whoop_asleep_min",
    "Deep (SWS) duration (min)": "whoop_deep_min", "REM duration (min)": "whoop_rem_min",
    "Sleep debt (min)": "whoop_sleepdebt", "Sleep efficiency %": "whoop_sleepeff",
}
cyc = cyc.rename(columns=ren)
wcols = list(ren.values())

# Two keyings of the same WHOOP rows.
by_start = cyc.assign(date=pd.to_datetime(D(cyc["Cycle start time"])))[["date"] + wcols]
by_wake = cyc.assign(date=pd.to_datetime(D(cyc["Wake onset"])))[["date"] + wcols]
# Where two cycles share a date (a nap or a fragmented night creates a second short
# "sleep" with its own recovery score), keep the main sleep: the longest one.
def one_per_day(f):
    f = f.dropna(subset=["whoop_recovery"]).sort_values("whoop_asleep_min", ascending=False)
    return f.drop_duplicates("date").sort_values("date").reset_index(drop=True)

by_start, by_wake = one_per_day(by_start), one_per_day(by_wake)
for f in (by_start, by_wake):
    f["whoop_sleep_h"] = f["whoop_asleep_min"] / 60.0

# Self-logged + behavioural columns (keyed by the morning they were logged).
selfcols = ["date", "zg_hrv", "zg_sleep_h", "zg_sleepeff", "zg_energy", "zg_soreness",
            "zg_readiness", "caffeine_mg", "alcohol_g", "water_ml", "srpe", "rpe", "train_min",
            "weight_kg"]
selflog = master[[c for c in selfcols if c in master.columns]]

M0 = selflog.merge(by_start, on="date", how="outer").sort_values("date").reset_index(drop=True)  # original keying
M = selflog.merge(by_wake, on="date", how="outer").sort_values("date").reset_index(drop=True)    # corrected keying
for f in (M0, M):
    for c in ["srpe", "train_min"]:
        f[c] = f[c].fillna(0)
    # "prior day" means the calendar day before, not the previous row.
    prev = f[["date", "alcohol_g", "caffeine_mg", "srpe", "whoop_strain", "rpe"]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    prev = prev.rename(columns={c: f"prev_{c}" for c in prev.columns if c != "date"})
    f[prev.columns.drop("date")] = f[["date"]].merge(prev, on="date", how="left").drop(columns="date").values


def corr(df, a, b, method="pearson"):
    s = df[[a, b]].dropna()
    if len(s) < 6:
        return np.nan, np.nan, len(s)
    f = stats.pearsonr if method == "pearson" else stats.spearmanr
    r, p = f(s[a], s[b])
    return float(r), float(p), int(len(s))


R = {"n_whoop_days": int(M.whoop_recovery.notna().sum()),
     "n_overlap_wake": int((M.whoop_recovery.notna() & M.zg_readiness.notna()).sum()),
     "n_overlap_start": int((M0.whoop_recovery.notna() & M0.zg_readiness.notna()).sum())}

# ---------------------------------------------------------------- A) alignment
A = {}
for label, df in (("cycle_start", M0), ("wake_onset", M)):
    A[label] = {
        "hrv_self_vs_device": corr(df, "zg_hrv", "whoop_hrv"),
        "readiness_vs_recovery": corr(df, "zg_readiness", "whoop_recovery"),
        "sleep_self_vs_device": corr(df, "zg_sleep_h", "whoop_sleep_h"),
    }
R["alignment"] = A

# ---------------------------------------------------------------- B) same-day drivers of recovery
drivers = ["whoop_deep_min", "whoop_sleepperf", "whoop_sleepdebt", "whoop_sleep_h", "whoop_rem_min",
           "whoop_strain", "whoop_skintemp", "whoop_resp", "caffeine_mg", "alcohol_g", "water_ml",
           "srpe", "weight_kg", "zg_energy", "zg_soreness"]
B = {d: corr(M, d, "whoop_recovery", "spearman") for d in drivers if d in M}
R["same_day_spearman"] = B
R["skin_vs_hrv"] = corr(M, "whoop_skintemp", "whoop_hrv")
R["weight_vs_recovery"] = corr(M, "weight_kg", "whoop_recovery")

# ---------------------------------------------------------------- C) lagged
C = {c: corr(M, f"prev_{c}", "whoop_recovery", "spearman") for c in ["whoop_strain", "srpe", "alcohol_g", "caffeine_mg"]}
C["same_day_strain"] = corr(M, "whoop_strain", "whoop_recovery", "spearman")
R["lagged_spearman"] = C

# ---------------------------------------------------------------- D) OLS (standardized)
feat = ["whoop_sleep_h", "prev_alcohol_g", "prev_whoop_strain", "caffeine_mg", "whoop_sleepdebt"]
sub = M[["whoop_recovery"] + feat].dropna()
X = (sub[feat] - sub[feat].mean()) / sub[feat].std()
lr = LinearRegression().fit(X, sub["whoop_recovery"])
R["ols"] = {"n": int(len(sub)), "r2": float(lr.score(X, sub["whoop_recovery"])),
            "beta_per_sd": {k: float(v) for k, v in zip(feat, lr.coef_)}}

# ---------------------------------------------------------------- E) random-forest importance
rf_feat = ["whoop_sleep_h", "whoop_deep_min", "whoop_rem_min", "whoop_sleepdebt", "whoop_resp",
           "whoop_skintemp", "caffeine_mg", "alcohol_g", "water_ml", "srpe", "prev_alcohol_g",
           "prev_whoop_strain", "prev_srpe", "weight_kg", "zg_soreness"]
sub = M[["whoop_recovery"] + rf_feat].dropna()
rf = RandomForestRegressor(n_estimators=400, random_state=0, min_samples_leaf=2).fit(sub[rf_feat], sub["whoop_recovery"])
imp = sorted(zip(rf_feat, rf.feature_importances_), key=lambda x: -x[1])
R["rf"] = {"n": int(len(sub)), "importance": {k: float(v) for k, v in imp}}

# ---------------------------------------------------------------- figures
NICE = {"whoop_deep_min": "Deep sleep (min)", "whoop_sleepperf": "Sleep performance %",
        "whoop_sleepdebt": "Sleep debt (min)", "whoop_sleep_h": "Sleep hours", "whoop_rem_min": "REM (min)",
        "whoop_strain": "Day strain", "whoop_skintemp": "Skin temp (°C)", "whoop_resp": "Respiratory rate",
        "caffeine_mg": "Caffeine (mg)", "alcohol_g": "Alcohol (g)", "water_ml": "Water (ml)",
        "srpe": "Training load (sRPE)", "weight_kg": "Bodyweight (kg)", "zg_energy": "Energy (self)",
        "zg_soreness": "Soreness (self)", "prev_alcohol_g": "Alcohol, prior day",
        "prev_whoop_strain": "Strain, prior day", "prev_srpe": "Training load, prior day"}


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def scatter(ax, x, y, color=BLUE):
    ax.scatter(x, y, s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)


def fitline(ax, x, y, color=BLUE):
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) >= 3:
        m, b = np.polyfit(s.iloc[:, 0], s.iloc[:, 1], 1)
        xs = np.linspace(s.iloc[:, 0].min(), s.iloc[:, 0].max(), 50)
        ax.plot(xs, m * xs + b, color=color, linewidth=2, zorder=2)


# Fig 1 — importance
top = imp[:8]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
names = [NICE.get(k, k) for k, _ in top][::-1]
vals = [v for _, v in top][::-1]
bars = ax.barh(names, vals, color=[BLUE if i == len(vals) - 1 else "#b9cfee" for i in range(len(vals))], height=0.62)
for b, v in zip(bars, vals):
    ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=MUTED, fontsize=9)
ax.set_xlim(0, max(vals) * 1.18)
ax.set_xlabel("Random-forest impurity importance (sums to 1 across all 15 features)")
ax.set_title(f"What predicted my recovery score  ·  {R['rf']['n']} days, 400 trees")
ax.grid(axis="y", visible=False)
style_axes(ax)
fig.tight_layout(); fig.savefig(FIG / "fig1_importance.png"); plt.close(fig)

# Fig 2 — sleep architecture vs recovery (two panels, one series)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, sharey=True)
for ax, col, lab in zip(axes, ["whoop_sleepdebt", "whoop_sleepperf"], ["Sleep debt (min)", "Sleep performance %"]):
    s = M[[col, "whoop_recovery"]].dropna()
    scatter(ax, s[col], s["whoop_recovery"]); fitline(ax, s[col], s["whoop_recovery"])
    r, p, n = corr(M, col, "whoop_recovery", "spearman")
    ax.set_title(f"{lab}\nρ = {r:+.2f}, p = {p:.3f}, n = {n}")
    ax.set_xlabel(lab); style_axes(ax)
axes[0].set_ylabel("WHOOP recovery %")
fig.suptitle("Sleep debt and sleep performance vs. recovery", x=0.02, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig2_sleep_vs_recovery.png"); plt.close(fig)

# Fig 3 — same-day vs prior-day strain
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, sharey=True)
for ax, col, lab in zip(axes, ["whoop_strain", "prev_whoop_strain"], ["Same-day strain", "Prior-day strain"]):
    s = M[[col, "whoop_recovery"]].dropna()
    scatter(ax, s[col], s["whoop_recovery"]); fitline(ax, s[col], s["whoop_recovery"])
    r, p, n = corr(M, col, "whoop_recovery", "spearman")
    ax.set_title(f"{lab}\nρ = {r:+.2f}, p = {p:.3f}, n = {n}")
    ax.set_xlabel("WHOOP day strain"); style_axes(ax)
axes[0].set_ylabel("WHOOP recovery %")
fig.suptitle("Training fatigue shows up a day late", x=0.02, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig3_strain_lag.png"); plt.close(fig)

# Fig 4 — skin temp vs HRV
fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
s = M[["whoop_skintemp", "whoop_hrv"]].dropna()
scatter(ax, s["whoop_skintemp"], s["whoop_hrv"]); fitline(ax, s["whoop_skintemp"], s["whoop_hrv"])
r, p, n = R["skin_vs_hrv"]
ax.set_title(f"Skin temperature vs. HRV\nr = {r:+.2f}, p = {p:.3f}, n = {n}")
ax.set_xlabel("Skin temperature during sleep (°C)"); ax.set_ylabel("HRV (ms)"); style_axes(ax)
fig.tight_layout(); fig.savefig(FIG / "fig4_skintemp_hrv.png"); plt.close(fig)

# Fig 5 — the alignment bug: self-logged HRV vs device HRV under both keyings
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, sharey=True)
for ax, (label, df, color) in zip(axes, (("Keyed to cycle start (wrong)", M0, ORANGE), ("Keyed to wake onset (right)", M, BLUE))):
    s = df[["zg_hrv", "whoop_hrv"]].dropna()
    scatter(ax, s["whoop_hrv"], s["zg_hrv"], color); fitline(ax, s["whoop_hrv"], s["zg_hrv"], color)
    r, p, n = corr(df, "zg_hrv", "whoop_hrv")
    ax.set_title(f"{label}\nr = {r:+.2f}, p = {p:.3f}, n = {n}")
    ax.set_xlabel("WHOOP HRV (ms)"); style_axes(ax)
axes[0].set_ylabel("HRV I logged in the app (ms)")
fig.suptitle("One join key, two conclusions", x=0.02, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig5_alignment_bug.png"); plt.close(fig)

# ---------------------------------------------------------------- results.md
def fmt(t):
    r, p, n = t
    return f"r = {r:+.2f}, p = {p:.3f}, n = {n}"


lines = ["# Results (regenerated by analyze.py)", "",
         f"WHOOP days: {R['n_whoop_days']} · overlap with self-logs (wake-onset keyed): {R['n_overlap_wake']} · (cycle-start keyed): {R['n_overlap_start']}", "",
         "## A. Alignment check — self-logged vs device", "",
         "| Pair | Cycle-start keyed | Wake-onset keyed |", "|---|---|---|"]
for k, lab in [("hrv_self_vs_device", "HRV, self-logged vs WHOOP"), ("readiness_vs_recovery", "Readiness (self) vs WHOOP recovery"), ("sleep_self_vs_device", "Sleep hours, self vs WHOOP")]:
    lines.append(f"| {lab} | {fmt(A['cycle_start'][k])} | {fmt(A['wake_onset'][k])} |")
lines += ["", "## B. Same-day correlates of WHOOP recovery (Spearman, wake-onset keyed)", "", "| Variable | ρ | p | n |", "|---|---|---|---|"]
for k, (r, p, n) in sorted(B.items(), key=lambda kv: -abs(kv[1][0]) if kv[1][0] == kv[1][0] else 0):
    lines.append(f"| {NICE.get(k, k)} | {r:+.2f} | {p:.3f} | {n} |")
lines += ["", f"Skin temp vs HRV (Pearson): {fmt(R['skin_vs_hrv'])}", f"Bodyweight vs recovery (Pearson): {fmt(R['weight_vs_recovery'])}", "",
          "## C. Prior-day vs same-day (Spearman)", "", "| Variable | ρ | p | n |", "|---|---|---|---|"]
for k, (r, p, n) in C.items():
    lines.append(f"| {k} | {r:+.2f} | {p:.3f} | {n} |")
lines += ["", f"## D. Standardized OLS, recovery ~ features (n = {R['ols']['n']}, R² = {R['ols']['r2']:.2f})", "", "| Feature | β (recovery pts per SD) |", "|---|---|"]
for k, v in sorted(R["ols"]["beta_per_sd"].items(), key=lambda kv: -abs(kv[1])):
    lines.append(f"| {NICE.get(k, k)} | {v:+.1f} |")
lines += ["", f"## E. Random-forest importance (n = {R['rf']['n']})", "", "| Feature | importance |", "|---|---|"]
for k, v in imp:
    lines.append(f"| {NICE.get(k, k)} | {v:.3f} |")
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "results.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
print("\n".join(lines))
