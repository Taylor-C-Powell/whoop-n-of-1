# whoop-n-of-1

**One person, one month of WHOOP data, and the join bug that changed the findings.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-2ea44f)
![n = 1](https://img.shields.io/badge/n-1-lightgrey)

**[Read the article](ARTICLE.md)** · [Results tables](results.md) · [Run it](#run-it) · [Honest limits](#honest-limits)

![One join key, two conclusions](figures/fig5_alignment_bug.png)

The same two HRV columns, joined two ways. Keyed to cycle start, my self-logged HRV and WHOOP's HRV correlate at r = +0.04. Keyed to wake onset, r = +1.00. A WHOOP cycle's recovery, HRV, RHR, and sleep fields describe the morning you woke up, not the night the cycle started, so my first join labeled every record a day early. Fixing that one line made about half of my original findings evaporate.

This repository reproduces every number and figure in the article
[*What a month of my WHOOP data actually predicted (and the bug that changed the answer)*](ARTICLE.md).
It exists so the analysis can be checked, not just read.

## Findings at a glance

| Claim | First pass (June, misaligned) | After the fix | Verdict |
|---|---|---|---|
| Yesterday's strain lowers today's recovery | — | ρ = −0.59 (p = 0.002, n = 24); same-day ρ = +0.05 | **Survived**, and got stronger |
| Sleep debt hurts, sleep performance helps, raw hours don't | — | debt ρ = −0.42 (p = 0.03); performance ρ = +0.38 (p = 0.05); hours ρ = −0.02 | **Survived** |
| Deep sleep is the top recovery predictor | RF importance 0.35, ranked first | 0.13, ranked fifth; ρ = +0.18 (p = 0.38) | Did not survive |
| Skin temperature tracks HRV | r = −0.50 (p = 0.02) | r = −0.26 (p = 0.20, n = 27) | Weakened |
| Alcohol and caffeine cost recovery points | −7 and −5 points per SD | +2 and +10 points per SD (both signs flipped) | Not usable |
| My self-logged HRV barely tracks the device | r = +0.04 (n = 17) | r = +1.00 (n = 22); the "self-log" was a transcription of the device | The bug itself |
| My self-report readiness score tracks WHOOP recovery | — | r = −0.06 (n = 22) | Did not hold |
| Respiratory rate rises with recovery | — | ρ = +0.52 (p = 0.005) | Unexplained; about what chance hands you across 15 tests |

Full tables are in [`results.md`](results.md); the reasoning is in the [article](ARTICLE.md).

![Training fatigue shows up a day late](figures/fig3_strain_lag.png)

Prior-day strain is the one relationship that a rank correlation, a linear model, and a random forest all agree on: one standard deviation of it costs about ten recovery points. The recovery score is computed at wake, before the day's strain exists, so the direction is expected. What is worth seeing is how clean the lag is.

## Run it

macOS / Linux:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analyze.py
```

Windows (PowerShell):

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python analyze.py
```

`analyze.py` reads `data/`, writes `figures/*.png`, `results.md`, and `results.json`, and prints the results table. It runs in a few seconds.

Checked from a clean clone on 2026-09-19 (Python 3.12, fresh virtual environment): `results.md` and `results.json` regenerate byte-for-byte.

## What's in here

| Path | What it is |
|---|---|
| `data/whoop/physiological_cycles.csv` | The WHOOP export (31 cycles, May 5 – Jun 4 2026): recovery, HRV, RHR, skin temp, SpO₂, strain, sleep staging |
| `data/whoop/sleeps.csv`, `data/whoop/workouts.csv` | The rest of the export, for anyone who wants to go further |
| `data/daily_master.csv` | My daily log for the same window: training sessions (duration × session RPE), caffeine, alcohol, water, bodyweight, and a self-report readiness composite |
| `analyze.py` | The whole analysis: two join keyings, correlations, lagged correlations, standardized OLS, random-forest importance, five figures |
| `results.md` / `results.json` | Regenerated output; the article quotes from these |
| `ARTICLE.md` | The write-up |

The WHOOP journal entries (yes/no behaviour questions) are not included; they add nothing the daily log doesn't already have and they are the most personal part of the export.

## The three decisions that matter

1. **WHOOP cycles are keyed to wake-onset date.** The recovery score, HRV, RHR, and sleep fields in a cycle row describe the morning you woke up. Keying to cycle start (the previous night's sleep onset) labels every record a day early. Section A of the results shows the same two HRV columns at r = 0.04 under the wrong key and r = 1.00 under the right one.
2. **One cycle per day.** Naps and split nights produce extra cycles with their own short sleep and tiny recovery score. The longest sleep of each wake date is kept; the rest are dropped rather than averaged in.
3. **"Prior day" means the previous calendar day**, computed by shifting dates, not by taking the previous row.

## How this differs from my June notes

My internal June write-up reported deep sleep as the top recovery predictor (RF importance 0.35), skin temp vs HRV at r = −0.50, alcohol at about −7 recovery points per SD, and self-logged HRV vs device at r = 0.68 after a partial fix. This repo does not reproduce those numbers, and the article says so. The differences come from decisions 1–3 above (the June re-run fixed the join key but still averaged duplicate cycles and lagged by row) and from the readiness snapshot in `daily_master.csv`, which predates a June 4 recompute in the app. Where the two disagree, this repo is the version I stand behind, because it is the one anyone can run.

## Honest limits

- n = 1. Nothing here generalizes to anyone else.
- 27 usable device days; 22 overlap with the daily log; the regression uses 21 rows.
- Fifteen same-day variables were tested with no multiple-comparison correction. At least one "significant" result is expected by chance, and one (respiratory rate, ρ = +0.52) looks like it.
- Consecutive days are autocorrelated, so every p-value is optimistic.
- The "self-logged" HRV and sleep hours were transcribed from WHOOP each morning. They are a transcription check, not an independent measurement.

## License

Code: MIT. Data: released for non-commercial analysis and reproduction of this article; it is my own physiological data and I'd ask that it not be redistributed as part of any aggregated dataset.

## Author

Taylor C. Powell — [linkedin.com/in/taylor-c-powell](https://www.linkedin.com/in/taylor-c-powell/) · taylorp661@gmail.com
