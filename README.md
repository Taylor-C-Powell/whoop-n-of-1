# whoop-n-of-1

One person, one month of WHOOP data, and the join bug that changed the findings.

This repository reproduces every number and figure in the article
[*What a month of my WHOOP data actually predicted (and the bug that changed the answer)*](ARTICLE.md).
It exists so the analysis can be checked, not just read.

```
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python analyze.py
```

`analyze.py` reads `data/`, writes `figures/*.png`, `results.md`, and `results.json`, and prints the results table. It runs in a few seconds.

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
