# qtf-montecarlo

Monte Carlo cross-check for the survey-calibrated model (Model A) in

> Pierrelouis, N. *Competing timelines under partial observation: measurement-based exposure analysis of post-quantum migration.* Under review.

The paper's reported quantities come from a deterministic quadrature engine. This script reproduces the same Model A scenario grid and sensitivity sweeps by sampling, as an independent check that the two agree to within sampling error.

## What it computes

For each of nine scenarios (three CRQC emergence timelines x three migration timelines) it draws 10,000 (emergence year, migration completion year) pairs and reports

- `p_vuln`: the fraction of draws where migration finishes after the CRQC arrives
- `exp_yrs`: the median vulnerability window, in years, among those draws

It then sweeps the emergence median (2029 to 2040) and the deployment duration (3 to 12 years) one at a time, holding the other at its central value, and writes three figures.

Emergence is a log-normal offset from 2026. Migration start and duration are truncated normals. All parameters are in the two tables at the top of the script, with their sources.

## Run

```
pip install -r requirements.txt
python qtf_montecarlo.py
```

Runs in a few seconds on a laptop. Output goes to stdout; the figures are written to the working directory.

## Checking your run

`expected_output.txt` is the stdout from the pinned environment in `requirements.txt`. A fixed seed is set, so with the same NumPy and SciPy versions your output should match it exactly. Different versions may change the random stream and move values by a few tenths of a percentage point; that is sampling noise, not a model change.

The reference cell (moderate emergence x baseline deployment) returns 51.74% here against 51.56% from the paper's exact engine. The gap is within the binomial standard error for 10,000 draws at p = 0.5 (about 0.5 points).

## Files

| File | Purpose |
|---|---|
| `qtf_montecarlo.py` | The whole analysis. No other code. |
| `requirements.txt` | Pinned versions used to produce `expected_output.txt`. |
| `expected_output.txt` | Reference stdout for verifying a run. |
| `figures/` | The three figures as produced by the reference run. |

## License

MIT. See `LICENSE`.
