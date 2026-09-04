# Point-in-time feature evidence

> Historical generation: the later cutoff audit withdrew the release candidate's
> strict T−24 claims. Recorded scores and checks below retain their original scope;
> they do not establish corrected forecast performance. See
> [current status](PROJECT_STATUS.md) and [current results](CURRENT_RESULTS.md).

The version 2 feature builder replaces the repository's retrospective aggregate
encodings with an auditable calendar-year state machine. For a flight in year
`Y`, every outcome prior and weather climatology is based only on years strictly
less than `Y`. Current-year records are transformed first and added to state only
after the entire year has been written.

## Variants

- `all_history`: expanding sufficient statistics from 2010 onward.
- `exponential_decay_2y`: the same statistics with a prespecified two-year
  half-life, allowing adaptation while preserving strict chronology.

Route, airline, origin, destination, and origin/month/weekday/hour priors use
empirical-Bayes shrinkage toward the applicable historical global rate. Both total
schedule support and observed delay-label support are exposed because cancellation
and arrival-delay denominators differ.

Weather climatology is aggregated at unique airport-date granularity before
airport-month pooling. This prevents busy airports or repeated flight rows from
implicitly receiving more weather weight. Missing airport-month normals fall back
to the historical month and then the historical overall mean, with an explicit
missing indicator. Realised daily weather is retained under `oracle_*` names and is
not admitted by the schedule-horizon feature contract.

## Cohorts and labels

The legacy 2010--2024 input is an approximately month-balanced sample restricted to
100 airports. The 2025 input is the complete eligible BTS cohort within those same
airports. Counts are therefore effective support within the recorded cohort, not
national traffic totals. Any 2026 operational update must account for the change
from sampled to census support rather than interpreting it as traffic growth.

Cancelled rows have no conditional delay label. In addition, 34 non-cancelled
legacy rows lack `ArrDel15`; they remain eligible for cancellation modeling but are
explicitly ineligible for delay and joint-state modeling. The failed version 1 run
that detected this condition is retained in
`manifests/failures/point_in_time_all_history_v1_failed.json`.

## Reproduction and validation

```powershell
python -m flightdelaybench.point_in_time `
  --output-dir data/derived/point_in_time/all_history_v2 `
  --manifest manifests/point_in_time_all_history_v2.json

python -m flightdelaybench.feature_validation `
  manifests/point_in_time_all_history_v2.json `
  --report reports/validation/point_in_time_all_history_v2.json
```

The decayed variant adds `--half-life-years 2`. Builders and validators are
create-only: existing evidence is never overwritten. Each manifest is self-hashed;
the validator checks all source/output/state hashes, all rows for null and domain
violations, label reconciliation, identifier uniqueness within partitions, and an
independent reconstruction of the global prior chronology from raw labels.

Both complete version 2 variants contain 17,418,780 rows in 26 partitions spanning
2011--2025. Their tracked validation reports have status `PASS`.
