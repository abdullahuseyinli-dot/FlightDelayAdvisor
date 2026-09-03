# FlightDelayBench research protocol

Protocol version: `1.0.0-draft`  
Evidence status: `DEVELOPMENT`  
Confirmation status: `UNOPENED`

## Research question

How do information availability, forecast lead time, and temporal regime shift affect
the discrimination, calibration, and decision value of US domestic flight disruption
predictions?

The study is designed around the information that could have been known when a
prediction was issued. It does not treat retrospective access to a variable as proof
that the variable was operationally available.

## Prespecified hypotheses

1. Models evaluated with realised same-day weather will appear better than deployable
   schedule/climatology models, and the difference will vary across years.
2. Archived weather forecasts issued before departure will improve proper scoring
   rules over climatology, with value increasing as departure approaches.
3. Strictly prior-only, time-decayed reliability features and rolling recalibration
   will improve out-of-time log loss or Brier score over fixed 2010--2018 aggregates.
4. A coherent three-state disruption model or calibrated hurdle model will improve
   joint probability quality over independently interpreted binary outputs.
5. Model rankings will be less stable across years than within a single pooled test
   period; simple baselines may remain competitive under severe shift.

Hypotheses 1--4 are primary method questions. Hypothesis 5 is a planned robustness
analysis. A result may be informative even when an intervention does not improve the
primary metrics.

## Evidence boundaries

- **Development:** rolling-origin evaluations for 2019--2023, using only earlier
  years for fitting and feature statistics.
- **Selection/calibration:** 2024. It may select a fixed candidate, calibrator,
  ensemble, smoothing strength, and time-decay rule.
- **Retrospective audit:** full-year 2025. January--July outcomes were used by the
  historical repository, so 2025 is not represented as an untouched test.
- **Confirmation:** January--June 2026. These outcomes remain unopened until code,
  feature contracts, selected models, metrics, and artifact hashes are written to the
  confirmation lock manifest.

The fixed-origin analysis trains through 2024 and evaluates both 2025 and 2026. A
separate operational-update analysis may refit the already locked method through 2025
before predicting 2026; it may not change the method based on 2025 results.

## Prediction targets

- Arrival delay of at least 15 minutes, conditional on a flight not being cancelled
  or diverted.
- Cancellation, among non-diverted scheduled flights.
- Joint disruption state: on time, delayed, or cancelled.

Rows with a binary cancellation label but no arrival-delay outcome are retained for
the cancellation task and explicitly marked ineligible for the delay and joint
tasks. They are never imputed as on time. The legacy archive contains 34 such rows:
32 in 2018, one in 2020, and one in 2021.

The conditional delay and cancellation probabilities must not be presented as
independent mutually exclusive probabilities. Joint analyses use either a multinomial
model or a hurdle construction whose probabilities sum to one.

## Feature availability

Every model input is registered in `flightdelaybench.contracts`. The primary horizon
uses schedule variables, historical statistics calculated strictly before the target
year, and historical climatology. Archived numerical-weather forecasts may enter only
the horizon for which their issue timestamp precedes the cutoff. Realised daily
weather is retained only as an oracle diagnostic.

Actual departure/arrival times, realised delays, delay causes, same-day realised
traffic totals, current/future labels, and aggregate encodings containing the current
row are prohibited.

## Candidate families

The minimum comparison set is:

1. prevalence and seasonally smoothed historical-rate baselines;
2. regularised logistic regression;
3. gradient-boosted trees;
4. native-categorical boosted trees;
5. empirical-Bayes prior features with fixed and time-decayed histories;
6. static and rolling calibration;
7. independent hurdle and coherent joint-state probability models.

Neural or graph models are admitted only if they answer a prespecified mechanism
question and are compared at a reasonable compute/data budget. They are not required
for a successful study.

## Metrics and uncertainty

Primary metrics are log loss, Brier score, and Brier skill relative to the applicable
historical-prevalence baseline. Secondary measures include AUROC, average precision,
calibration intercept/slope, equal-mass ECE, top-decile lift/capture, classwise joint
Brier score, and risk--coverage curves.

Primary paired comparisons resample flight dates as clusters. Airline, airport,
route, month, year, lead-time, and disruption-regime results are subgroup/reliability
analyses, not demographic fairness estimates. Multiplicity and small groups remain
visible.

## Decision replay

A reliability-only historical replay may rank observed alternatives within the same
origin, destination, date, and declared time window. It does not establish that a
passenger could buy or use each alternative, and it makes no causal claim because
fares, inventory, preferences, connections, and schedule changes are unavailable.

## Claim limits

The study does not claim real-time operational deployment, causal passenger benefit,
airline safety, complete schedule-snapshot fidelity, demographic fairness, or
generalisation outside the observed US reporting network. The historical dataset is
sampled and airport-restricted; every report must state the exact cohort.

The 2010--2024 input is an approximately month-balanced legacy sample, whereas the
2025 retrospective cohort is the complete eligible BTS census within the same
100-airport restriction. Census results are primary for the 2025 deployment-like
audit; matched-sampling and reweighting analyses must be reported as robustness
checks so cohort construction is not mistaken for temporal drift.
