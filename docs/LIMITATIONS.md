# Limitations

[Documentation index](README.md) · [Current results](CURRENT_RESULTS.md)

## Timing is the principal unresolved issue

The historical prior-day contract is weaker than flight-specific T−24 availability.
A preceding operating date may contain outcomes occurring after the target cutoff,
and BTS retrospective publication is not a live operations feed. Weather issue
proxies, schedule revisions and static airport metadata also need explicit vintage
review. Hashes establish file integrity, not the authenticity of timestamp claims.

The corrected software enforces explicit availability ordering and censors training
and stopping labels. No real-data rerun yet shows what happens to performance after
those constraints are satisfied.

## The retrospective year is not untouched

Earlier 2025 results influenced later method design. Forward model selection in 2024
does not erase that knowledge. Confidence intervals describe uncertainty for the
evaluated comparison, not uncertainty from the entire adaptive research search.
Historical date-cluster intervals also do not remove all multi-day serial dependence.

## Context is not observed operational state

The complete schedule stack does not reveal active runway configuration, actual
acceptance/departure rates, gate allocation, crew legality, maintenance, live aircraft
assignment or all traffic-management restrictions. Runway headings are geometric
envelopes; inferred capacity, queues and shadow prices are proxies. One overload
proxy saturated and some planned fields had zero coverage.

These are plausible missing sources of information, not proof that any specific
feed or architecture would yield a five- or ten-point accuracy gain.

## Coverage and confounding

Both endpoints of scored flights remain in the frozen 100-airport cohort. An
additional 255 airports supply boundary context only; the study does not evaluate
their own delay/cancellation performance. BTS reporting coverage is not every
aircraft movement at an airport. A static October 2024 NASR snapshot is not a
cycle-correct metadata panel for earlier operations.

The boundary campaign changed a CatBoost categorical-interaction limit and ensemble
choices as well as context. Its small accuracy/Brier gain cannot be attributed
solely to small-airport data. The negative residual result is likewise not proof
that boundary information is intrinsically useless.

## Metrics and decision use

An always-on-time classifier achieves 76.5222% on the latest observed joint cohort.
The selected boundary ensemble's 77.3512% accuracy coexists with very low delayed
and cancelled argmax recall. Rankings, probability quality, calibration and
decision thresholds answer different questions; a higher accuracy score is not
automatically better disruption warning.

The benchmark does not establish causal passenger savings, airline safety,
demographic fairness, actual aircraft-identity recovery, generalization outside
the reporting cohort or a universal state of the art. Model outputs must not be
presented as live operational advice.

## Reproduction and redistribution

Large datasets, training models and monthly predictions remain external, bound by
manifests. Source-only tests and legacy-model smoke checks are not a complete
scientific reproduction. Historical peak process-tree memory was not instrumented.

The issued-TAF pilot covers only 16 station/time cases. Latency values are assumptions,
not measured historical receipt. Its 60-minute check keeps the same selected product
instead of searching for an older valid one; it is not total archive availability.

The MIT license covers project software, not every provider dataset. Upstream
licensing and archive redistribution need review before deposit; see
[third-party notices](../THIRD_PARTY_NOTICES.md).
