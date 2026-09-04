# Schedule-context evidence limitation note

Status: binding interpretation note for the v1 schedule-context development runs  
Recorded: 2026-09-03

The following immutable tuning reports used the phrase "full target-day cohort
schedule census" in their embedded construction profile:

- `runs/model_tuning/recent_context_direct_delay_2018_v1.json`
  (SHA-256 `32c406f18c0e90a3eb0528954e45a81185af8633f0ee7fd83155c4e472a5c8aa`)
- `runs/model_tuning/recent_context_network_cancellation_2018_v1.json`
  (SHA-256 `819cb153a44ec3a74bbab6b4de89ab7cf5de5f2a1fef7680d052ac691b775b47`)

That wording describes a census *within the available target-day cohort*, not a
census of the actual published schedule. The source manifest labels 2011--2024 as
`legacy_sample`; its filename also identifies it as a balanced research dataset.
The separately acquired 2025 cohort excludes diverted flights upstream. Therefore,
the resulting counts are cohort-density proxies and cannot substantiate a
schedule-census, operational-feed, or deployment claim.

The numerical results remain valid for the recorded retrospective cohorts and are
preserved without alteration. They are supplementary sensitivity evidence only.
Promotion requires reproduction from a complete, timestamped advance schedule
snapshot. The official BTS census track added later retains diversions for more
faithful retrospective counts, but it too does not establish equivalence to an
advance schedule feed.
