# Small issued-weather archive feasibility pilot

The preserved run is `data/external/cutoff_taf_pilot_v1/report.json`, with original
API responses, original NWS text, retrieval metadata and hashes beside it. It
uses no flight labels. Four airports and four fixed 2024 cutoffs were selected
before retrieval: January 15 and July 15 at 06:00 and 12:00 UTC. This is not a
representative airport sample or a flight-weighted coverage estimate.

| Station | Source headers audited | T+24 covered, producer time + 15 minutes <= cutoff |
|---|---:|---:|
| JFK | 4 | 4 |
| O'Hare | 4 | 4 |
| Des Moines | 4 | 0 |
| Central Wisconsin | 4 | 0 |

No returned product had a producer timestamp after its requested cutoff.
Coverage uses a strict valid-start <= target < valid-end interval. A forecast
expiring exactly at the target time is not extrapolated. In these cases the
30-hour products covered the T+24 and T+27 probe horizons; the 24-hour products
did not. The NWS documents both forecast lengths and routine six-hour issuance
cycles in its [TAF product information](https://aviationweather.gov/help/data/).

An important parsing finding: IEM's single-TAF JSON field `utc_taf_issue` can be
the validity start. For example the Des Moines product at the January 15 06:00
cutoff contains a 05:32 issuance in the original header but 06:00 in that JSON
field. The pilot independently reads the original issue and expiry groups and
the NWS product identity; it does not derive issue time as valid time minus a
nominal forecast lead. See the [IEM API documentation](https://mesonet.agron.iastate.edu/api/1/docs).

The 0-, 15- and 60-minute latency values are assumptions, not measured historical
receipt delays. In particular the recorded 60-minute calculation tests the
*same selected product* and does not re-query for an older still-valid product.
It must not be interpreted as total archive availability under a 60-minute feed
delay. Retrieval happened in 2026; that timestamp is retained separately and is
not relabelled as historical consumer availability.

Decision: retain TAF as a possible issue-stamped, airport-specific input, but not
as a universal replacement for longer-range forecasts at T-24. Before model
experiments, expand coverage across actual target flights, handle amendments and
missingness, and establish a defensible source-availability contract. A forecast
revision/uncertainty feature or resource-capacity scenario model remains an
untested candidate, not an established invention or accuracy improvement.

Historical planned-restriction coverage has not been established by this pilot.
No new architecture gate or predictive-gain gate has passed.
