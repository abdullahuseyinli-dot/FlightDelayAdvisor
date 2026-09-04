# Third-party notices

The [MIT license](LICENSE) covers this project's software. It does not relicense
upstream datasets, service software, dependencies or separately attributed material.
Provider acknowledgements do not imply endorsement.

## Data and services

| Source | Use in this repository | Attribution and limits |
|---|---|---|
| US DOT Bureau of Transportation Statistics | Reporting Carrier On-Time Performance schedules and outcome labels | Acknowledge BTS and identify the extract, years and processing; provider metadata governs the original records |
| FAA NASR | Runway/reference airport geometry, including the 2024-10-03 cycle | Acknowledge FAA; geometry is not active runway or historical operational capacity |
| Open-Meteo / NOAA GFS | Archived fixed-lead weather API values and derived aviation features | Open-Meteo API data is CC BY 4.0; retain attribution, a license link and a description of transformations |
| Iowa Environmental Mesonet / NOAA NWS | Issued TAF JSON and original text in the small archive pilot | Credit IEM at Iowa State University and NOAA NWS; preserve original product identity and retrieval metadata |

Open-Meteo distinguishes its API data license from its server software license.
This project consumes the API and does not redistribute the Open-Meteo server.
Weather transformations include airport/time selection, joining, wind/runway and
corridor derivatives; derived values are not original provider measurements.
See [Open-Meteo licensing and attribution](https://open-meteo.com/en/licence) and
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

IEM describes its website materials as public domain and requests attribution;
its products are supplied without an accuracy warranty.
See the [IEM disclaimer](https://mesonet.agron.iastate.edu/disclaimer.php),
[dataset notes](https://mesonet.agron.iastate.edu/info/datasets.php) and
[API documentation](https://mesonet.agron.iastate.edu/api/1/docs).
Archival retrieval does not establish historical consumer receipt.

Source/acquisition identities and hashes are retained in the corresponding
`manifests/` records. The notices here are not a blanket redistribution clearance
for every future feed or archive. Review the precise payload and provider terms
before depositing data or models.

## Software dependencies

Dependency declarations are in [pyproject.toml](pyproject.toml), resolved packages in
[uv.lock](uv.lock), and measured versions in individual run records. NumPy, pandas,
SciPy, scikit-learn, PyArrow, DuckDB, CatBoost, LightGBM, Torch, TabM and other
dependencies retain their upstream notices and licenses. A project dependency
declaration is not a claim of authorship of those libraries.

## Methods and contributors

The [literature map](docs/LITERATURE_AND_NOVELTY.md) records methodological sources
and unresolved novelty boundaries. FLARE-24, CC-RTH and BC-POT-R are project method
labels, not claims to have invented tree boosting, tabular neural networks,
graph learning, reconciliation or delay propagation.

The [legacy design document](docs/FlightDelayAdvisor_Documentation.md) retains its
original named contributors. Repository citation metadata does not replace that
attribution or assert authorship of upstream data and methods.
