FlightDelayAdvisor

Intelligent flight disruption risk advisor for US domestic routes

Version: 1.0
Authors: Abdulla Huseyinli, Ibrahim Alkali, Khalid Suliman, Kaixin Zhu,
Raghuraman Thirugnana Sambandam Mythily

1. Overview

FlightDelayAdvisor is an interactive web application that helps users understand and manage the risk of delays and cancellations for US domestic flights. It combines more than a decade of operational performance data with historical airport weather and modern machine‑learning models to turn raw data into clear, actionable guidance.

At a high level, the app lets a user:

Pick an airline, route, date, and departure time

See probabilities of:

- Arrival delay of at least 15 minutes, and outright cancellation

- Explore which hours, airlines, and airports tend to be riskier

- Discover safer alternatives (e.g. different departure hours, airlines, or nearby airports) based on historical performance, not guesswork.

1.1 Data & modelling foundations

Under the hood, FlightDelayAdvisor is built on:

2010–2024 US DOT/BTS on‑time performance data

Tens of millions of historical flight legs with detailed operational outcomes (delays, cancellations, carriers, routes, schedules).

Airport‑level historical weather

Monthly “climatology” for temperature, wind, precipitation, snow, and a derived bad‑weather indicator at both origin and destination. Weather is used as a pattern feature, not as a live forecast.

Gradient‑boosted models for tabular data

CatBoost for predicting delay ≥ 15 minutes

LightGBM for predicting cancellations

Both are trained on a rich feature set (calendar, route/airline reliability statistics, congestion indicators, and weather) using time‑based train/validation/test splits to mimic real‑world generalisation.

Out‑of‑time backtesting on 2025 flights

Models trained on 2010–2024 are evaluated on separately prepared 2025 BTS data, using the same feature pipeline as the app. This gives a realistic view of how the system behaves on truly unseen data, not just on a random test split.

From a user’s perspective, all of this complexity is hidden behind a simple interface; they see only probabilities, explanations, and concrete suggestions.

1.2 Core capabilities

The Streamlit UI is organised into three high‑level capabilities:

Single flight risk estimate:

Users specify a concrete itinerary (airline, origin, destination, date, departure hour, and optional return leg). The app returns:

- Probability of delay ≥ 15 minutes for each leg

- Probability of cancellation for each leg

- For round trips, the combined risk of “at least one delayed” and “at least one cancelled” leg

- Short, human‑readable explanations of why the risk is high or low (seasonality, route behaviour, airline reliability, congestion, weather patterns)

- Model‑based alternatives, such as safer departure hours for the same flight or other airlines that historically perform better on that route.

- In other words: this screen answers “If I book this specific flight, how risky is it and what easy changes could make it safer?”

High‑risk hours exploration
This view uses raw historical data (not model predictions) to show how delay and cancellation rates vary by departure hour for:

all airports and airlines, or a specific origin airport and/or airline, optionally filtered by month.

Users can see:

- Line charts of delay and cancellation rates by hour

- Hour×month heatmaps for delay risk (when all months are selected)

- Bar charts of hourly flight volumes for context

- Automatically highlighted safer and riskier hours to prioritise when choosing a departure time.

- This screen answers “If I’m flexible on departure time, which hours are historically safest for leaving this airport / with this airline?”


Airline, airport, and state‑level comparison. 

This section exposes the underlying reliability patterns in a way that’s easy to explore:

 Airlines tab – compare airlines on a specific route (e.g. JFK→LAX), with:

 - delay and cancellation rates per carrier

 - minimum‑volume filters (only show airlines with enough flights)

 - simple summaries of “best on delay” and “best on cancellations” for the route.


 Airports tab – rank airports by delay or cancellation rates, either:

 - as origins (which airports are risky to depart from?)

 - as destinations (which airports are risky to arrive into?), possibly conditioned on a focus airport (e.g. “best destinations from JFK”).


 States tab – recommend good origin/destination airport + airline combinations between US states, based on historical non‑stop (or one‑stop) patterns and combined delay/cancellation risk.

 - Collectively, these tools answer questions like “Which airline should I favour on this route?”, “Is there a better nearby airport?”, and “What are the most reliable ways to travel from state A to state B?”


1.3 Scope & design choices

To keep the product focused and interpretable, FlightDelayAdvisor deliberately makes a few clear choices:

Historical‑pattern, not real‑time forecasting:

The app applies long‑term patterns (2010–2024 performance + monthly weather) to user‑selected dates, rather than consuming live ATC or short‑term weather feeds. This makes the system:

easier to deploy and reproduce, and

more stable for educational and planning use.

Users are explicitly reminded that probabilities reflect historical tendencies, not “today’s live situation”.

Future dates limited to a fixed horizon (e.g. 2025)
The current build only allows dates within 2025. This is intentional:

- it avoids speculating too far beyond the range for which we’ve validated the models, and

- it keeps the feature space (e.g. airline behaviours, route structures) aligned with what the model actually “knows”.


This is the kind of constraint we commonly see in production‑grade ML products: the app prefers honest generalisation over unrealistic claims.


Probability‑first UX

The app always surfaces calibrated probabilities instead of hard YES/NO classifications. Users can interpret these as “chances out of 100” and compare options (e.g. 18% vs 32% delay risk) instead of being given black‑box decisions.

Heavy evaluation behind a simple interface
The underlying models are evaluated with:

- ROC‑AUC, PR‑AUC, and Brier score

- temporal robustness (metrics by year)

- group diagnostics (per‑airline reliability of predictions)

- out‑of‑time backtesting on 2025 data

Only after this evaluation does the team expose the best‑performing, best‑calibrated models in the app. This is standard practice in production ML: the UI you see is backed by models that have already passed a battery of tests.

1.4 Target audience & usage scenarios

FlightDelayAdvisor is designed for:

- Travellers and travel planners

- Choosing flights and departure times with a clearer sense of disruption risk.

- Comparing airlines or nearby airports before booking.

Airline and airport analysts

- Inspecting empirical reliability patterns across time of day, route, airline, and airport.

- Using the “Explore & compare” views as a quick, visual diagnostic on performance.

- Product, data, and operations teams

- Evaluating an ML‑based risk scoring service before integrating it into other products (e.g. booking flows, ops dashboards, alerting tools).

- Stress‑testing model behaviour on future data (via the 2025 backtest pipeline) in a way that is transparent and reproducible.


2. Key design decisions

2.1 Training window and prediction horizon

Training data window: 2010–2024 (US DOT/BTS on‑time performance + airport‑level weather)

Prediction horizon in the app: flights departing in calendar year 2025

This is a deliberate choice, not a limitation.

Long, diverse history (2010–2024).
The models see over a decade of operations across:

- multiple airlines and business models (full‑service, low‑cost),

- a full range of seasons and holiday peaks,

- different congestion regimes and schedule strategies,

- a wide variety of weather patterns at each airport.

This breadth makes the model much more robust than a “last 12 months only” approach and helps it generalise across unusual but historically observed situations (e.g. severe winters, summer storms, busy holiday weekends).

Forward‑looking predictions, not hindsight.

The app only allows the user to select dates in 2025. Behind the scenes, the models only “know” about 2010–2024 when they make those predictions. That means:

- No information from 2025 is used at training time.

- Every prediction behaves like a genuine out‑of‑sample forecast.

- We avoid the common mistake of “peeking into the future” that makes models look better in notebooks than they really are in production.

Explicit out‑of‑time backtest for 2025.
2025 is treated as a hold‑out year in our evaluation pipeline:

- A dedicated backtest script scores the deployed models on real 2025 BTS data.

- The same feature builder used in the app is applied to 2025 flights, so backtest results reflect exactly what users see in production.

- This backtest is what we use to sanity‑check generalisation before shipping a model update.

Design intent:
Think of FlightDelayAdvisor as behaving like a production system that an airline or OTA would run internally:

“Train on all historical data up to the end of last year, validate carefully on the first unseen year, and only then expose predictions to users.”

By fixing the training horizon at 2010–2024 and the prediction horizon at 2025, we get the best of both worlds: rich historical coverage and a clean, auditable separation between “what the model learned” and “where we evaluate it”.

2.2 Historical, not real‑time – by design

FlightDelayAdvisor is intentionally historical‑pattern‑driven, not a live weather or ATC feed.

Instead of streaming real‑time METAR/TAF or air‑traffic‑control data, the app uses:

- Monthly climatological weather per airport
(e.g. typical temperature, precipitation, snow and wind patterns for JFK in January vs July).

- Long‑run congestion and reliability statistics
(e.g. how busy certain departure hours are at ATL on Mondays, or how delay‑prone a specific airline–route pair has been historically).


These design choices are deliberate and have several practical advantages:

2.2.1 Reproducible and auditable

Because we rely on static, versioned datasets (2010–2024 BTS + matched weather), every prediction is fully reproducible:

Given the same input (airline, airports, date, hour), the model will always return the same probability, independent of current weather or operational noise.

This makes it easy to:

- debug odd cases (“why did we call this flight high‑risk?”),

- compare model versions over time, and

- reproduce figures and tables in the accompanying technical report.

For stakeholders (airlines, regulators, universities), this is critical: they can verify that the app is driven by measurable historical patterns, not by an opaque live feed that is hard to audit.

2.2.2 Honest about what we’re predicting

Many “delay prediction” tools implicitly mix forecasting and pattern‑matching, which can confuse users:

If you show a probability next to a live radar map, users naturally interpret it as a short‑term weather forecast.

In reality, most tabular ML models are answering a different question:

- “Given flights like this in the past, how often did they get delayed or cancelled?”

FlightDelayAdvisor leans into that reality and makes it explicit:

- All explanations in the UI are framed as “historically, in situations like this, disruption was more/less likely.”

We do not claim to know whether there is a specific storm or ATC restriction on a given day.

This keeps user expectations aligned with what the model actually knows, and avoids over‑promising.

2.2.3 Simpler, more robust operations

From an engineering point of view, avoiding real‑time feeds has clear benefits:

No dependency on live weather APIs or ATC systems.

No streaming infrastructure or hard real‑time SLAs just to get a prediction.

The app can be deployed as a self‑contained web service backed by static feature tables and saved models.

For a first production deployment (especially in an academic or pilot setting), this dramatically reduces operational risk: there are fewer moving parts, fewer things that can break, and a much easier path to containerisation and CI/CD.

2.2.4 Ready for real‑time when needed

Even though the current version is historical by design, the feature layout is future‑proof:

Time‑invariant features like distance, route history, and airline statistics are cleanly separated from time‑varying drivers like:

- departure hour and day of week,

- congestion indicators,

- weather features (Origin_tavg, Origin_prcp, etc.).

Real‑time signals can be plugged into those weather and congestion slots later:

- swap monthly climatology for now‑casts from a weather API;

- add live departure bank load based on current schedule and known disruptions.

That means the current version is already doing something valuable and dependable today, while the architecture naturally supports a v2 that integrates live feeds without forcing a redesign or a breaking change to the app.

In other words: today FlightDelayAdvisor is a “historical risk advisor” that you can trust to be stable and explainable; tomorrow it can evolve into a hybrid system that blends those long‑run patterns with live operational data.


3. System architecture (high‑level)


[Figure 1 – System architecture diagram]
Boxes for Data layer → Feature & Metadata layer → Model layer → Streamlit app.

3.1 Data layer

Historical dataset:
data/processed/bts_delay_2010_2024_balanced_research_weather.parquet

Contains:

- Raw BTS flight fields (Year, Month, DayOfWeek, Origin, Dest, Reporting_Airline, Distance, ArrDel15, Cancelled, etc.).

- Weather features per (airport, date) for origin and destination.

- Aggregated route/airline/congestion statistics used as features. 


2025 backtest dataset: 

Prepared via prepare_bts_2025_for_backtest.py, matches the historical schema.

3.2 Feature & metadata layer

app.load_metadata() loads the processed dataset and constructs a series of reusable metadata tables and defaults, such as: 

- Route‑level performance per airline (route_meta)

- Delay/cancellation rates and volume for each (airline, origin, dest).

- Airline‑level performance (airline_meta)

- Network‑wide delay/cancellation rates and volume for each airline.

- Congestion by time slot (slot_meta)

- Typical number of flights per (Origin, Month, DayOfWeek, DepHour).

- Daily airport and airline‑airport volume

- Daily total departures per airport and per (airport, airline).

- Route‑pair meta (route_pair_meta)

- Route volume and disruption rates aggregated across airlines for each (origin, dest).

- Route distances (route_distance_meta)

- Monthly climatological weather for origin and destination airports.

- Global defaults (overall delay rate, cancellation rate, etc.) for unseen combinations.

- Airport ↔ state mapping built from BTS state fields; used in the state‑based recommendations tab. 


All app and backtest features are derived via a single function, build_feature_row(), which ensures consistent feature construction across training, UI and backtesting. 



3.3 Model layer   

Models are trained in train_models.py on a temporally separated train/validation/test split: 

- Delay model: CatBoost classifier, calibrated with isotonic regression.

- Cancellation model: LightGBM classifier, also calibrated.

Training scripts also:

- Balance class distributions.

- Tune hyperparameters with cross‑validation.

- Produce diagnostic plots and fairness/drift checks (for the report).

- The app loads the deployed models via load_models() and uses predict_proba for probability outputs.

3.4 Application layer

Streamlit app (app.py) exposes the models through an interactive interface:

- Three top‑level tabs:

- Single flight

- High‑risk hours

- Explore & compare (subtabs: Airlines, Airports, States) 

- All predictions and visualisations are based solely on historical data + models, with no side effects.


4. Installation & deployment

This section is for anyone running or deploying the app.
 
4.1 Prerequisites

- Python 3.10+ (or the version used in your project)

- Git

- A recent version of pip

- Optional: virtual environment 

4.2 Local setup
  
# 1. Clone the repository
git clone <https://github.com/abdullahuseyinli-dot/FlightDelayAdvisor> FlightDelayAdvisor
cd FlightDelayAdvisor

# 2. Create and activate a virtual environment (Windows PowerShell example)
python -m venv .project1venv
.\.project1venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (If needed) Prepare processed historical dataset
python .\src\prepare_dataset.py

# 5. (If retraining) Train models
python .\src\train_models.py

# 6. Run the app
streamlit run .\src\app.py

4.3 Backtesting on 2025 data

To validate deployed models on real‑world 2025 data:
# Prepare normalized 2025 dataset (if you use the real-world pipeline)
python .\src\prepare_bts_2025_for_backtest.py

# Run backtest (example with 200k rows for speed)
python .\src\backtest_2025_real_world.py --max-rows 200000 > reports\backtest_2025_log.txt

The backtest writes a concise summary to reports/backtest_2025_metrics.txt, including ROC‑AUC, PR‑AUC, Brier score, and how concentrated risk is in the top X% of flights.


5. Using the app

[Screenshot placeholder – Full home screen showing tabs]

5.1 Landing page

On launch, the app displays:

- The title and a short description.

- An “How to use FlightDelayAdvisor” expander explaining the main capabilities.

- Tabs for Single flight, High‑risk hours, and Explore & compare. 

The app loads metadata and models once and reuses them across interactions.


6. Single flight risk estimate

[Screenshot placeholder – Single flight tab with input form and results panel]

The Single flight tab is the primary entry point for end users. It provides a personalised risk assessment and alternatives for a specific itinerary.

6.1 Input fields

Left‑hand column:

Airline – BTS carrier code (e.g. AA, DL, UA).

Flight date – limited to 2025‑01‑01 to 2025‑12‑31.

Rationale: predictions are calibrated for the year immediately after the training window (2010–2024). This mimics how production systems are deployed with a one‑year horizon and avoids extrapolating too far into the future.

Origin airport – US domestic airport.

Destination airport – US domestic airport (must differ from origin).

Scheduled departure hour – local time, 0–23.

Optional round‑trip section:

- Checkbox to “Estimate round‑trip risk for this itinerary”.

- Return date (≥ outbound date).

- Return departure hour.

6.2 What the app computes

Under the hood, the app builds a feature vector for the requested itinerary via build_feature_row() using:

- Calendar features – Year, Month, DayOfWeek, DayOfYear, weekend flag, holiday season flag.

- Time of day – departure hour (numeric plus sine/cosine encoding).

- Route‑level performance – historical delay/cancellation rates and volume for this airline–route pair.

- Airline‑level performance – the carrier’s global delay/cancellation profile and volume.

- Congestion – typical number of flights in this origin–month–day‑of‑week–hour slot.

- Weather climatology – average temperature, precipitation, snow, wind speed, and frequency of “bad weather” days at both origin and destination for that month.

- Distance and route encodings – great‑circle distance, route ID, season and distance band.

These features are passed into the deployed models to obtain probabilities:

- Delay ≥ 15 minutes probability

- Cancellation probability

6.3 Output and interpretation

Right‑hand column:

Two metric cards showing:

- Delay ≥ 15 min probability with percentage and a qualitative label (Low / Medium / High) and emoji.

- Cancellation probability with percentage and qualitative label.

- Short text comparing cancellation risk to the network‑wide average (e.g. “about 0.02 percentage points above the network average”). 


If round‑trip is enabled:

- Joint probability that at least one leg is delayed ≥ 15 minutes.

- Joint probability that at least one leg is cancelled (under an independence assumption).

6.4 “Why this risk level” explanation

Below the metrics, the app displays a bullet list explaining the main risk drivers in natural language, based on: 

- Holiday season / summer peak.

- Weekend vs weekday.

- Whether the route is historically more or less delay‑prone than average.

- Whether the airline is historically more or less delay‑prone than average.

- Congested vs quiet departure hour at the origin airport.

- Climatological weather risk (e.g. high share of bad‑weather days).

This turns the model from a “black box” into an explainable tool.

6.5 Data coverage and alternative suggestions

If there is limited history for that airline on that route:

- The app clearly states that the carrier either rarely operates the route or has few observations.

- If other airlines have stronger historical coverage for the same route, it lists them ordered by lower combined delay + cancellation risk.

If no strong non‑stop history exists, the app suggests:

- Nearby departure airports in the same state with stronger direct history to the destination.

- Nearby arrival airports in the destination state with good connectivity from the origin.

-One‑stop patterns (origin → hub → destination) that are historically well‑served.

Finally, the app offers:

- Safer departure hours for the same airline & route (hours with lower predicted delay risk).

- Alternative airlines on the same route and hour with lower predicted delay risk.

[Screenshot placeholder – Example of “Why this risk level” reasons and model‑based alternatives]


7. High‑risk hours (historical)

[Screenshot placeholder – High‑risk hours tab with line chart and heatmap]

The High‑risk hours tab is designed for exploration rather than single‑flight prediction.

7.1 Filters

- Origin airport (or “All airports”)

- Airline (or “All airlines”)

- Month (or “All months”)

- Minimum flights per hour – to ensure statistically stable estimates.

- Preset buttons help quickly select:

- Global view

- Last used origin airport

- Last used airline

7.2 Charts and tables

The tab shows:

- Delay and cancellation rates by departure hour

- A line chart of empirical delay and cancellation probabilities (0–23 hours).

- Context sentence summarising how many flights were used for this view.

- Hour × month heatmap (when month = All months)

- Heatmap where each tile is the delay rate for a (month, hour) combination.

- Darker tiles indicate higher delay risk.

- Flight volume per hour

- A bar chart showing how many flights depart in each hour, for context.

- Best/worst hours

- List of three hours with highest and lowest delay rates.

- Companion list of hours with highest cancellation rates.

This helps users identify structural patterns (e.g. very early morning vs late evening departures at a specific airport) and then feed that insight back into the Single flight tab.


8. Explore & compare

The Explore & compare tab has three subtabs: Airlines, Airports, and States. 


8.1 Airlines subtab – compare carriers on a route

[Screenshot placeholder – Airline comparison bar charts and scatter plot]

Inputs:

- Origin and destination airports.

- Month (or “All months”).

- Minimum flights per airline.

- Optional airline filter.

Outputs:

- Route‑level summary:

- Overall delay and cancellation rate on this route.

- Best airline for delay; best airline for cancellations.

- Bar charts of delay rate and cancellation rate by airline.

- Scatter plot of delay vs cancellation with point size proportional to flight volume.

- This view is intended for informed carrier selection on a specific route.

8.2 Airports subtab – compare origins or destinations

[Screenshot placeholder – Airport ranking bar chart]

Options:

- Choose whether to treat airports as origins or destinations.

- Filter by month.

- Optionally focus on flights from a specific origin or into a specific destination.

- Minimum flights per airport and number of airports to display.

- Choose to show best (lowest risk) or worst (highest risk).

Outputs:

- Ranked bar chart of airports by chosen metric (delay or cancellation rate).

- Contextual caption (e.g. “best destinations from JFK by delay rate in July”).

- This helps users and analysts identify structurally risky or reliable airports.

8.3 States subtab – state‑level recommendations

[Screenshot placeholder – State‑based recommendations text and list]

Inputs:

- Departure state.

- Destination state.

- Month (optional).

- Minimum flights per itinerary.

Outputs:

- If non‑stop state‑to‑state options have sufficient volume:

- Recommended combination of origin airport, destination airport, and airline with lowest combined risk.

- Additional strong options ranked by combined delay + cancellation risk.

If non‑stop options are scarce:

- Data‑driven one‑stop patterns: origin airport → hub → destination airport.

- Indication of leg volumes and combined risk.

- This subtab is particularly useful for high‑level planning (e.g. “best way to travel from Texas to California in March”).


9. How predictions should (and should not) be used

9.1 Intended use

- Provide probabilistic guidance on disruption risk.

Support informed choices about:

- Departure time.

- Airline.

- Choice of nearby airports.

- Whether a route is naturally disruption‑prone.

9.2 Limitations

- No real‑time weather or air traffic control information.

- Does not see operational details such as aircraft swaps, crew availability, or specific schedule changes.

- Restricted to the US domestic network and 2025 departure dates in its current version.

- Assumes that historical patterns from 2010–2024 remain broadly informative for 2025.

- Round‑trip risk assumes independence between legs, which slightly underestimates the risk when systemic disruptions affect both directions.

- A short disclaimer appears in the app reminding users that this is not a guarantee, and that operational decisions should consider additional factors.


10. Maintenance, monitoring, and future evolution

10.1 Periodic retraining

A production deployment would typically:

- Refresh BTS and weather data annually.

- Recompute metadata tables (route, airline, congestion, weather).

- Retrain and re‑calibrate the models.

- Re‑run the backtest on the following year (e.g. train to 2025, backtest on 2026).

Scripts such as train_models.py and backtest_2025_real_world.py are designed to support this cycle with minimal changes.

10.2 Monitoring

Key metrics to monitor over time:

- ROC‑AUC, PR‑AUC and Brier score on new data.

- Concentration of events in top‑risk buckets (e.g. top 10% of predicted risk).

- Drift in predicted probabilities by year, airline, and airport.

Existing training and backtest scripts already compute many of these diagnostics; they can be integrated into CI/CD or scheduled jobs.

10.3 Possible future enhancements

- Integration of near‑real‑time weather and NOTAM/ATC feeds.

- Support for international routes and additional years.

- Personalisation by user preferences (e.g. risk tolerance, maximum connections).

- A programmatic API (REST/GraphQL) on top of the current Python API.

11. Authors and acknowledgements

Authors

Abdulla Huseyinli

Ibrahim Alkali

Khalid Suliman

Kaixin Zhu

Raghuraman Thirugnana Sambandam Mythily

Acknowledgements

- US DOT/BTS for making detailed on‑time performance data publicly available.

- NOAA and related providers for historical weather data.

- The maintainers of CatBoost, LightGBM, PyTorch, and Streamlit for the open‑source tooling used throughout the project.



