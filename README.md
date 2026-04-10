# FastF1-Pit-Strategy-Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> Formula 1 pit strategy optimization using **probabilistic decision models**, tire degradation modeling, and real telemetry data via the FastF1 library. A concrete application of sequential decision-making under uncertainty — with direct analogues in logistics and supply chain optimization.

---

## Overview

A Formula 1 race is a 90-minute constrained optimization problem played out in real time. The primary decision variable is pit stop timing: each stop costs roughly 20-25 seconds in stationary time, but the alternative — running an increasingly degraded tire — costs time continuously through slower lap speeds and heightened DNF risk. The optimal strategy is rarely obvious and depends on dozens of interacting variables.

This project builds a quantitative framework for pit strategy decisions using actual F1 race telemetry, probabilistic tire degradation models, and sequential decision-making under uncertainty.

---

## What is FastF1?

[FastF1](https://github.com/theOehrly/Fast-F1) is an open-source Python library that provides structured access to Formula 1 timing and telemetry data via the official F1 data API. It surfaces:

- **Lap-by-lap timing data** — sector times, lap times, gap to leader for every driver
- **Tire compound and age** — which compound each driver is running and how many laps it has been on
- **Car telemetry** — throttle, brake, speed, gear, DRS, RPM at 10Hz (data stream, not summary)
- **Weather data** — track and air temperature, rainfall, wind
- **Pit stop events** — timestamps, durations, compound changes

FastF1 covers all sessions from 2018 onward with near-complete telemetry, making it one of the richest freely available time series datasets in motorsport.

---

## The Optimization Problem

### Decision Variables

- **Pit stop timing** — which lap to stop on
- **Compound selection** — which tire compound to fit at each stop
- **Number of stops** — one-stop, two-stop, or three-stop strategy

### Key Uncertainties

- **Tire degradation rate** — varies by track surface, temperature, and driving style; must be estimated in real time
- **Safety Car / Virtual Safety Car probability** — a VSC makes pit stops much cheaper; conditional strategy switching
- **Opponent strategy** — a "reactive" stop to cover an opponent's undercut requires real-time decision-making
- **Weather changes** — rain events that force an intermediate/wet tire switch

### Objective Function

Minimize total race time, which decomposes as:

```
Race Time = Σ(clean lap times given compound/age) + Σ(pit stop durations) + DNF risk penalty
```

The first term is a function of tire degradation, the second is measured directly, and the third is a tail-risk term that grows with tire age past the cliff point.

---

## Technical Approach

### Tire Degradation Model

Tire performance degrades non-linearly with age. The model fits a degradation curve per compound per track:

```
Δlap_time(age, compound) = α·age + β·age² + ε(track_temp, fuel_load)
```

The quadratic term captures the "cliff" — the point past which degradation accelerates sharply. Parameters are estimated from historical FastF1 data for each (circuit, compound) pair.

### Strategy Evaluation via Dynamic Programming

The optimal stopping problem can be framed as a finite-horizon Markov Decision Process:

```
State:  (lap, tire_compound, tire_age, position, gap_to_front)
Action: {continue, pit_soft, pit_medium, pit_hard}
Reward: -(lap_time + pit_cost·1[pit_action])
```

The Bellman equation gives the optimal value function:

```
V*(s) = max_a [ R(s,a) + E[V*(s'|s,a)] ]
```

For a 60-lap race with discretized state spaces, this is computationally tractable via backward induction. The resulting policy maps any race state to an optimal action.

### Safety Car Probability Model

A logistic regression model estimates Safety Car probability on each lap, conditioned on:
- Lap number (accident rates are higher at race start and certain circuit sectors)
- Track temperature (tire failure risk)
- Rainfall probability (weather feed)
- Current field spread (tightly packed cars have higher contact risk)

The MDP value function is computed both with and without a Safety Car event per lap, and the final policy is an expectation over these outcomes weighted by the SC probability.

---

## Key Results

The model evaluates strategy options against historical outcomes using post-race FastF1 data:

- For each race analyzed, the model generates a ranked list of strategies and their expected lap time costs
- Comparison against the strategy actually chosen by each team reveals where teams were sub-optimal and where they correctly identified the optimal window
- The model's "optimal" strategy is validated against the actual fastest race completion time for that event

*Specific numerical results depend on the circuit and year analyzed; see individual race notebooks for detailed outputs.*

---

## Usage

```python
from fastf1_strategy import RaceAnalyzer, StrategyOptimizer

# Load race data (FastF1 handles caching)
analyzer = RaceAnalyzer(year=2023, circuit="Monza", session="Race")
analyzer.load()

# Fit tire degradation model from historical data
degradation = analyzer.fit_degradation_model(
    compounds=["SOFT", "MEDIUM", "HARD"],
    laps_data=analyzer.laps
)

# Run strategy optimization
optimizer = StrategyOptimizer(
    race_laps=53,
    degradation_model=degradation,
    pit_cost_seconds=22.0,
    safety_car_model=analyzer.sc_probability_model
)

# Get optimal strategy for a given starting compound
policy = optimizer.solve(initial_compound="MEDIUM")
print(f"Optimal strategy: {policy.summary()}")
# → Optimal strategy: MEDIUM(27) → pit → HARD(26), expected cost: +0.0s

# Compare against all team strategies
comparison = analyzer.compare_strategies(policy)
comparison.plot()
```

---

## Getting Started

```bash
git clone https://github.com/A-Kuo/FastF1-Pit-Strategy-Optimization.git
cd FastF1-Pit-Strategy-Optimization

pip install -r requirements.txt

# FastF1 will download and cache race data on first run
# Default cache directory: ~/fastf1_cache/
python analyze.py --year 2023 --circuit monza
```

---

## Business Analogues

The techniques here are directly applicable to high-stakes resource allocation decisions in logistics and operations:

| F1 Problem | Business Analogue |
|-----------|-------------------|
| Tire compound selection | Fleet maintenance scheduling — when to service vs. run degraded |
| Pit timing window | Inventory replenishment timing under demand uncertainty |
| Safety Car = unexpected cheap opportunity | Demand surge = unexpected cheap opportunity to restock |
| Opponent undercut | Competitor pricing move requiring reactive response |
| Degradation cliff | Equipment failure probability curves in predictive maintenance |

The shared structure is: **sequential decisions under uncertainty with a time-varying cost function and a finite horizon**. Markov Decision Processes are the natural framework for all of these.

---

## Related Work

- **[Financial-Economic-Ticker-Analyzer-Agent](https://github.com/A-Kuo/Financial-Economic-Ticker-Analyzer-Agent)** — Sequential decision-making applied to financial data
- **[Tax-Data-System-with-S4-Architecture](https://github.com/A-Kuo/Tax-Data-System-with-S4-Architecture)** — Temporal sequence modeling for long-horizon prediction problems
- **[Language-Model-Hallucination-Detection-via-Entropy-Divergence](https://github.com/A-Kuo/Language-Model-Hallucination-Detection-via-Entropy-Divergence)** — Uncertainty quantification; entropy methods for decision confidence

---

## References

- FastF1 library: [github.com/theOehrly/Fast-F1](https://github.com/theOehrly/Fast-F1)
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming.* Wiley.
- Heilmeier, A., et al. (2020). "Minimum Race Time Planning with Virtual Strategy Engineer for Formula 1." *SAE Technical Paper*.

---

*Last updated: April 2026*
