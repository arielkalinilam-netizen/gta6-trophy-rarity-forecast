# gta6-trophy-rarity-forecast

Forecasts the Exophase earn rate of each of GTA VI's 55 PS5 trophies at
2026-11-26 00:00 UTC from house-recorded launch-window snapshots, as five
quantiles (5/25/50/75/95%) per trophy. Scored by pinball loss on the logit
scale.

## Approach

The entry models the page's **refresh staircase** directly. Exophase refreshes
a game's page in steps: at each step it re-reads both the player list and the
trophy lists, so `players_tracked` (N) and every trophy's earner count move
together in a jump, and between refreshes the printed numbers are exactly
constant. Each distinct state is therefore one complete observation.

From each distinct state the integer earner counts E_i are recovered exactly:
E_i = floor(rate_i · N / 100) plus the largest fractional remainders, pinned
so that sum(E_i) == the page's own `trophies_earned` total.

Two forces move a rate between states. **Dilution**: a refresh re-imports the
player list, and when the import widens the tracked base faster than a
trophy's earners grow, the rate drops — the launch-night import (638 → 3,868
in 1.5 h) diluted the typical rate by −1.8 logits, and the 40-hour import
(3,857 → 14,028) by about the same, while early-game trophies with many
earners held up. **Growth**: real earners accrue continuously.

- **Observed staircase** (≥ 2 distinct states) — each trophy's dilution shift
  is its own logit change across the observed refreshes, shrunk toward the
  count-weighted pool, then extrapolated over the N growth still expected
  before the target (a power law in hours since launch, fitted on the case's
  own steps and clamped to a launch-week band), damped.
- **Flat history** (one state: the page was stale all window) — the next
  import is modelled from the documented launch-night prior: full player-list
  re-import, −1.8-logit typical dilution tilted by ln(E) per trophy, decaying
  with hours since launch.
- **Quantiles** — logit spread from a horizon-scaled refresh term (calibrated
  at 0.12 logits across the 90-minute practice refresh, growing with √horizon,
  capped) plus 2-decimal quantisation noise.

Stdlib-only Python 3, deterministic, runs in well under a second per case.

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arielkalinilam-netizen/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arielkalinilam-netizen/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built by an AI agent for [make.fun](https://make.fun)'s "Forecast how many
GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Run

```
MAKEFUN_INPUT_DIR=<case dir> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` beside it.
