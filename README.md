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

- **N(h)** — a saturating power law in hours since launch, fitted on the
  case's own steps, floored at the last observed N.
- **E_i(h)** — each trophy's last observed count grown on a per-trophy slope
  fitted on ln E against ln(1 + h), shrunk toward the count-weighted pooled
  slope across trophies (small-count trophies carry mostly pooled information;
  a trophy with a healthy count and a clean fit keeps its own slope).
- **Quantiles** — logit-scale spread from a backtest-calibrated mixture: the
  fit's own residuals, count noise (1/sqrt(E)), a horizon term, and extra
  spread while the page is stale past its refresh cadence.

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
