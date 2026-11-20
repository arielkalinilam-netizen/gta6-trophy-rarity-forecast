"""GTA VI PS5 trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every <id>.in in the input directory (one case per run: the grader leaves
exactly one there), and writes <id>.out with five quantiles (5/25/50/75/95%)
of each trophy's Exophase earn rate at the case's target.

## The model

A rate is round(100 * E / N, 2): E the tracked players who hold the trophy, N
the players the page tracks.  Exophase does not update on a schedule; it
refreshes a game's page in steps, and at each step it re-reads both the player
list and the trophy lists, so N and every E move together in a jump.  A read
therefore shows the state as of the last refresh before it, and between
refreshes the printed numbers are exactly constant (six and a half hours of
live reads on 19-20 Nov show every rate vector identical across a refresh that
moved N from 14,882 to 14,028: the re-import re-based both sides of the ratio).

So the history is a staircase, and every step is one complete observation of
(E_i, N) at a refresh instant.  Two things move a rate between states:

  dilution   a refresh re-imports the player list; when the import widens the
             tracked base faster than any trophy's earners grow, every rate
             drops.  The launch-night import (638 -> 3,868 in 1.5 h) diluted
             the typical rate by -1.8 logits, and the 40-hour import (3,857 ->
             14,028) by about the same, while a few early-game trophies with
             many earners held up: the per-trophy shift tilts by ln(E).
  growth     real earners accrue continuously between refreshes.

Per distinct state we take the integer earner counts E_i, pinned by the page's
own trophies_earned total: sum_i(E_i) == trophies_earned and E_i =
floor(rate_i * N / 100) plus the largest fractional remainders.  Then:

  * Observed staircase (>= 2 distinct states): each trophy's dilution shift is
    its own logit change across the observed refreshes, shrunk toward the
    pool's (a handful of earners makes a private fit noise), and extrapolated
    over the N growth still expected before the target, damped.  N(h) follows
    a power law in hours since launch, fitted on the case's own steps and
    clamped to a launch-week band.
  * Flat history (one state: the page was stale the whole window, as on launch
    night): no refresh has been observed, so the next one is modelled from the
    documented prior - the player list re-imports (x6) and the typical rate
    dilutes by -1.8 logits, tilted per trophy by ln(E) at x0.4 per ln above
    the median earner count (early-game trophies track the base).  The shift
    scales with the horizon's share of the refresh cadence.
  * A trophy every tracked player holds (rate >= 99.5%) is the platinum: it
    stays at 100.

The spread is the quadratic mix of a horizon-scaled refresh term (calibrated
at 0.12 logits across the 90-minute practice refresh and growing with the
square root of the horizon, capped) and the quantisation noise of a count
behind a 2-decimal rate (1/sqrt(E), capped).  The three house baselines all
lose the practice at ~1.51 because they forecast the stale rate with a narrow
spread into a refresh; here a refresh is part of the state model, not a
surprise.

Deterministic: no randomness, no network, standard library only.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

T0 = "2026-11-19T00:00:00Z"   # launch instant (the game released 19 Nov 2026)

# ------------------------------------------------------------------ tuning
N_EXP_MIN, N_EXP_MAX = 0.5, 0.9    # plausible launch-week N elasticity band
N_CAP_GROWTH = 4.0                 # N may at most x this over the forecast gap
DILUTE_FLAT = -1.80                # flat-history dilution per full import (x6)
DILUTE_TILT = 0.40                 # per ln(E) above the median earner count
DILUTE_MED_E = 1.5                 # the median trophy's ln(E) on launch night
FLAT_HORIZON_DAYS = 1.5            # a full import is expected within this span
EXTRAP_DAMP = 0.60                 # damping of the observed dilution extrapolation
SHIFT_UP_CAP = 1.2                 # the median may not outrun the last rate by more
SHRINK_E = 30.0                    # earner count at which a private shift is trusted
SIGMA_BASE = 0.12                  # refresh spread at the 90-minute practice horizon
SIGMA_HORIZON_D = 0.0625           # the horizon sigma was calibrated on (90 min)
SIGMA_MAX = 0.42


def clamp(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def expit(l):
    if l > 30.0:
        l = 30.0
    elif l < -30.0:
        l = -30.0
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_ts(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    frac = 0.0
    if "." in s:
        s, fs = s.split(".", 1)
        frac = float("0." + fs) if fs else 0.0
    date, clock = s.split("T")
    y, mo, d = (int(v) for v in date.split("-"))
    parts = [int(v) for v in clock.split(":")]
    while len(parts) < 3:
        parts.append(0)
    yy = y - (1 if mo <= 2 else 0)
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    doy = (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2] + frac


# ------------------------------------------------------------------ counts

def recover_counts(rates, n_players, total):
    """Integer earner counts of one read: floor(rate*N/100) plus the largest
    fractional remainders, so the counts sum to the page's own total."""
    idx = [i for i, r in enumerate(rates) if r is not None]
    if not idx:
        return [None] * len(rates)
    exact = {i: rates[i] * n_players / 100.0 for i in idx}
    base = {i: math.floor(exact[i] + 1e-9) for i in idx}
    rem = int(total) - sum(base[i] for i in idx)
    counts = [None] * len(rates)
    if rem < 0 or rem > len(idx) * 4:
        for i in idx:
            counts[i] = max(0, int(round(exact[i])))
        return counts
    order = sorted(idx, key=lambda i: -(exact[i] - base[i]))
    for k in range(rem):
        base[order[k % len(order)]] += 1
    for i in idx:
        counts[i] = base[i]
    return counts


# ------------------------------------------------------------------ fitting

def wls(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    return a, b


# ------------------------------------------------------------------ model

class Case:
    def __init__(self, case):
        self.case = case
        self.trophies = case["trophies"]
        self.n = len(self.trophies)
        self.reads = case["history"]
        self.t0 = parse_ts(T0)
        self.t_target = parse_ts(case["target"])
        self.t_last = parse_ts(self.reads[-1]["slot"])
        self.horizon_days = case["horizon_minutes"] / 1440.0
        # distinct states, oldest first: (h, N, rates, [E_i])
        self.steps = []
        for r in self.reads:
            h = max(0.0, (parse_ts(r["slot"]) - self.t0) / 3600.0)
            if self.steps and self.steps[-1][1] == r["players_tracked"] and \
                    self.steps[-1][2] == r["rates"]:
                continue
            E = recover_counts(r["rates"], float(r["players_tracked"]),
                               r.get("trophies_earned") or 0)
            self.steps.append((h, float(r["players_tracked"]), r["rates"], E))
        if not self.steps:
            raise ValueError("no usable reads")

    # ---- N(h)
    def n_slope(self, upto=None):
        k = len(self.steps) if upto is None else upto
        if k < 2:
            return 0.7
        xs = [math.log1p(self.steps[j][0]) for j in range(k)]
        ys = [math.log(self.steps[j][1]) for j in range(k)]
        fit = wls(xs, ys, [1.0] * k)
        if not fit:
            return 0.7
        return max(N_EXP_MIN, min(N_EXP_MAX, fit[1]))

    def n_growth_factor(self, upto=None):
        """Expected N(h_target) / N(last read), floored at 1 and capped."""
        k = len(self.steps) if upto is None else upto
        h_last, n_last = self.steps[k - 1][0], self.steps[k - 1][1]
        h_target = max(0.0, (self.t_target - self.t0) / 3600.0)
        gap = h_target - h_last
        if gap <= 0:
            return 1.0
        b = self.n_slope(k)
        factor = math.exp(b * (math.log1p(h_target) - math.log1p(h_last)))
        return max(1.0, min(N_CAP_GROWTH, factor))

    def last_counts(self, i):
        """The trophy's most recent earner count, 0.0 if it never printed."""
        for j in range(len(self.steps) - 1, -1, -1):
            c = self.steps[j][3][i]
            if c is not None:
                return float(c)
        for j in range(len(self.reads) - 1, -1, -1):
            v = self.reads[j]["rates"][i]
            if v is not None:
                return max(0.0, v * self.steps[min(j, len(self.steps) - 1)][1] / 100.0)
        return 0.0


def observed_shifts(c: Case):
    """Per-trophy logit shift across the observed refreshes (shrunk toward the
    pool), plus the N factor those refreshes spanned.  Returns (None, 1.0)
    when the history holds a single distinct state."""
    k = len(c.steps)
    if k < 2:
        return None, 1.0
    first, last = c.steps[0], c.steps[k - 1]
    n_factor = last[1] / first[1] if first[1] > 0 else 1.0
    # pooled shift, weighted by the earner count each shift rests on
    num = den = 0.0
    for i in range(c.n):
        if first[2][i] is None or last[2][i] is None:
            continue
        d = logit_pct(last[2][i]) - logit_pct(first[2][i])
        e = max(1.0, last[3][i] if last[3][i] is not None else 1.0)
        w = math.sqrt(min(e, SHRINK_E))
        num += d * w
        den += w
    pool = num / den if den > 0 else DILUTE_FLAT
    shifts = []
    for i in range(c.n):
        if first[2][i] is None or last[2][i] is None:
            shifts.append(pool)
            continue
        d = logit_pct(last[2][i]) - logit_pct(first[2][i])
        e = max(1.0, last[3][i] if last[3][i] is not None else 1.0)
        w = min(1.0, math.sqrt(e) / math.sqrt(SHRINK_E))
        shifts.append(w * d + (1.0 - w) * pool)
    return shifts, n_factor


def dilution_shifts(c: Case):
    """Per-trophy logit shift to apply between the last read and the target."""
    shifts, n_seen = observed_shifts(c)
    if shifts is None:
        # Flat history: no refresh observed.  Model the next import from the
        # documented launch-night prior, tilted per trophy by ln(E), scaled by
        # how much of a refresh cadence the horizon spans.
        # A 90-minute practice horizon holds a full import on launch night
        # (638 -> 3,868 between 04:00 and 04:30): the pre-launch snapshot was
        # re-based onto the real player base, diluting every rate -1.8 logits
        # in full.  Later imports are smaller relative to a mature base, so
        # the prior decays with hours since launch, and only the horizon's
        # share of a cadence day applies.
        h_last = c.steps[-1][0]
        if h_last < 12.0:
            # launch window: an import is imminent and huge (the practice's
            # 90-minute horizon held a full x6 re-base)
            frac = 1.0
        else:
            decay = 12.0 / (h_last + 12.0)
            frac = decay * min(1.0, c.horizon_days / 1.0)
        out = []
        for i in range(c.n):
            e = max(1.0, c.last_counts(i))
            tilt = DILUTE_TILT * (math.log(e) - DILUTE_MED_E)
            out.append((DILUTE_FLAT + tilt) * frac)
        return out
    # Observed staircase: extrapolate the shifts over the N growth still
    # expected before the target, damped.
    n_future = c.n_growth_factor()
    if n_seen > 1.01:
        scale = EXTRAP_DAMP * math.log(n_future) / math.log(n_seen)
    else:
        # refreshes observed but N unchanged (a ratio-invariant re-read): no
        # dilution evidence; fall back to a hard-damped flat prior.
        scale = 0.25 * min(1.0, c.horizon_days / FLAT_HORIZON_DAYS)
    scale = max(-1.5, min(1.5, scale))
    return [s * scale for s in shifts]


def medians_at(c: Case):
    k = len(c.steps)
    shifts = dilution_shifts(c)
    logits = []
    for i in range(c.n):
        r_last = None
        for j in range(k - 1, -1, -1):
            if c.steps[j][2][i] is not None:
                r_last = c.steps[j][2][i]
                break
        if r_last is None:
            logits.append(logit_pct(0.01))
            continue
        if r_last >= 99.5:
            # the platinum: every tracked player holds it
            logits.append(logit_pct(100.0))
            continue
        lg = logit_pct(r_last) + shifts[i]
        # the median cannot outrun the last printed rate by much, so a noisy
        # private shift cannot explode upward
        lg = min(lg, logit_pct(r_last) + SHIFT_UP_CAP)
        logits.append(lg)
    return logits


def spreads_for(c: Case, meds):
    # refresh uncertainty grows with the square root of the horizon,
    # calibrated at the 90-minute practice refresh
    s_h = SIGMA_BASE * math.sqrt(max(SIGMA_HORIZON_D, c.horizon_days) / SIGMA_HORIZON_D)
    s_h = min(SIGMA_MAX, s_h)
    last_n = c.steps[-1][1]
    spreads = []
    for i in range(c.n):
        # quantisation of a 2-decimal rate: negligible next to the refresh
        # term, keep a small floor so tiny rates still get a little room
        p = expit(meds[i]) / 100.0
        noise = 0.005 * 100.0 / max(1e-4, p * (100.0 - p))
        spreads.append(math.sqrt(s_h ** 2 + min(0.05, noise) ** 2))
    return spreads


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    c = Case(case)
    meds = medians_at(c)
    spreads = spreads_for(c, meds)
    out = []
    for i, trophy in enumerate(c.trophies):
        s = min(SIGMA_MAX, max(0.06, spreads[i]))
        qs = [round4(expit(meds[i] + z * s)) for z in Z]
        out.append({"id": trophy["id"], "q": qs})
    return {"trophies": out}


def fallback(case):
    last = case["history"][-1] if case["history"] else None
    d = max(0.02, case.get("horizon_minutes", 60) / 1440.0)
    s = 0.5 * math.sqrt(d)
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i] if last else None
        l = logit_pct(r if r is not None else 0.01)
        rows.append({"id": t["id"], "q": [round4(expit(l + z * s)) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            out = forecast_case(case, path.name)
        except Exception:
            out = fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
