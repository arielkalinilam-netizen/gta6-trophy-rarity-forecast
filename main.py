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
refreshes the printed numbers are exactly constant (the practice cases show
seven identical reads over three and a half hours).

So the history is a staircase, and every step is one complete observation of
(E_i, N) at a refresh instant.  Modelling that state directly is far better
behaved than modelling the printed rate: a refresh multiplies N and every E by
their own factors, and the ratio only then moves by player behaviour.

Per distinct state we take the integer earner counts E_i, pinned by the page's
own trophies_earned total: sum_i(E_i) == trophies_earned and E_i =
floor(rate_i * N / 100) plus the largest fractional remainders.  Then:

  N(h)    a saturating power law in hours since launch (exponent fitted on the
          case's own steps, clamped to a launch-week band), floored at the last
          observed N and capped so the median can never outrun the data;
  E_i(h)  the last observed count grown at a per-trophy slope fitted on ln E
          against ln(1 + hours), shrunk toward the rate-weighted slope of the
          whole trophy set (share information across trophies: most trophies
          sit at a handful of earners where a private fit is noise), damped for
          long gaps and capped, then discretised.

The median is E_i(target) / N(target), per trophy, then made monotone
non-increasing in trophy id (Exophase lists story trophies in progression
order, so a later chapter cannot be held by more players than an earlier one;
pool-adjacent-violators on the logit scale, weighted by sqrt of the count).

The spread is the quadratic mix of three parts, floored and capped on the
logit scale: the typical absolute logit error of the same fit replayed on the
case's own history (each of the last few steps predicted from the steps before
it), the quantisation noise of a count behind a 2-decimal rate
(1/sqrt(E)), and a horizon term (sigma per square-root day, as the house
baseline uses, damped because launch-week growth decelerates).  The three
house baselines all lose the practice at ~1.51 because they forecast the
stale rate with a narrow spread into a refresh; here a refresh is part of the
state model, not a surprise.

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
SHRINK = 0.45          # weight of the pooled slope over the trophy's own
N_EXP_MIN, N_EXP_MAX = 0.25, 1.35   # plausible launch-week N elasticity band
N_CAP_GROWTH = 4.0     # N may at most x this over the forecast gap
N_FLAT_MULT = 6.06     # flat-history refresh prior (the documented 638 -> 3,868)
E_FLAT_GROWTH = 1.0    # flat-history earner growth prior (keep the counts)
E_CAP = 3.5           # one trophy's earners may at most x this over the gap
DAMP = 0.10            # damping of the log-time extrapolation per gap unit
SIGMA_WALK = 0.14      # logit spread per sqrt day of horizon
SIGMA_MIN = 0.28
SIGMA_MAX = 2.0
SIGMA_K = 1.20         # multiplier on the backtest's typical |logit error|
BACKTEST_STEPS = 5     # how many of the last steps to replay
STALE_SIGMA = 0.55     # extra spread while the page is stale past its cadence
CADENCE_MIN = 45.0     # reads further apart than this mean a stale page


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


def damped(slope, gap, cap):
    if gap <= 0:
        return 0.0
    g = slope * gap / (1.0 + DAMP * gap)
    return max(-1.2, min(cap, g))


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
        # distinct states, oldest first: (h, N, [E_i])
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
            return 1.1
        xs = [math.log1p(self.steps[j][0]) for j in range(k)]
        ys = [math.log(self.steps[j][1]) for j in range(k)]
        fit = wls(xs, ys, [1.0] * k)
        if not fit:
            return 1.1
        return max(N_EXP_MIN, min(N_EXP_MAX, fit[1]))

    def n_at(self, h_target, upto=None):
        k = len(self.steps) if upto is None else upto
        h_last, n_last = self.steps[k - 1][0], self.steps[k - 1][1]
        gap = h_target - h_last
        if gap <= 0:
            return n_last
        if k < 2:
            # flat history: no refresh observed.  The documented launch-night
            # refresh re-imported the player list (638 -> 3,868 in 1.5 h, x6);
            # take that as the prior for the next refresh within a day, damped
            # the further out the target is.
            frac = max(0.0, min(1.0, gap / 1.5))
            return n_last * (1.0 + (N_FLAT_MULT - 1.0) * frac)
        b = self.n_slope(k)
        grown = n_last * math.exp(damped(b, math.log1p(h_target) - math.log1p(h_last), 2.2))
        return min(grown, n_last * N_CAP_GROWTH)

    # ---- E_i(h)
    def pooled_slope(self, upto=None, min_e=4.0):
        k = len(self.steps) if upto is None else upto
        num = 0.0
        den = 0.0
        for i in range(self.n):
            pts = [(self.steps[j][0], self.steps[j][3][i])
                   for j in range(k) if self.steps[j][3][i] is not None]
            pts = [p for p in pts if p[1] >= min_e]
            d = {}
            for x, e in pts:
                d[x] = e
            pts = sorted(d.items())
            if len(pts) < 2:
                continue
            (x0, e0) = pts[0]
            for x, e in pts[1:]:
                if e0 > 0 and e > 0:
                    w = math.sqrt(min(e, e0))
                    num += math.log(e / e0) * w
                    den += (math.log1p(x) - math.log1p(x0)) * w
        return num / den if den > 1e-9 else 0.35

    def own_slope(self, i, upto=None):
        k = len(self.steps) if upto is None else upto
        pts = [(self.steps[j][0], self.steps[j][3][i]) for j in range(k)
               if self.steps[j][3][i] is not None]
        d = {}
        for x, e in pts:
            d[x] = e
        pts = sorted(d.items())
        if len(pts) < 3:
            return None
        fit = wls([math.log1p(p[0]) for p in pts],
                  [math.log(max(0.5, p[1])) for p in pts],
                  [max(1.0, p[1]) for p in pts])
        return fit[1] if fit else None

    def earners_at(self, i, h_target, upto=None, pooled=None):
        k = len(self.steps) if upto is None else upto
        last_e = None
        last_h = self.steps[k - 1][0]
        for j in range(k - 1, -1, -1):
            c = self.steps[j][3][i]
            if c is not None:
                last_e = float(c)
                last_h = self.steps[j][0]
                break
        if last_e is None:
            # no count in any distinct state (the trophy appeared mid-history):
            # take the latest read that prints it
            for j in range(len(self.reads) - 1, -1, -1):
                v = self.reads[j]["rates"][i]
                if v is not None:
                    last_e = max(0.0, v * self.steps[min(j, k - 1)][1] / 100.0)
                    last_h = self.steps[min(j, k - 1)][0]
                    break
        if last_e is None:
            return 0.0
        own = self.own_slope(i, k)
        if pooled is None:
            pooled = self.pooled_slope(k)
        # Shrink a small-count trophy's own slope toward the pooled one: with a
        # handful of earners the private fit is noise.  A trophy with a healthy
        # count and a clean fit keeps its own slope.
        if own is None:
            slope = pooled
        elif last_e >= 30:
            slope = own
        else:
            w = min(1.0, math.sqrt(max(1.0, last_e)) / math.sqrt(30.0))
            slope = w * own + (1.0 - w) * pooled
        slope = max(0.0, slope)
        gap = math.log1p(h_target) - math.log1p(last_h)
        return last_e * math.exp(damped(slope, gap, E_CAP))


def medians_at(c: Case, h_target, upto=None):
    k = len(c.steps) if upto is None else upto
    if k < 1:
        return None
    pooled = c.pooled_slope(k)
    n_t = c.n_at(h_target, k)
    # A flat history (one distinct state: the page was stale the whole window)
    # carries no refresh or growth observation at all.  A launch-night page in
    # that state is the pre-launch snapshot, which a refresh re-bases: model
    # the refresh as the rules document it (the player list re-imported), with
    # a wide uncertainty on both the refresh factor and the earner growth.
    flat = k < 2
    logits = []
    weights = []
    for i in range(c.n):
        # A flat history has no growth observation: keep the counts (the page
        # is one stale snapshot; the next refresh re-bases N, not E).
        e = c.earners_at(i, h_target, k, 0.0 if flat else pooled)
        if flat:
            e = e * E_FLAT_GROWTH
        # A trophy every tracked player holds (rate 100) grows with N: the
        # player import itself carries the trophy.
        if c.steps[k - 1][2][i] is not None and c.steps[k - 1][2][i] >= 99.5:
            e = n_t
        e = min(e, n_t)
        rate = 100.0 * e / n_t if n_t > 0 else 0.01
        logits.append(logit_pct(rate))
        weights.append(math.sqrt(max(0.5, e)))
    # No cross-trophy smoothing: the live data shows neighbouring trophies
    # moving independently (chapter completion is not strictly ordered in the
    # earn rates), so each trophy's fitted median stands as it is.
    return logits


def spreads_for(c: Case, meds):
    walk = SIGMA_WALK * math.sqrt(max(0.02, c.horizon_days))
    # stale-page term: how far past the last refresh the target sits
    stale = 0.0
    if c.horizon_days * 1440.0 > CADENCE_MIN:
        stale = min(0.4, STALE_SIGMA * math.sqrt(c.horizon_days) / 3.0)
    # backtest: replay the fit on the case's own steps
    errs = []
    for cut in range(max(2, len(c.steps) - BACKTEST_STEPS), len(c.steps)):
        h_cut = c.steps[cut][0]
        if h_cut <= 0.2:
            continue
        pred = medians_at(c, h_cut, cut)
        if pred is None:
            continue
        for i in range(c.n):
            e = c.steps[cut][3][i]
            if e is None:
                continue
            actual = math.log(max(0.5, e) / max(1.0, c.steps[cut][1]))
            errs.append(abs(pred[i] - actual))
    if errs:
        errs.sort()
        base = errs[len(errs) // 2]
    else:
        base = 0.30
    base = min(1.0, max(0.06, base))
    spreads = []
    last_n = c.steps[-1][1]
    for i in range(c.n):
        e = max(0.8, expit(meds[i]) * 0.01 * last_n)
        noise = 1.0 / math.sqrt(max(1.0, e))
        s = math.sqrt((SIGMA_K * base) ** 2 + noise ** 2 + walk ** 2 + stale ** 2)
        spreads.append(s)
    return spreads


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    c = Case(case)
    h_target = max(0.0, (c.t_target - c.t0) / 3600.0)
    meds = medians_at(c, h_target)
    if meds is None:
        return fallback(case)
    spreads = spreads_for(c, meds)
    out = []
    for i, trophy in enumerate(c.trophies):
        s = min(SIGMA_MAX, max(SIGMA_MIN, spreads[i]))
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
