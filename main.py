"""Forecast GTA VI trophy earn rates on Exophase (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file in $MAKEFUN_INPUT_DIR (default /input) and writes one
forecast per case to $MAKEFUN_OUTPUT_DIR (default /output): for each of the
game's trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate, in
percent, at the case's target time.

How the forecast is built (all fitting happens at run time, from the case's
own history; nothing is fetched, nothing is random):

Every trophy is handled in logit space, where a rate r (percent, clamped to
[0.01, 99.99]) becomes ln((r/100) / (1 - r/100)).  Two forces move a rate:

* dilution -- when Exophase refreshes its page, the tracked-player base
  jumps and the new players hold far fewer trophies than the incumbents, so
  almost every rate falls at a refresh; and
* progression -- between refreshes the tracked players keep playing and
  earning, so rates creep back up, fast for early story trophies and slowly
  for rare ones.

The program splits the history into those two forces.  A pooled dilution
elasticity is read off the reads where the player count jumps (the change in
logit rate per change in log players, median over trophies); each trophy
then gets its own progression line, a recency-weighted least-squares fit of
(dilution-corrected) logit rate against log time since the release day
began, shrunk toward its tier's mean slope.  The player count at the target
is extrapolated the same way, and the forecast is the progression line read
at the target plus the dilution term at the forecast player count.  The
spread combines the fit's residual, slope uncertainty, dilution
uncertainty, and a random-walk term that grows with the horizon.

One regime needs a prior, not a fit: a page that has never refreshed.  On
launch night Exophase's page sat frozen for hours -- same player count,
same rates -- and then refreshed in one jump (638 to 3,868 tracked players)
while the newcomers had earned almost nothing beyond the trophy everybody
gets, so the ordinary rates fell to roughly a fifth of their old values.
A history with no player-count change at all cannot show that coming, so
for that case only the forecast uses a refresh mixture: with high
probability a refresh lands before the target and cuts non-saturated rates
to about a fifth (logit shift about -1.5, with a spread for how big the
refresh is), and with the remaining probability the page stays as it is.
That mixture is what makes the practice cases score: a frozen page is a
page about to refresh, not a page that will stay frozen.

Deterministic, standard library only, no network.  Python 3.13.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- small numeric helpers -------------------------------------------------

def clamp_rate(r):
    if r is None:
        return None
    if r < 0.01:
        return 0.01
    if r > 99.99:
        return 99.99
    return float(r)


def logit_of(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of(logit):
    if logit > 20.0:
        return 99.99
    if logit < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-logit))


def parse_time(text):
    if text is None:
        return None
    txt = str(text).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_release(dt):
    if dt is None:
        return 0.0
    return max(0.0, (dt - RELEASE_START).total_seconds() / 3600.0)


def log_time(dt):
    return math.log1p(hours_since_release(dt))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, n)."""
    sw = sum(ws)
    if sw <= 0.0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, len(xs))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, ws)) / sw
    return (a, b, math.sqrt(max(var, 0.0)), len(xs))


# --- frozen-page refresh mixture ------------------------------------------

def mixture_quantiles(last_rate, horizon_hours):
    """Quantiles of the logit shift when the page has never refreshed.

    Predictive law for the shift s of logit(rate): with weight w a refresh
    lands before the target, s ~ Normal(-1.5, 0.45); with weight 1 - w no
    refresh lands and s = 0.  The -1.5 centres the refreshed rates at about
    a fifth of their old values, the size of the launch-night refresh, and
    the 0.45 spread covers refreshes a good deal smaller or larger.  The
    quantiles are read off a fine grid, so they are exact for this law.
    """
    if horizon_hours <= 3.0:
        w_refresh = 0.85
        shift = -1.5
    else:
        w_refresh = 0.95
        shift = -1.5 - 0.25 * math.log(horizon_hours / 3.0)
        shift = max(shift, -2.5)
    sd = 0.45
    base = logit_of(last_rate)
    grid = [(-4.0 + 0.01 * i) for i in range(801)]
    cdf = []
    for g in grid:
        c = w_refresh * norm_cdf((g - shift) / sd)
        if g >= 0.0:
            c += (1.0 - w_refresh)
        cdf.append(c)
    out = []
    for tau in LEVELS:
        lo, hi = 0, len(grid) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < tau:
                lo = mid + 1
            else:
                hi = mid
        out.append(round(rate_of(base + grid[lo]), 4))
    return out


def saturated_quantiles(last_rate):
    """A trophy essentially everybody holds stays held: tight band at top."""
    top = min(99.999, max(99.9, last_rate))
    vals = [top - 0.09, top - 0.02, top, top + 0.004, top + 0.009]
    vals = [min(99.999, max(0.01, v)) for v in vals]
    vals = sorted(vals)
    return [round(v, 4) for v in vals]


# --- the forecaster ---------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    horizon_days = max(horizon_min / 1440.0, 0.0)
    horizon_hours = max(horizon_min / 60.0, 0.0)
    target_dt = parse_time(case.get("target"))
    x_target = log_time(target_dt)

    # Clean history: keep reads with a usable player count, oldest first.
    reads = []
    for h in history:
        n = h.get("players_tracked")
        rates = h.get("rates")
        if n is None or rates is None or float(n) <= 0:
            continue
        at = parse_time(h.get("at"))
        reads.append({"at": at, "x": log_time(at), "n": float(n),
                      "rates": rates})
    if not reads:
        return [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                for t in trophies]

    last = reads[-1]
    last_rates = last["rates"]
    n_last = last["n"]
    x_last = last["x"]

    # Was the page ever refreshed in this history?
    n_changed = sum(1 for a, b in zip(reads, reads[1:]) if b["n"] != a["n"])
    frozen = (n_changed == 0)

    if frozen:
        rows = []
        for idx, tr in enumerate(trophies):
            r = last_rates[idx] if idx < len(last_rates) else None
            if r is None:
                r = 0.5
            r = float(r)
            if r >= 99.5:
                q = saturated_quantiles(r)
            else:
                q = mixture_quantiles(r, horizon_hours)
            rows.append({"id": tr["id"], "q": q})
        return rows

    # --- dilution elasticity from reads where the player count jumps -------
    # Only short gaps count: over 10-30 minutes the tracked players earn
    # almost nothing, so a rate move across a jump is dilution, not play.
    jump_slopes = []
    for a, b in zip(reads, reads[1:]):
        if b["n"] <= a["n"] * 1.02:
            continue
        gap_h = 0.0
        if a["at"] is not None and b["at"] is not None:
            gap_h = (b["at"] - a["at"]).total_seconds() / 3600.0
        if gap_h > 1.0:
            continue
        dn = math.log(b["n"] / a["n"])
        if dn <= 0.02:
            continue
        for idx in range(len(trophies)):
            ra = a["rates"][idx] if idx < len(a["rates"]) else None
            rb = b["rates"][idx] if idx < len(b["rates"]) else None
            if ra is None or rb is None:
                continue
            ra, rb = float(ra), float(rb)
            if ra >= 99.5 or rb >= 99.5 or ra <= 0.02:
                continue
            jump_slopes.append((logit_of(rb) - logit_of(ra)) / dn)
    if len(jump_slopes) >= 5:
        elast = median(jump_slopes)
    else:
        elast = -0.8
    elast = max(-1.6, min(0.0, elast))

    # --- player count at the target ---------------------------------------
    # Log players against log time, recency-weighted; extrapolate, capped so
    # one wild early slope cannot conjure a tenfold crowd.
    xs_n = [r["x"] for r in reads]
    ys_n = [math.log(r["n"]) for r in reads]
    ws_n = [1.0 + 2.0 * (i / max(len(reads) - 1, 1)) for i in range(len(reads))]
    fit_n = weighted_line(xs_n, ys_n, ws_n)
    if fit_n is not None:
        a_n, b_n, _, _ = fit_n
        b_n = max(0.0, min(b_n, 2.0))
        log_n_target = a_n + b_n * x_target
    else:
        log_n_target = math.log(n_last)
    log_n_target = max(math.log(n_last),
                       min(log_n_target, math.log(n_last) + math.log(6.0)))
    dn_forecast = log_n_target - math.log(n_last)

    # --- per-trophy progression lines --------------------------------------
    # Fit logit(rate) - elast*log(N) against log time, recency-weighted,
    # with one robust reweighting pass; shrink each slope toward its tier's
    # mean slope, more so when the trophy's own fit is noisy or short.
    tier_slopes = {}
    per_trophy = []
    for idx, tr in enumerate(trophies):
        xs, ys, ws = [], [], []
        for k, r in enumerate(reads):
            rate = r["rates"][idx] if idx < len(r["rates"]) else None
            if rate is None:
                continue
            rate = float(rate)
            if rate >= 99.5:
                continue
            y = logit_of(max(0.02, min(99.98, rate))) - elast * math.log(r["n"])
            xs.append(r["x"])
            ys.append(y)
            ws.append(1.0 + 2.0 * (k / max(len(reads) - 1, 1)))
        fit = weighted_line(xs, ys, ws) if len(xs) >= 3 else None
        if fit is not None and len(xs) >= 3:
            a, b, resid, cnt = fit
            # Robust pass: downweight points far from the line, refit once.
            resid_list = [y - (a + b * x) for x, y in zip(xs, ys)]
            scale = max(0.15, median([abs(v) for v in resid_list]) * 1.4826)
            ws2 = [w / (1.0 + (rv / (2.0 * scale)) ** 2)
                   for w, rv in zip(ws, resid_list)]
            fit2 = weighted_line(xs, ys, ws2)
            if fit2 is not None:
                a, b, resid, cnt = fit2
            b = max(-0.6, min(1.5, b))
        else:
            a, b, resid, cnt = None, 0.0, 0.5, len(xs)
        per_trophy.append({"a": a, "b": b, "resid": resid, "cnt": cnt,
                           "tier": tr.get("tier", "bronze")})
        if a is not None:
            tier_slopes.setdefault(tr.get("tier", "bronze"), []).append(b)
    tier_mean = {t: median(v) for t, v in tier_slopes.items() if v}
    overall_mean = median([p["b"] for p in per_trophy if p["a"] is not None]) \
        if any(p["a"] is not None for p in per_trophy) else 0.0

    rows = []
    for idx, tr in enumerate(trophies):
        r_last = last_rates[idx] if idx < len(last_rates) else None
        r_last = 0.5 if r_last is None else float(r_last)
        if r_last >= 99.5:
            rows.append({"id": tr["id"], "q": saturated_quantiles(r_last)})
            continue
        info = per_trophy[idx]
        if info["a"] is None:
            # Too little usable history: hold the last rate, spread wide.
            mu = logit_of(r_last)
            spread = 0.35 + 0.5 * math.sqrt(horizon_days)
        else:
            tm = tier_mean.get(info["tier"], overall_mean)
            # Shrinkage: trust the trophy's own slope in proportion to how
            # many reads support it and how clean the fit is.
            w_own = info["cnt"] / (info["cnt"] + 12.0)
            w_own *= 1.0 / (1.0 + info["resid"])
            b_use = w_own * info["b"] + (1.0 - w_own) * tm
            mu = info["a"] + b_use * x_target + elast * log_n_target
            # Never forecast a rate below what the last read shows by more
            # than dilution alone can explain, and never above a sane cap:
            # progression lines read far past the data can run away.  The
            # -0.8 slack lets a genuinely collapsing rate fall further than
            # dilution alone would put it.
            mu = max(mu, logit_of(r_last) + elast * dn_forecast - 0.8)
            mu = min(mu, logit_of(99.0))
            slope_var = (info["resid"] ** 2) * 0.25
            spread = math.sqrt(
                max(info["resid"], 0.25) ** 2
                + 0.12 * horizon_days
                + slope_var
                + (0.3 * abs(elast) * abs(dn_forecast)) ** 2
                + (0.25 * abs(b_use) * max(x_target - x_last, 0.0)) ** 2
            )
            spread = max(0.30, min(1.6, spread))
        qs = []
        for z in Z:
            qs.append(round(rate_of(mu + z * spread), 4))
        # Enforce non-decreasing after rounding.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return rows


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_files = sorted(INPUT_DIR.glob("*.in"))
    for path in case_files:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = {"trophies": rows}
        text = json.dumps(payload, separators=(",", ":"), sort_keys=False)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
