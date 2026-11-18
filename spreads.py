"""Bid-ask spread estimators from daily high / low / close prices.

* Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily
  High and Low Prices", Journal of Finance 67(2). https://doi.org/10.1111/j.1540-6261.2012.01729.x
* Abdi & Ranaldo (2017), "A Simple Estimation of Bid-Ask Spreads from Daily
  Close, High, and Low Prices", Review of Financial Studies 30(12).
  https://doi.org/10.1093/rfs/hhx084

Both return the spread as a fraction of price (multiply by 1e4 for basis points).

Run the unit tests with:  python spreads.py   (or: pytest spreads.py)
"""
import numpy as np
import pandas as pd

K = 3 - 2 * np.sqrt(2)  # constant from Corwin-Schultz eq. (14)


# ---------------------------------------------------------------------------
# Corwin-Schultz
# ---------------------------------------------------------------------------
def corwin_schultz(high, low, close):
    """Daily two-day Corwin-Schultz spread estimates.

    The estimate for the pair of days (t, t+1) is stored on day t+1. Negative
    two-day estimates are set to 0, as in the paper.

    Overnight adjustment (paper, Section II.B): if day t+1's low is above day t's
    close, the price is assumed to have gapped up overnight, so day t+1's high and
    low are both lowered by (L[t+1] - C[t]). If day t+1's high is below day t's
    close, both are raised by (C[t] - H[t+1]). This keeps overnight moves (news,
    ex-dividend drops) from being mistaken for a wide intraday range.
    """
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)

    h0, l0, c0 = high.shift(1), low.shift(1), close.shift(1)  # day t
    h1, l1 = high.copy(), low.copy()                            # day t+1

    gap_up = (l1 - c0).clip(lower=0).fillna(0)
    gap_down = (c0 - h1).clip(lower=0).fillna(0)
    adj = -gap_up + gap_down
    h1, l1 = h1 + adj, l1 + adj

    beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(np.maximum(h0, h1) / np.minimum(l0, l1)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K - np.sqrt(gamma / K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


# ---------------------------------------------------------------------------
# Abdi-Ranaldo
# ---------------------------------------------------------------------------
def abdi_ranaldo(high, low, close, groups):
    """Abdi-Ranaldo spread estimate per group (e.g. per calendar month).

    eta_t = (ln H_t + ln L_t) / 2      (log mid-range, a proxy for the efficient price)
    c_t   = ln C_t
    S^2   = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})]

    The expectation is the mean over days t in the group. A negative S^2 is floored
    at 0. `groups` is an array-like of group labels aligned with the prices.
    """
    high = pd.Series(np.asarray(high, dtype=float))
    low = pd.Series(np.asarray(low, dtype=float))
    close = pd.Series(np.asarray(close, dtype=float))
    eta = (np.log(high) + np.log(low)) / 2
    c = np.log(close)
    prod = (c - eta) * (c - eta.shift(-1))
    s2 = 4 * prod.groupby(np.asarray(groups)).mean()
    return np.sqrt(s2.clip(lower=0))


# ---------------------------------------------------------------------------
# Unit tests on tiny hand-checked examples
# ---------------------------------------------------------------------------
def test_cs_constant_range_no_drift():
    """Two days, each with H=101, L=99, no overnight gap.

    By hand: r = ln(101/99) = 0.0200007
      beta  = 2 r^2                 = 0.00080005
      gamma = r^2 (two-day range is the same as one day's) = 0.00040003
      alpha = (sqrt(2b) - sqrt(b)) / K - sqrt(g / K)
            = (0.0400013 - 0.0282852) / 0.1715729 - sqrt(0.0023315)
            = 0.0682864 - 0.0482857 = 0.0200007   (= r exactly)
      S     = 2(e^a - 1)/(1 + e^a)  = 0.0200000  (about 200 bp)
    Intuition: if the range never grows with the horizon there's no volatility,
    so all of the range is attributed to the spread, and S equals the log range.
    """
    s = corwin_schultz([101, 101], [99, 99], [100, 100])
    assert np.isnan(s.iloc[0])  # first day has no pair
    assert abs(s.iloc[1] - 0.0200000) < 1e-6


def test_cs_pure_volatility_gives_zero():
    """Day 1 trades 100-101 and day 2 trades 101-102, so the two-day range (100-102)
    is twice the daily range. Pure drift with no spread: alpha < 0, so S is set to 0.
      r = ln(1.01) ~= 0.00995, beta = 2 r^2, gamma = (ln 1.02)^2 ~= 4 r^2
      alpha = r(2 - sqrt2)/K - 2r/sqrt(K) = r(3.414 - 4.828) < 0
    (Day 2's low equals day 1's close, so there is no gap and no adjustment.)"""
    s = corwin_schultz([101, 102], [100, 101], [101, 101.5])
    assert s.iloc[1] == 0.0


def test_cs_overnight_gap_is_removed():
    """Day 1: H=101, L=99, closes at 99. Day 2 gaps up and trades H=111, L=109.
    Gap = L[t+1] - C[t] = 109 - 99 = 10, so day 2 is lowered to H=101, L=99, which is
    exactly the constant-range case above, so S should be 0.0200000 again.
    Without the adjustment, gamma = ln(111/99)^2 is huge, alpha < 0, and S = 0:
    the overnight jump would be misread as volatility.
    Gap down is the mirror case: day 1 closes at 101, day 2 trades H=91, L=89.
    Gap = C[t] - H[t+1] = 10, so day 2 is raised to 101/99 and S = 0.0200000."""
    up = corwin_schultz([101, 111], [99, 109], [99, 110])
    down = corwin_schultz([101, 91], [99, 89], [101, 90])
    assert abs(up.iloc[1] - 0.0200000) < 1e-6
    assert abs(down.iloc[1] - 0.0200000) < 1e-6


def test_ar_pure_bid_ask_bounce():
    """Mid-range is fixed at 100 (H=101, L=99 every day) and the close alternates
    between the high and the low, like a trade bouncing between ask and bid.
      c - eta alternates +/- d with d = (ln101 - ln99)/2 = 0.0100003
      eta is constant, so (c_t - eta_t)(c_t - eta_{t+1}) = d^2 every day
      S^2 = 4 d^2  ->  S = 2d = ln(101/99) = 0.0200007
    """
    n = 6
    high, low = [101] * n, [99] * n
    close = [101 if i % 2 == 0 else 99 for i in range(n)]
    s = abdi_ranaldo(high, low, close, ["m"] * n)
    assert abs(s["m"] - np.log(101 / 99)) < 1e-9


def test_ar_close_at_mid_gives_zero():
    """Close always at the log mid-range, so c - eta = 0 and S = 0."""
    high, low = [101, 102, 103], [99, 100, 101]
    close = [np.sqrt(h * l) for h, l in zip(high, low)]
    s = abdi_ranaldo(high, low, close, ["m"] * 3)
    assert abs(s["m"]) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)} tests passed")
