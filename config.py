"""Project configuration: ticker universe, date range, stress windows.

Edit this file to change what the analysis covers. Everything else reads from here.
"""

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
# yfinance treats `end` as exclusive. Pinned so the README numbers are reproducible;
# set to None to pull through the latest available day.
END_DATE = "2026-11-18"

# ---------------------------------------------------------------------------
# Universe
#   category    -> how the ETF is grouped in charts/tables (as in the project spec)
#   asset_class -> the economic asset class, used for the regression dummies.
#                  The "Niche" group is defined by being thinly traded, so using it
#                  as a regression dummy would soak up the volume effect we want to
#                  measure. Each niche ETF is therefore mapped to its true asset class.
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Large-cap US equity
    "SPY": ("Large-cap US", "US equity"),
    "IVV": ("Large-cap US", "US equity"),
    "VOO": ("Large-cap US", "US equity"),
    "QQQ": ("Large-cap US", "US equity"),
    "IWM": ("Large-cap US", "US equity"),
    "DIA": ("Large-cap US", "US equity"),
    # Sectors
    "XLF": ("Sector", "US equity"),
    "XLE": ("Sector", "US equity"),
    "XLK": ("Sector", "US equity"),
    "XLV": ("Sector", "US equity"),
    "XLU": ("Sector", "US equity"),
    "XLRE": ("Sector", "US equity"),
    # International equity
    "EFA": ("International", "Intl equity"),
    "EEM": ("International", "Intl equity"),
    "VWO": ("International", "Intl equity"),
    "EWZ": ("International", "Intl equity"),
    "FXI": ("International", "Intl equity"),
    # Bonds
    "AGG": ("Bonds", "Bonds"),
    "TLT": ("Bonds", "Bonds"),
    "IEF": ("Bonds", "Bonds"),
    "SHY": ("Bonds", "Bonds"),
    "LQD": ("Bonds", "Bonds"),
    "HYG": ("Bonds", "Bonds"),
    "JNK": ("Bonds", "Bonds"),
    "EMB": ("Bonds", "Bonds"),
    "MUB": ("Bonds", "Bonds"),
    # Commodities / other
    "GLD": ("Commodities", "Commodities"),
    "SLV": ("Commodities", "Commodities"),
    "USO": ("Commodities", "Commodities"),
    "UNG": ("Commodities", "Commodities"),
    # Thinner / niche, chosen for contrast (median dollar volume ~$1-5M/day vs
    # ~$30B/day for SPY), one from each of four different corners of the market,
    # all trading continuously since before 2018:
    "PSCH": ("Niche (thin)", "US equity"),    # Invesco S&P SmallCap Health Care
    "PEJ": ("Niche (thin)", "US equity"),     # Invesco Leisure & Entertainment
    "EWN": ("Niche (thin)", "Intl equity"),   # iShares MSCI Netherlands
    "FXF": ("Niche (thin)", "Commodities"),   # Invesco CurrencyShares Swiss Franc
}

TICKERS = list(UNIVERSE)
CATEGORY = {t: v[0] for t, v in UNIVERSE.items()}
ASSET_CLASS = {t: v[1] for t, v in UNIVERSE.items()}
CATEGORY_ORDER = ["Large-cap US", "Sector", "International", "Bonds",
                  "Commodities", "Niche (thin)"]

# ---------------------------------------------------------------------------
# Stress windows (inclusive). A window can be a union of date ranges.
# ---------------------------------------------------------------------------
STRESS_WINDOWS = {
    "COVID crash (Feb 20 - Apr 30, 2020)": [("2020-02-20", "2020-04-30")],
    "2022 rate shock (Jun, Sep-Oct 2022)": [("2022-06-01", "2022-06-30"),
                                            ("2022-09-01", "2022-10-31")],
    "Tariff selloff (Apr 2 - 30, 2025)": [("2025-04-02", "2025-04-30")],
}
# Short labels for charts
STRESS_SHORT = {
    "COVID crash (Feb 20 - Apr 30, 2020)": "COVID 2020",
    "2022 rate shock (Jun, Sep-Oct 2022)": "Rates 2022",
    "Tariff selloff (Apr 2 - 30, 2025)": "Tariffs 2025",
}

# Five representative ETFs for the time-series chart
REPRESENTATIVE = ["SPY", "XLRE", "EEM", "LQD", "PEJ"]

DATA_DIR = "data"
CHART_DIR = "charts"
RESULTS_DIR = "results"
