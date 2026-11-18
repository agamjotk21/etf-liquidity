"""End-to-end analysis: spreads -> monthly panel -> regression -> stress -> charts -> README.

    python analysis.py

Downloads data on first run (see data.py), writes tables to results/, charts to
charts/, and renders README.md from readme_template.md so every number in the
write-up comes straight from this run.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

import config  # noqa: E402
import data  # noqa: E402
import spreads  # noqa: E402

BP = 1e4
MIN_DAYS = 15  # drop partial months with fewer trading days than this

# ---------------------------------------------------------------------------
# Chart style (light surface; categorical slots in fixed order, one per category)
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT_COLOR = dict(zip(config.CATEGORY_ORDER,
                     ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CAT_MARKER = dict(zip(config.CATEGORY_ORDER, ["o", "s", "^", "D", "v", "P"]))
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 200,
})


def _save(fig, name):
    path = os.path.join(config.CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _source(fig):
    fig.text(0.01, -0.02, "Source: Yahoo Finance daily OHLCV; spread estimates by the author.",
             fontsize=8, color=INK2)


# ---------------------------------------------------------------------------
# 1. Daily estimates and monthly panel
# ---------------------------------------------------------------------------
def build_daily(prices):
    out = []
    for t, d in prices.items():
        d = d.copy()
        d["cs"] = spreads.corwin_schultz(d["High"], d["Low"], d["Close"])
        d["ret"] = np.log(d["Adj Close"]).diff()  # total-return log returns
        d["dollar_vol"] = d["Close"] * d["Volume"]
        d["tick"] = 0.01 / d["Close"]  # one-cent tick as a fraction of price
        d["month"] = d.index.strftime("%Y-%m")
        d["ticker"] = t
        out.append(d)
    return pd.concat(out)


def build_monthly(daily):
    rows = []
    for t, d in daily.groupby("ticker"):
        g = d.groupby("month")
        p = pd.DataFrame({
            "cs": g["cs"].mean(),
            "vol": g["ret"].std() * np.sqrt(252),
            "dollar_vol": g["dollar_vol"].mean(),
            "tick": g["tick"].mean(),
            "n_days": g.size(),
        })
        p["ar"] = spreads.abdi_ranaldo(d["High"], d["Low"], d["Close"], d["month"])
        p["ticker"] = t
        rows.append(p)
    m = pd.concat(rows).reset_index()
    m = m[m["n_days"] >= MIN_DAYS].copy()
    m["category"] = m["ticker"].map(config.CATEGORY)
    m["asset_class"] = m["ticker"].map(config.ASSET_CLASS)
    for c in ["cs", "ar", "tick"]:
        m[c + "_bp"] = m[c] * BP
    return m


# ---------------------------------------------------------------------------
# 2. Regressions
# ---------------------------------------------------------------------------
def run_regressions(m):
    m = m.copy()
    m["log_spread"] = np.log(m["cs_bp"])
    m["log_dv"] = np.log(m["dollar_vol"])
    m["log_vol"] = np.log(m["vol"])
    m["asset_class"] = pd.Categorical(
        m["asset_class"], ["Bonds", "US equity", "Intl equity", "Commodities"])
    cl = {"cov_type": "cluster", "cov_kwds": {"groups": m["ticker"]}}
    models = {
        "main": smf.ols("log_spread ~ log_dv + vol + C(asset_class)", m).fit(**cl),
        "volume_only": smf.ols("log_spread ~ log_dv + C(asset_class)", m).fit(**cl),
        "vol_only_loglog": smf.ols("log_spread ~ log_vol", m).fit(**cl),
        "loglog_full": smf.ols("log_spread ~ log_dv + log_vol + C(asset_class)", m).fit(**cl),
    }
    # Pure cross-section robustness: one row per ETF (full-period medians)
    e = m.groupby("ticker").agg(cs_bp=("cs_bp", "median"), dv=("dollar_vol", "median"),
                                vol=("vol", "median"),
                                asset_class=("asset_class", "first")).reset_index()
    e["log_spread"], e["log_dv"] = np.log(e["cs_bp"]), np.log(e["dv"])
    models["etf_cross_section"] = smf.ols(
        "log_spread ~ log_dv + vol + C(asset_class)", e).fit(cov_type="HC1")

    with open(os.path.join(config.RESULTS_DIR, "regression.txt"), "w") as f:
        for k, r in models.items():
            se = "HC1 robust" if k == "etf_cross_section" else "clustered by ETF"
            f.write(f"===== {k}  (standard errors {se}) =====\n{r.summary()}\n\n")

    # Residual: how far each ETF's estimated spread sits from what its volatility,
    # volume and asset class predict (log points, averaged over months).
    m["resid"] = models["loglog_full"].resid
    resid = m.groupby("ticker")["resid"].mean().sort_values(ascending=False)
    return models, resid


# ---------------------------------------------------------------------------
# 3. Stress windows
# ---------------------------------------------------------------------------
def _in_window(idx, ranges):
    mask = np.zeros(len(idx), dtype=bool)
    for a, b in ranges:
        mask |= (idx >= a) & (idx <= b)
    return mask


def stress_analysis(daily):
    any_stress = np.zeros(len(daily), dtype=bool)
    for r in config.STRESS_WINDOWS.values():
        any_stress |= _in_window(daily.index, r)
    calm = daily[~any_stress].groupby("ticker").agg(cs=("cs", "mean"), vol=("ret", "std"))
    rows = []
    for name, ranges in config.STRESS_WINDOWS.items():
        s = daily[_in_window(daily.index, ranges)].groupby("ticker").agg(
            cs=("cs", "mean"), vol=("ret", "std"))
        r = pd.DataFrame({
            "window": config.STRESS_SHORT[name],
            "calm_cs_bp": calm["cs"] * BP,
            "stress_cs_bp": s["cs"] * BP,
            "spread_ratio": s["cs"] / calm["cs"],
            "vol_ratio": s["vol"] / calm["vol"],
        })
        r["excess"] = r["spread_ratio"] / r["vol_ratio"]
        r["category"] = r.index.map(config.CATEGORY)
        rows.append(r.reset_index())
    by_etf = pd.concat(rows)
    by_cat = (by_etf.groupby(["window", "category"])[["spread_ratio", "vol_ratio", "excess"]]
              .median().reset_index())
    return by_etf, by_cat


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------
def chart_category(etf):
    g = etf.groupby("category")[["cs_bp", "tick_bp"]].median().reindex(config.CATEGORY_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(g))
    ax.bar(x, g["cs_bp"], width=0.62, color=[CAT_COLOR[c] for c in g.index],
           edgecolor=SURFACE, linewidth=2)
    ax.bar([0], [0], color=INK2, label="Estimated spread (Corwin-Schultz)")
    ax.scatter(x, g["tick_bp"], marker="_", s=900, color=INK, linewidths=2.5, zorder=3,
               label="One-cent tick (the smallest possible quoted spread)")
    for xi, v in zip(x, g["cs_bp"]):
        ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, g.index)
    ax.set_ylabel("Basis points (median ETF in category)")
    ax.set_title("Estimated spreads look high everywhere, even for the most liquid funds")
    ax.set_ylim(0, g["cs_bp"].max() * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.1, "Source: Yahoo Finance daily OHLCV; spread estimates by the author.",
             fontsize=8, color=INK2)
    return _save(fig, "1_spread_by_category.png")


LABEL = ["SPY", "QQQ", "IWM", "SHY", "MUB", "AGG", "LQD", "HYG", "TLT", "UNG", "USO", "GLD",
         "FXF", "PEJ", "PSCH", "EWN", "EWZ", "XLE", "XLRE", "EEM"]


def chart_scatter(etf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, xcol, xlabel, title in [
        (axes[0], "dollar_vol", "Average daily dollar volume (log scale)",
         "vs. dollar volume: no clear relationship"),
        (axes[1], "vol", "Annualized volatility (log scale)",
         "vs. volatility: an almost one-for-one line"),
    ]:
        for cat in config.CATEGORY_ORDER:
            s = etf[etf["category"] == cat]
            ax.scatter(s[xcol], s["cs_bp"], s=46, color=CAT_COLOR[cat],
                       marker=CAT_MARKER[cat], edgecolor=SURFACE, linewidth=1.5,
                       label=cat, zorder=3)
        for _, r in etf[etf["ticker"].isin(LABEL)].iterrows():
            ax.annotate(r["ticker"], (r[xcol], r["cs_bp"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Median estimated spread, bp (log scale)")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    for ax in axes:
        ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1].xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5]))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"${v / 1e9:g}B" if v >= 1e9 else f"${v / 1e6:g}M"))
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("What drives the estimated spread? (one point per ETF, 2018-2026 medians)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    _source(fig)
    fig.tight_layout()
    return _save(fig, "2_spread_vs_volume_and_vol.png")


def chart_timeseries(m):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, ranges in config.STRESS_WINDOWS.items():
        for i, (a, b) in enumerate(ranges):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d8d6d0", alpha=0.6, lw=0)
        ax.text(pd.Timestamp(ranges[0][0]), 1.0, " " + config.STRESS_SHORT[name],
                transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="top")
    for t in config.REPRESENTATIVE:
        s = m[m["ticker"] == t].set_index("month")["cs_bp"]
        s.index = pd.to_datetime(s.index)
        c = CAT_COLOR[config.CATEGORY[t]]
        ax.plot(s.index, s.values, color=c, lw=2, label=f"{t} ({config.CATEGORY[t]})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Monthly avg. estimated spread, bp (log scale)")
    ax.set_title("Estimated spreads spike in every stress window, for every kind of ETF",
                 pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8)
    ax.grid(axis="x", visible=False)
    _source(fig)
    return _save(fig, "3_spread_over_time.png")


def chart_stress(by_cat):
    windows = list(config.STRESS_SHORT.values())
    fig, axes = plt.subplots(1, len(windows), figsize=(12, 4.4), sharey=True)
    for ax, w in zip(axes, windows):
        s = by_cat[by_cat["window"] == w].set_index("category").reindex(config.CATEGORY_ORDER)
        y = np.arange(len(s))[::-1]
        ax.barh(y, s["spread_ratio"], height=0.62, color=[CAT_COLOR[c] for c in s.index],
                edgecolor=SURFACE, linewidth=2)
        ax.scatter(s["vol_ratio"], y, marker="|", s=420, color=INK, linewidths=2.2, zorder=3,
                   label="Volatility ratio (same window)")
        xmax = max(s["spread_ratio"].max(), s["vol_ratio"].max())
        for yi, v, vr in zip(y, s["spread_ratio"], s["vol_ratio"]):
            ax.text(max(v, vr) + xmax * 0.03, yi, f"{v:.1f}x", fontsize=8, color=INK,
                    va="center")
        ax.set_xlim(0, xmax * 1.18)
        ax.axvline(1, color=INK2, lw=1, ls=":")
        ax.set_yticks(y, s.index)
        ax.set_title(w, fontsize=11)
        ax.set_xlabel("Stress / calm")
        ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=INK2, label="Estimated spread, stress / calm (bar, label)"),
                        Line2D([], [], color=INK, marker="|", ls="", ms=14, mew=2.2,
                               label="Volatility, stress / calm (tick)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)
    fig.suptitle("Spread widening in stress (bars) closely tracks volatility (black ticks)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.text(0.01, -0.1, "Source: Yahoo Finance daily OHLCV; spread estimates by the author. "
             "Ratios are medians across ETFs in each category.", fontsize=8, color=INK2)
    return _save(fig, "4_stress_widening.png")


def chart_agreement(m):
    s = m[m["ar_bp"] > 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    for cat in config.CATEGORY_ORDER:
        c = s[s["category"] == cat]
        ax.scatter(c["cs_bp"], c["ar_bp"], s=10, alpha=0.55, color=CAT_COLOR[cat],
                   marker=CAT_MARKER[cat], linewidth=0, label=cat)
    lo, hi = 0.5, max(s["cs_bp"].max(), s["ar_bp"].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", label="45-degree line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Corwin-Schultz, bp (log)")
    ax.set_ylabel("Abdi-Ranaldo, bp (log)")
    ax.set_title("The two estimators agree only loosely (ETF-months)")
    ax.legend(fontsize=7, loc="upper left", markerscale=1.8)
    zero = (m["ar_bp"] == 0).mean()
    ax.text(0.99, 0.02, f"{zero:.0%} of months with Abdi-Ranaldo = 0 not shown",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    _source(fig)
    return _save(fig, "5_estimator_agreement.png")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def pct(x):
    return f"{x:.0%}"


def main():
    for d in (config.CHART_DIR, config.RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    # Guard: estimator unit tests must pass before any numbers are produced.
    for name in [n for n in dir(spreads) if n.startswith("test_")]:
        getattr(spreads, name)()

    prices = data.load_all()
    daily = build_daily(prices)
    m = build_monthly(daily)

    etf = m.groupby("ticker").agg(
        category=("category", "first"), cs_bp=("cs_bp", "median"), ar_bp=("ar_bp", "median"),
        tick_bp=("tick_bp", "median"), dollar_vol=("dollar_vol", "median"),
        vol=("vol", "median"), ar_zero_share=("ar_bp", lambda x: (x == 0).mean()),
    ).reset_index()
    etf["cs_over_tick"] = etf["cs_bp"] / etf["tick_bp"]
    models, resid = run_regressions(m)
    etf["resid"] = etf["ticker"].map(resid)
    etf.sort_values("cs_bp").to_csv(os.path.join(config.RESULTS_DIR, "etf_summary.csv"),
                                    index=False, float_format="%.4g")
    m.to_csv(os.path.join(config.RESULTS_DIR, "monthly_panel.csv"), index=False,
             float_format="%.6g")

    by_etf, by_cat = stress_analysis(daily)
    by_etf.to_csv(os.path.join(config.RESULTS_DIR, "stress_by_etf.csv"), index=False,
                  float_format="%.4g")
    by_cat.to_csv(os.path.join(config.RESULTS_DIR, "stress_by_category.csv"), index=False,
                  float_format="%.4g")

    charts = [chart_category(etf), chart_scatter(etf), chart_timeseries(m),
              chart_stress(by_cat), chart_agreement(m)]

    # ---------------- numbers for the write-up ----------------
    E = etf.set_index("ticker")
    main_m, vonly, loglog = models["main"], models["vol_only_loglog"], models["loglog_full"]
    cat_med = etf.groupby("category")["cs_bp"].median()
    covid = by_etf[by_etf["window"] == "COVID 2020"].set_index("ticker")
    covid_cat = by_cat[by_cat["window"] == "COVID 2020"].set_index("category")
    widest = {w: by_cat[by_cat["window"] == w].sort_values("spread_ratio").iloc[-1]
              for w in config.STRESS_SHORT.values()}
    bonds = [t for t in config.TICKERS if config.CATEGORY[t] == "Bonds"]
    hi_resid = [t for t in resid.index[:6]]
    lo_resid = [t for t in resid.index[::-1][:6]]
    first, last = daily.index.min(), daily.index.max()
    xs = models["etf_cross_section"]

    N = {
        "n_etfs": len(config.TICKERS),
        "start": first.strftime("%b %Y"), "end": last.strftime("%b %d, %Y"),
        "years": f"{first.year}-{last.year}",
        "n_obs": int(main_m.nobs), "n_days": daily.groupby("ticker").size().iloc[0],
        "spy_cs": f"{E.loc['SPY', 'cs_bp']:.1f}", "spy_ar": f"{E.loc['SPY', 'ar_bp']:.1f}",
        "spy_tick": f"{E.loc['SPY', 'tick_bp']:.2f}",
        "spy_x": f"{E.loc['SPY', 'cs_over_tick']:.0f}",
        "psch_cs": f"{E.loc['PSCH', 'cs_bp']:.0f}", "pej_cs": f"{E.loc['PEJ', 'cs_bp']:.0f}",
        "spy_vs_psch_dv": f"{round(E.loc['SPY', 'dollar_vol'] / E.loc['PSCH', 'dollar_vol'], -3):,.0f}",
        "shy_cs": f"{E.loc['SHY', 'cs_bp']:.1f}", "shy_tick": f"{E.loc['SHY', 'tick_bp']:.1f}",
        "shy_x": f"{E.loc['SHY', 'cs_over_tick']:.1f}",
        "cs_min_t": E["cs_bp"].idxmin(), "cs_min": f"{E['cs_bp'].min():.1f}",
        "cs_max_t": E["cs_bp"].idxmax(), "cs_max": f"{E['cs_bp'].max():.0f}",
        "bond_med": f"{cat_med['Bonds']:.1f}", "lc_med": f"{cat_med['Large-cap US']:.1f}",
        "niche_med": f"{cat_med['Niche (thin)']:.1f}", "sector_med": f"{cat_med['Sector']:.1f}",
        "comm_med": f"{cat_med['Commodities']:.1f}", "intl_med": f"{cat_med['International']:.1f}",
        "ar_zero": pct((m["ar_bp"] == 0).mean()),
        "rho_cs_ar": f"{m[['cs_bp', 'ar_bp']].corr('spearman').iloc[0, 1]:.2f}",
        "r2_vol": f"{vonly.rsquared:.2f}", "elast_vol": f"{vonly.params['log_vol']:.2f}",
        "elast_vol_se": f"{vonly.bse['log_vol']:.2f}",
        "r2_main": f"{main_m.rsquared:.2f}", "r2_volume_only": f"{models['volume_only'].rsquared:.2f}",
        "b_dv": f"{main_m.params['log_dv']:+.3f}", "se_dv": f"{main_m.bse['log_dv']:.3f}",
        "dv_10x": f"{np.exp(main_m.params['log_dv'] * np.log(10)) - 1:+.0%}",
        "b_vol": f"{main_m.params['vol']:.2f}", "se_vol": f"{main_m.bse['vol']:.2f}",
        "vol_10pt": f"{np.exp(main_m.params['vol'] * 0.10) - 1:+.0%}",
        "b_us": f"{main_m.params['C(asset_class)[T.US equity]']:.2f}",
        "us_pct": f"{np.exp(main_m.params['C(asset_class)[T.US equity]']) - 1:+.0%}",
        "b_dv_ll": f"{loglog.params['log_dv']:+.3f}", "se_dv_ll": f"{loglog.bse['log_dv']:.3f}",
        "r2_ll": f"{loglog.rsquared:.2f}",
        "b_dv_xs": f"{xs.params['log_dv']:+.3f}", "se_dv_xs": f"{xs.bse['log_dv']:.3f}",
        "r2_xs": f"{xs.rsquared:.2f}",
        "covid_bond_s": f"{covid_cat.loc['Bonds', 'spread_ratio']:.1f}",
        "covid_bond_v": f"{covid_cat.loc['Bonds', 'vol_ratio']:.1f}",
        "covid_lc_s": f"{covid_cat.loc['Large-cap US', 'spread_ratio']:.1f}",
        "covid_lc_v": f"{covid_cat.loc['Large-cap US', 'vol_ratio']:.1f}",
        "lqd_s": f"{covid.loc['LQD', 'spread_ratio']:.1f}",
        "lqd_v": f"{covid.loc['LQD', 'vol_ratio']:.1f}",
        "hyg_s": f"{covid.loc['HYG', 'spread_ratio']:.1f}",
        "hyg_v": f"{covid.loc['HYG', 'vol_ratio']:.1f}",
        "mub_s": f"{covid.loc['MUB', 'spread_ratio']:.1f}",
        "mub_v": f"{covid.loc['MUB', 'vol_ratio']:.1f}",
        "bond_excess_lt1": int((covid.loc[bonds, "excess"] < 1).sum()), "n_bonds": len(bonds),
        "corr_stress": f"{np.corrcoef(np.log(by_etf['spread_ratio']), np.log(by_etf['vol_ratio']))[0, 1]:.2f}",
        "widest_covid": f"{widest['COVID 2020']['category']} ({widest['COVID 2020']['spread_ratio']:.1f}x)",
        "widest_2022": f"{widest['Rates 2022']['category']} ({widest['Rates 2022']['spread_ratio']:.1f}x)",
        "widest_2025": f"{widest['Tariffs 2025']['category']} ({widest['Tariffs 2025']['spread_ratio']:.1f}x)",
        "hi_resid": ", ".join(f"{t} ({np.exp(resid[t]) - 1:+.0%})" for t in hi_resid),
        "lo_resid": ", ".join(f"{t} ({np.exp(resid[t]) - 1:+.0%})" for t in lo_resid),
    }
    with open(os.path.join(config.RESULTS_DIR, "key_numbers.json"), "w") as f:
        json.dump({k: str(v) for k, v in N.items()}, f, indent=2)

    with open("readme_template.md") as f:
        readme = f.read().format(**N)
    with open("README.md", "w") as f:
        f.write(readme)

    bullet = (
        f"Analyzed trading costs for {N['n_etfs']} ETFs ({N['years']}) by estimating bid-ask "
        f"spreads from daily price data (Corwin-Schultz, Abdi-Ranaldo); found that the "
        f"estimates are driven by volatility rather than liquidity (R² = {N['r2_vol']} on "
        f"volatility alone; SPY estimated at {N['spy_cs']} bp vs. a {N['spy_tick']} bp tick "
        f"floor) and that stress-period widening tracks volatility (COVID bond-ETF "
        f"spreads {N['covid_bond_s']}x vs. volatility {N['covid_bond_v']}x), and recommended "
        f"volatility-adjusted spread screens plus quote-level data for any ETF liquidity "
        f"pricing decision."
    )
    with open(os.path.join(config.RESULTS_DIR, "resume_bullet.txt"), "w") as f:
        f.write(bullet + "\n")

    # ---------------- console summary ----------------
    print(f"\n{N['n_etfs']} ETFs, {first.date()} to {last.date()}, {N['n_obs']} ETF-months")
    print(f"SPY: CS {N['spy_cs']} bp, AR {N['spy_ar']} bp, one-cent tick {N['spy_tick']} bp "
          f"-> CS is {N['spy_x']}x the tick floor")
    print("SANITY CHECK FAILED: SPY estimate is far above its true ~1-tick spread. "
          "This is the estimator's known upward bias for liquid ETFs; see README.")
    print(f"log(spread) ~ log(vol): R2 {N['r2_vol']}, elasticity {N['elast_vol']}")
    print(f"Main regression R2 {N['r2_main']}; log dollar volume {N['b_dv']} "
          f"(SE {N['se_dv']}); vol {N['b_vol']} (SE {N['se_vol']})")
    print("\nStress widening by category (median spread ratio / vol ratio):")
    print(by_cat.pivot(index="category", columns="window",
                       values="spread_ratio").reindex(config.CATEGORY_ORDER).round(2))
    print(f"\nHighest spread vs. vol/volume prediction: {N['hi_resid']}")
    print(f"Lowest: {N['lo_resid']}")
    print("\nCharts:", *charts, sep="\n  ")
    print("\nWrote README.md, results/, and results/resume_bullet.txt")


if __name__ == "__main__":
    main()
