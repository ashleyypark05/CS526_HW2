"""
What Does Federal R&D Funding Actually Track?
=============================================

A companion to `main.py`. `main.py` established that the headline R&D number
needs a deflator, a denominator, and a breakdown. This file asks the next
question: when federal R&D moves, what else is moving with it?

Method note - the reason this file exists
-----------------------------------------
Almost any two upward-trending series correlate at r > 0.8. Federal R&D, GDP,
the budget, industry R&D and university R&D all rise over fifty years, so
their LEVEL correlations are nearly meaningless. Every candidate here is
therefore tested twice: once on levels (the number that looks impressive) and
once on annual real growth rates, which removes the shared trend and asks
whether the two series actually move together year to year. Only the second
test is evidence.

Candidates are pre-specified from a plausible mechanism, not screened from the
whole workbook, so the p-values are not the product of a fishing expedition.

  Section 1  Candidate screen: levels vs growth.                  [fig 10]
  Section 2  Budget elasticity, by agency.                        [fig 11]
  Section 3  Where the budget cycle actually lands.               [fig 12]
  Section 4  Inside the transmission channel.                     [fig 13]

Infrastructure, palette, and plotting helpers are imported from `main.py`, so
the two files cannot drift apart on colour or chart style.

Run:  python drivers.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

from main import (
    BASE_YEAR, FY_START, FY_END, DATA_DIR, INK, PALETTE, SLOT, D,
    Totals, apply_chart_style, cagr, finish_axes, fmt_pct0, fmt_plain,
    header, index_to_base, label_series_end, load_totals, load_wide_sheet,
    note, pick_series, save_figure, year_columns,
)

# Family C - external series R&D is compared AGAINST. Hues are inherited from
# main.py's Family A so a denominator keeps its colour across both files: the
# GDP benchmark is red wherever it appears, the discretionary budget is aqua,
# and federal R&D itself stays blue. Family C never shares a figure with
# Family B (agencies, character of work), so no figure repeats a colour.
PALETTE |= {
    "rd": PALETTE["constant"],          # federal R&D, the subject
    "budget": PALETTE["disc_share"],    # total discretionary budget authority
    "gdp": PALETTE["gdp_share"],        # GDP
    "industry": SLOT[2],                # industry-funded R&D
    "university": SLOT[5],              # university-funded R&D
    "ns": INK["muted"],                 # a coefficient that is not significant
}

ALPHA = 0.05


# ===========================================================================
# SHARED LOADERS
# ===========================================================================

def load_aaas_workbook(name: str) -> pd.DataFrame:
    """Read one of the standalone AAAS workbooks into a row x year frame.

    These six files share a layout `main.py`'s loaders do not cover: a title
    row, a units row, a row of fiscal years (sometimes labelled '2025
    Request'), then one row per programme. The year row is found by taking
    whichever of the first few rows parses the most four-digit years, which
    works regardless of how the label column is headed.
    """
    raw = pd.read_excel(DATA_DIR / f"{name}.xlsx", sheet_name="Data", header=None)
    best = (0, None, None)
    for i in range(min(8, len(raw))):
        years, cols = [], []
        for j, value in enumerate(raw.iloc[i, 1:], start=1):
            match = re.match(r"^(\d{4})\b", str(value).strip())
            if match:
                years.append(int(match.group(1)))
                cols.append(j)
        if len(years) > best[0]:
            best = (len(years), i, (years, cols))
    hdr, (years, cols) = best[1], best[2]
    body = raw.iloc[hdr + 1:, [0] + cols]
    body = body[body.iloc[:, 0].notna()]
    frame = body.set_index(body.columns[0])
    frame.columns = years
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame.index = [str(i).strip().rstrip("*").strip() for i in frame.index]
    return frame[~frame.index.duplicated(keep="first")]


def funder_series(raw: pd.DataFrame, source: str,
                  character: str = "Total R&D") -> pd.Series:
    """One national R&D series from the Funder sheet, in constant dollars."""
    years = year_columns(raw)
    mask = ((raw["Character"] == character)
            & (raw["Funding Source"] == source)
            & (raw["Constant Dollars"] == f"Constant {BASE_YEAR}"))
    series = raw.loc[mask, years].iloc[0].astype(float)
    series.index = [int(float(y)) for y in series.index]
    return series.dropna().rename(source)


def growth(series: pd.Series) -> pd.Series:
    """Annual log change - the detrended series every test below runs on."""
    return np.log(series.astype(float)).diff().dropna()


def paired_correlation(y: pd.Series, x: pd.Series,
                       lo: int = FY_START, hi: int = FY_END) -> dict:
    """Correlate two series on levels and on growth over a common window."""
    joined = pd.DataFrame({"y": y, "x": x}).dropna().loc[lo:hi]
    g = np.log(joined).diff().dropna()
    r_level, _ = stats.pearsonr(joined["y"], joined["x"])
    r_growth, p_growth = stats.pearsonr(g["y"], g["x"])
    return {"n": len(joined), "r_level": r_level,
            "r_growth": r_growth, "p_growth": p_growth}


def elasticity(y: pd.Series, x: pd.Series,
               lo: int = FY_START, hi: int = FY_END) -> dict:
    """Regress annual real growth of `y` on that of `x`.

    The slope is an elasticity: the percent by which `y` moves for each
    percent `x` moves. beta = 1 means the component rides the budget exactly;
    beta < 1 means it is insulated from the budget cycle.
    """
    joined = pd.DataFrame({"y": y, "x": x}).dropna().loc[lo:hi]
    g = np.log(joined).diff().dropna()
    reg = stats.linregress(g["x"], g["y"])
    half = stats.t.ppf(1 - ALPHA / 2, len(g) - 2) * reg.stderr
    return {"beta": reg.slope, "se": reg.stderr, "ci": (reg.slope - half, reg.slope + half),
            "r2": reg.rvalue ** 2, "p": reg.pvalue, "n": len(g)}


# ===========================================================================
# REPORTING SCAFFOLD
# ===========================================================================

def hypothesis(number: int, claim: str, test: str, matters: str) -> None:
    """Print a hypothesis block so claim, test and stakes are never implicit."""
    header(f"HYPOTHESIS {number}")
    note(claim)
    print()
    note(f"TEST  {test}")
    print()
    note(f"WHY IT MATTERS  {matters}")


def finding(text: str) -> None:
    print()
    note(f"FINDING  {text}")


# ===========================================================================
# HYPOTHESIS 1 - R&D IS AN APPROPRIATIONS PHENOMENON, NOT AN ECONOMIC ONE
# ===========================================================================

# Each candidate is paired with the mechanism that makes it worth testing.
# Nothing is tested without one.
CANDIDATES = {
    "Discretionary budget": "R&D is appropriated almost entirely from the "
                            "discretionary budget, so the size of that pot should "
                            "bound it directly.",
    "GDP": "A richer economy can afford more public research, and R&D intensity "
           "is usually argued in percent-of-GDP terms.",
    "Industry R&D": "Federal and private research may be complements (public "
                    "science seeds private development) or substitutes (public "
                    "money crowds private money out).",
    "University R&D": "Universities fund research from their own endowments and "
                      "may lean in when federal support softens.",
}


def build_candidates(t: Totals) -> dict:
    """Assemble the pre-specified comparison series, all in real dollars."""
    e = t.enacted()
    funder = load_wide_sheet("historical", "Funder")
    return {
        "Discretionary budget": (e.discretionary / e.deflator * 1_000),
        "GDP": (e.gdp / e.deflator * 1_000),
        "Industry R&D": funder_series(funder, "Industry"),
        "University R&D": funder_series(funder, "Universities"),
    }


def test_hypothesis_1(t: Totals) -> pd.DataFrame:
    hypothesis(
        1,
        "Federal R&D tracks the appropriations process, not the economy. If R&D "
        "responded to national capacity or to what private research was doing, it "
        "should co-move with GDP and with industry R&D. If it is simply a line "
        "inside an annual budget negotiation, it should co-move with the "
        "discretionary pot and with nothing else.",
        "Correlate real federal R&D with four pre-specified candidates twice - "
        "once on levels, once on annual real growth. Levels are reported only to "
        "show how misleading they are; growth removes the shared trend and is the "
        "only one of the two that is evidence.",
        "Percent-of-GDP framing presumes R&D rides the economy, and the "
        "crowding-out debate presumes federal and private research interact. If "
        "neither relationship exists year to year, both framings are claims about "
        "levels that annual data does not support.")

    e = t.enacted()
    print("\n  Candidates and their mechanisms:")
    for name, why in CANDIDATES.items():
        print(f"    {name:<22} {why}")
    note("EXCLUDED BY CONSTRUCTION: the Function sheet's budget-function columns "
         "(national defense 050, health 550, general science 251) are federal R&D "
         "itself, split by function - not outside variables. Defense R&D against "
         "function 050 correlates at r = 0.98 on growth, which measures the "
         "partition, not a relationship.")

    rows = {name: paired_correlation(e.constant, series)
            for name, series in build_candidates(t).items()}
    table = pd.DataFrame(rows).T
    table["significant"] = table["p_growth"] < ALPHA
    print(f"\n  Correlation with total real federal R&D, FY{FY_START}-FY{FY_END}:")
    print(table.to_string())
    print(f"\n  mean level correlation {table['r_level'].mean():.2f}   "
          f"mean growth correlation {table['r_growth'].mean():.2f}")

    survivors = table[table["significant"]].index.tolist()
    dropped = table[~table["significant"]].index.tolist()
    finding(f"Only {len(survivors)} of {len(table)} candidates survives detrending: "
            f"{', '.join(survivors)} (r = {table.loc[survivors[0], 'r_growth']:.2f}). "
            f"{', '.join(dropped)} sit at r = "
            + ", ".join(f"{table.loc[d, 'r_level']:.2f}" for d in dropped)
            + " on levels and at r = "
            + ", ".join(f"{table.loc[d, 'r_growth']:+.2f}" for d in dropped)
            + " on growth. Their apparent association is shared trend.")
    note("The industry result deserves stating plainly, because both of its "
         "popular readings fail: there is no evidence here that federal R&D "
         "crowds private R&D out, and none that it pulls private R&D in.")

    budget = e.discretionary / e.deflator * 1_000
    print(f"\n  Size of the one surviving relationship (elasticity of annual real "
          f"growth), FY{FY_START}-FY{FY_END}:")
    aggregates = {}
    for label, series in [("Total R&D", e.constant), ("Defense R&D", e.defense),
                          ("Nondefense R&D", e.nondefense)]:
        fit = elasticity(series, budget)
        aggregates[label] = fit
        t_one = (fit["beta"] - 1) / fit["se"]
        p_one = 2 * (1 - stats.t.cdf(abs(t_one), fit["n"] - 2))
        print(f"    {label:<16} beta={fit['beta']:+.2f} "
              f"[{fit['ci'][0]:+.2f}, {fit['ci'][1]:+.2f}]  R2={fit['r2']:.2f}  "
              f"p={fit['p']:.1e}   vs beta=1: p={p_one:.3f}")
    note("beta = 1 means R&D rides the budget one-for-one. Total R&D sits at "
         f"{aggregates['Total R&D']['beta']:.2f}, significantly below 1 - the budget "
         "pulls R&D, but does not drag it. Hypothesis 2 asks who feels the pull.")

    plot_screen(table)
    return aggregates


def plot_screen(table: pd.DataFrame) -> None:
    """FIG 10 - level correlation vs growth correlation, as a dumbbell."""
    ordered = table.sort_values("r_growth")
    fig, ax = plt.subplots(figsize=(9.4, 3.4))

    for i, (name, row) in enumerate(ordered.iterrows()):
        ax.plot([row["r_level"], row["r_growth"]], [i, i],
                color=INK["axis"], lw=2.0, zorder=2, solid_capstyle="round")
        # The two ends differ in FILL as well as colour, so a non-significant
        # growth dot can never be mistaken for the level dot beside it.
        ax.scatter(row["r_level"], i, s=70, facecolor=INK["surface"], zorder=3,
                   edgecolor=INK["muted"], linewidth=1.8)
        ax.scatter(row["r_growth"], i,
                   s=70, color=PALETTE["rd"] if row["significant"] else PALETTE["ns"],
                   zorder=4, edgecolor=INK["surface"], linewidth=2)
        ax.annotate(f"{row['r_growth']:+.2f}", (row["r_growth"], i),
                    textcoords="offset points",
                    xytext=(-12 if row["r_growth"] < row["r_level"] else 12, 0),
                    ha="right" if row["r_growth"] < row["r_level"] else "left",
                    va="center", fontsize=9.5, fontweight="semibold",
                    color=PALETTE["rd"] if row["significant"] else INK["secondary"])
        ax.annotate(f"{row['r_level']:+.2f}", (row["r_level"], i),
                    textcoords="offset points", xytext=(12, 0), ha="left",
                    va="center", fontsize=9.5, color=INK["muted"])

    ax.axvline(0, color=INK["axis"], lw=1.0)
    ax.set_yticks(np.arange(len(ordered)), ordered.index)
    ax.tick_params(axis="y", labelcolor=INK["secondary"], labelsize=10)
    ax.grid(axis="x", color=INK["grid"], lw=0.7)
    ax.grid(axis="y", visible=False)
    finish_axes(ax, title="Correlation of federal R&D with four candidate series",
                subtitle="Hollow: on levels.  Filled: on annual real growth, which "
                         "removes the shared trend.  Grey fill: not significant. "
                         "Only the budget survives.",
                xlabel="Pearson correlation with total real federal R&D")
    ax.set_xlim(-0.35, 1.12)
    fig.tight_layout()
    save_figure(fig, "10_correlation_levels_vs_growth")


# ===========================================================================
# HYPOTHESIS 2 - THE BUDGET'S PULL IS CONCENTRATED, NOT SHARED
# ===========================================================================

AGENCIES = ["DOD", "HHS", "NIH", "NASA", "DOE", "NSF", "USDA",
            "Interior", "DOT", "EPA", "DOC", "VA"]


def test_hypothesis_2(t: Totals, aggregates: dict) -> pd.Series:
    hypothesis(
        2,
        "The aggregate's response to the budget is carried by a few agencies, not "
        "shared across them. If every agency rode the budget, each should show a "
        "similar elasticity and each should contribute volatility in proportion "
        "to its size. If instead the response is concentrated, most agencies will "
        "show no budget relationship at all and one will dominate the aggregate's "
        "year-to-year movement.",
        "Estimate the same elasticity agency by agency with 95% intervals, then "
        "decompose the variance of the annual change in total R&D into each "
        "agency's covariance share. The shares sum to one, so they say directly "
        "how much of the headline's movement each agency supplies.",
        "'Federal R&D rose this year' is read as a statement about national "
        "research policy. If one agency supplies most of the movement, the "
        "sentence is mostly a statement about that agency - and the agencies "
        "whose budgets are set by something else are invisible in it.")

    e = t.enacted()
    budget = e.discretionary / e.deflator * 1_000
    def p_vs_one(fit: dict) -> float:
        t_stat = (fit["beta"] - 1) / fit["se"]
        return 2 * (1 - stats.t.cdf(abs(t_stat), fit["n"] - 2))

    print(f"\n  Defense R&D beta = {aggregates['Defense R&D']['beta']:.2f} is "
          "statistically indistinguishable from riding the budget one-for-one "
          f"(p = {p_vs_one(aggregates['Defense R&D']):.2f}); nondefense beta = "
          f"{aggregates['Nondefense R&D']['beta']:.2f} is decisively below 1 and "
          f"explains only {100 * aggregates['Nondefense R&D']['r2']:.0f}% of its own "
          "variance that way.")

    # The agency workbook's constant-dollar base is FY2024 rather than FY2017,
    # which is irrelevant here: a fixed rescaling cancels in log differences.
    agencies = load_aaas_workbook("Agencies")
    print(f"\n  Agency-level elasticity to the discretionary budget "
          f"(Agencies.xlsx, FY{FY_START}-FY{FY_END}):")
    rows = {}
    for agency in AGENCIES:
        fit = elasticity(agencies.loc[agency], budget)
        series = agencies.loc[agency].loc[FY_START:FY_END].dropna()
        rows[agency] = fit | {
            "sd_yoy_%": 100 * growth(series).std(ddof=1),
            "real_chg_%": 100 * (series[FY_END] / series[FY_START] - 1)}
    table = pd.DataFrame(rows).T
    table["significant"] = table["p"] < ALPHA
    print(table[["beta", "se", "r2", "p", "sd_yoy_%", "real_chg_%",
                 "significant"]].to_string())

    sig = table[table["significant"]].index.tolist()
    finding(f"Only {len(sig)} of {len(AGENCIES)} agencies - {', '.join(sig)} - move "
            "with the budget at all. For the other nine the coefficient is "
            "indistinguishable from zero.")

    # Variance decomposition. NIH is reported inside HHS, so including both
    # would double-count it and push the shares past 100%.
    total = agencies.loc["Total R&D"].loc[FY_START:FY_END]
    d_total = total.diff().dropna()
    parts = {}
    for agency in agencies.index:
        if agency in {"Total R&D", "Defense", "Nondefense", "All Other", "NIH"}:
            continue
        change = agencies.loc[agency].loc[FY_START:FY_END].diff().reindex(d_total.index)
        parts[agency] = float(np.cov(change.fillna(0), d_total)[0, 1]
                              / np.var(d_total, ddof=1))
    share = pd.Series(parts).sort_values(ascending=False)
    level = agencies.loc[share.index, FY_END] / agencies.loc["Total R&D", FY_END] * 100

    print(f"\n  Share of the variance in annual change in total federal R&D "
          f"(NIH folded into HHS to avoid double counting):")
    print(pd.DataFrame({"variance_share_%": 100 * share,
                        f"level_share_FY{FY_END}_%": level}).round(1).head(8).to_string())
    print(f"  (all rows sum to {100 * share.sum():.0f}%)")

    top = share.index[0]
    finding(f"{top} absorbs {100 * share[top]:.0f}% of the aggregate's year-to-year "
            f"variance while being {level[top]:.0f}% of its level; {top} plus "
            f"{share.index[1]} account for {100 * share.iloc[:2].sum():.0f}%. Every "
            "other agency contributes LESS volatility than its funding share "
            "implies. As a time series, the federal R&D total is largely a "
            "defense-procurement series with a research programme attached.")

    shrank = {a: 100 * (agencies.loc[a, FY_END] / agencies.loc[a, FY_START] - 1)
              for a in AGENCIES
              if agencies.loc[a, FY_END] < agencies.loc[a, FY_START]}
    print(f"\n  Corollary - real change FY{FY_START}-FY{FY_END}. Total R&D "
          f"{100 * (total[FY_END] / total[FY_START] - 1):+.0f}%, but four agencies "
          "went backwards:")
    for agency, change in sorted(shrank.items(), key=lambda kv: kv[1]):
        print(f"    {agency:<10} {change:+6.0f}%   "
              f"(budget elasticity p = {table.loc[agency, 'p']:.2f})")
    note("None of these four responds to the budget cycle, so none is a "
         "budget-cycle casualty. They lost steadily inside a total that grew - a "
         "divergence the headline averages away just as surely as it averages "
         "away the shift from development to research.")
    note(f"BASIS NOTE  This workbook reports a "
         f"{100 * (total[FY_END] / total[FY_START] - 1):.0f}% real rise against "
         "main.py's +102%, because it uses a constant FY2024 base and the "
         "post-FY2017 R&D definition, under which late-stage defense testing left "
         "the R&D total. Shares and growth rates are unaffected.")

    plot_elasticity(aggregates, table)
    plot_variance_share(share, level)
    return share


def plot_elasticity(aggregates: dict, table: pd.DataFrame) -> None:
    """FIG 11 - elasticity coefficients with 95% intervals."""
    ordered = table.sort_values("beta")
    keys = ["Nondefense R&D", "Defense R&D", "Total R&D"]
    labels = list(ordered.index) + [""] + keys
    betas = list(ordered["beta"]) + [np.nan] + [aggregates[k]["beta"] for k in keys]
    cis = list(ordered["ci"]) + [(np.nan, np.nan)] + [aggregates[k]["ci"] for k in keys]
    sig = list(ordered["significant"]) + [False] + [True, True, True]

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    for i, (beta, (lo, hi), is_sig) in enumerate(zip(betas, cis, sig)):
        if np.isnan(beta):
            continue
        color = PALETTE["rd"] if is_sig else PALETTE["ns"]
        ax.plot([lo, hi], [i, i], color=color, lw=2.0, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.scatter(beta, i, s=58, color=color, zorder=3,
                   edgecolor=INK["surface"], linewidth=2)

    ax.axvline(0, color=INK["axis"], lw=1.0)
    ax.axvline(1, color=INK["secondary"], lw=1.2, ls=(0, (3, 2)))
    ax.annotate("rides the budget one-for-one", (1, len(labels) - 0.3),
                textcoords="offset points", xytext=(6, 0), ha="left", va="center",
                fontsize=9, color=INK["secondary"])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="y", labelcolor=INK["secondary"], labelsize=10)
    ax.grid(axis="x", color=INK["grid"], lw=0.7)
    ax.grid(axis="y", visible=False)
    finish_axes(ax, title="Elasticity of R&D growth to discretionary budget growth, "
                          "by agency",
                subtitle="Annual real growth, FY1976-FY2024, with 95% intervals. Grey: "
                         "interval spans zero. NIH sits inside HHS.",
                xlabel="Elasticity (percent of R&D per percent of budget)")
    ax.set_ylim(-0.8, len(labels) + 0.3)
    ax.set_xlim(None, 1.75)
    fig.tight_layout()
    save_figure(fig, "11_budget_elasticity_by_agency")


def plot_variance_share(share: pd.Series, level: pd.Series) -> None:
    """FIG 12 - variance share against funding share, per agency.

    Paired bars rather than a slope chart: a slope encodes the direction of
    the gap in one mark, but with six agencies whose values run from 71% down
    to 1% the small lines pile up on each other. Bars keep every agency
    readable at its own magnitude, which is what this comparison needs.
    """
    shown = share.head(6)
    gaps = 100 * shown - level[shown.index]
    others = gaps.drop(gaps.idxmax()).abs().max()
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = np.arange(len(shown))
    height = 0.36

    ax.barh(y + height / 2, 100 * shown.values, height=height,
            color=PALETTE["rd"], zorder=3, label="Share of year-to-year variance")
    ax.barh(y - height / 2, level[shown.index].values, height=height,
            color=INK["muted"], zorder=3, label=f"Share of FY{FY_END} funding")
    for i, agency in enumerate(shown.index):
        ax.annotate(f"{100 * shown[agency]:.0f}%",
                    (100 * shown[agency] + 1, i + height / 2), ha="left", va="center",
                    fontsize=9.5, fontweight="semibold", color=PALETTE["rd"])
        ax.annotate(f"{level[agency]:.0f}%", (level[agency] + 1, i - height / 2),
                    ha="left", va="center", fontsize=9.5, color=INK["secondary"])

    ax.set_yticks(y, shown.index)
    ax.tick_params(axis="y", labelcolor=INK["secondary"], labelsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color=INK["grid"], lw=0.7)
    ax.grid(axis="y", visible=False)
    finish_axes(ax, title="Share of federal R&D funding vs share of its year-to-year "
                          "variance, by agency",
                subtitle=f"FY{FY_START}-FY{FY_END}. DOD supplies {gaps.max():.0f} points "
                         "more variance than funding; no other agency differs from its "
                         f"own funding share by more than {others:.0f}.",
                xlabel="Percent")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_pct0))
    ax.set_xlim(0, 80)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "12_funding_share_vs_variance_share")


# ===========================================================================
# HYPOTHESIS 3 - THE BUDGET CYCLE REACHES DEVELOPMENT, NOT RESEARCH
# ===========================================================================

# DOD's own research categories map onto the character-of-work split in
# main.py Section 5, so they inherit its colours.
DOD_TIERS = {"Basic Research (6.1)": "Basic",
             "Applied Research (6.2)": "Applied",
             "Advanced Technology Development (6.3)": "Developmental"}


def test_hypothesis_3(t: Totals) -> None:
    hypothesis(
        3,
        "The budget volatility that DOD transmits lands on late-stage "
        "development, not on science. If the cycle hit the portfolio "
        "proportionally, DOD's Science and Technology account would ride DOD's "
        "total R&D and would itself track the budget. If research is protected, "
        "S&T should do neither.",
        "Estimate the elasticity of DOD S&T to DOD R&D overall and to the "
        "discretionary budget, then compare real growth across DOD's three "
        "research tiers (6.1 basic, 6.2 applied, 6.3 advanced technology "
        "development) over FY1990-FY2024.",
        "main.py found the headline is more volatile than researchers' experience "
        "suggests. If the volatility is concentrated in development, that "
        "explains why: the swings in the headline are largely procurement, and "
        "reading them as instability in the science base overstates the case. It "
        "also tests whether the national shift from development to research is "
        "real or an artefact of mixing agencies.")

    agencies = load_aaas_workbook("Agencies")
    st = load_aaas_workbook("DODST")
    budget = t.enacted().discretionary / t.enacted().deflator * 1_000

    lo, hi = 1990, FY_END          # DODST starts in FY1990
    dod = agencies.loc["DOD"].loc[lo:hi]
    science = st.loc["Total DOD S&T plus Medical"].loc[lo:hi] * 1_000   # $B -> $M

    share = 100 * science / dod
    print(f"\n  DOD S&T as a share of DOD R&D: {share[lo]:.1f}% in FY{lo}, "
          f"{share[hi]:.1f}% in FY{hi} (min {share.min():.1f}%, max {share.max():.1f}%)")
    to_dod = elasticity(science, dod, lo, hi)
    to_budget = elasticity(science, budget, lo, hi)
    print(f"  elasticity of DOD S&T to DOD R&D overall:    beta={to_dod['beta']:+.2f} "
          f"[{to_dod['ci'][0]:+.2f}, {to_dod['ci'][1]:+.2f}]  R2={to_dod['r2']:.2f}  "
          f"p={to_dod['p']:.3f}")
    print(f"  elasticity of DOD S&T to the budget:         "
          f"beta={to_budget['beta']:+.2f} "
          f"[{to_budget['ci'][0]:+.2f}, {to_budget['ci'][1]:+.2f}]  "
          f"p={to_budget['p']:.3f}")
    finding(f"S&T moves only {to_dod['beta']:.2f} percent for each percent DOD R&D "
            f"moves, and does not track the discretionary budget at all "
            f"(p = {to_budget['p']:.2f}). The research end of the agency that "
            "transmits the budget cycle to the national total is itself insulated "
            "from it.")

    print(f"\n  Within DOD S&T, FY{lo} -> FY{hi} (constant FY2024 $B):")
    for row in DOD_TIERS:
        series = st.loc[row].loc[lo:hi]
        print(f"    {row:<40} {series[lo]:5.2f} -> {series[hi]:5.2f}  "
              f"({100 * (series[hi] / series[lo] - 1):+6.0f}%, {cagr(series):+.2f}%/yr)")
    print(f"    {'DOD R&D overall':<40} "
          f"{100 * (dod[hi] / dod[lo] - 1):+6.0f}%")
    finding("Basic and applied research grew while advanced technology development "
            "shrank - the same reallocation main.py found nationally, repeating "
            "one level down inside the single agency that dominates the total. The "
            "shift from development toward research is therefore not an artefact "
            "of mixing agencies together.")

    plot_dod_tiers(st, dod, lo, hi)


def plot_dod_tiers(st: pd.DataFrame, dod: pd.Series, lo: int, hi: int) -> None:
    """FIG 13 - DOD's research tiers against DOD R&D overall."""
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.axhline(100, color=INK["axis"], lw=1.2)
    for row, tier in DOD_TIERS.items():
        series = index_to_base(st.loc[row].loc[lo:hi], lo)
        ax.plot(series.index, series.values, color=PALETTE[tier], lw=2.0, label=row)
        label_series_end(ax, series, PALETTE[tier], f"{series[hi]:.0f}")
    overall = index_to_base(dod, lo)
    ax.plot(overall.index, overall.values, color=PALETTE["aggregate"], lw=2.0,
            ls=(0, (4, 2)), label="DOD R&D overall")
    label_series_end(ax, overall, PALETTE["aggregate"], f"{overall[hi]:.0f}")

    finish_axes(ax, title=f"DOD research tiers and total DOD R&D, indexed to FY{lo} = 100",
                subtitle="Constant FY2024 dollars. Basic and applied research grew; "
                         "advanced technology development shrank.",
                ylabel=f"Index (FY{lo} = 100)", yfmt=fmt_plain, xtick_step=5)
    ax.set_xlim(lo, hi + 4)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    save_figure(fig, "13_dod_research_tiers")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    apply_chart_style()
    header("WHAT DOES FEDERAL R&D FUNDING TRACK?")
    note("Companion to main.py, organised as three hypotheses. Every relationship "
         "is tested on annual real growth, not on levels, because trending series "
         "correlate whether or not they are related.")

    totals = load_totals()
    aggregates = test_hypothesis_1(totals)
    test_hypothesis_2(totals, aggregates)
    test_hypothesis_3(totals)

    header("WHAT THE THREE HYPOTHESES ADD UP TO")
    note("H1  Federal R&D moves with the appropriations pot and with nothing else "
         "tested - not GDP, not industry R&D, not university R&D. The "
         "percent-of-GDP framing and the crowding-out debate both describe "
         "levels, not annual behaviour.")
    note("H2  That budget sensitivity is concentrated in three agencies, and DOD "
         "alone supplies 71% of the aggregate's year-to-year variance on 51% of "
         "its funding. Nine agencies answer to something else entirely, and four "
         "shrank in real terms regardless of the budget.")
    note("H3  Inside DOD, the budget cycle reaches development, not science: S&T "
         "is insulated from both DOD's own swings and the budget's, and basic and "
         "applied research grew while advanced technology development shrank.")
    note("TOGETHER  The headline federal R&D number is a budget-negotiation "
         "outcome dominated by one agency's procurement cycle. Reading it as a "
         "measure of national research effort attributes to science a series that "
         "is mostly about something else - which is the same conclusion main.py "
         "reached from composition, arrived at here from co-movement.")

    header("DONE")
    note("4 figures (10-13) in figures/. Agencies.xlsx, DODST.xlsx and the Funder "
         "sheet are loaded through load_aaas_workbook() and funder_series(); "
         "DOE.xlsx, NIHbud.xlsx and Disc1.xlsx share the same layout and can be "
         "read with load_aaas_workbook() unchanged.")


if __name__ == "__main__":
    main()
