"""
U.S. Federal R&D Funding - Exploratory Data Analysis
====================================================

Source
------
`data/Historical Data.xlsx` - the AAAS Historical Trends in Federal R&D data
file (last updated 2025-10-23). Internal series are compiled by the AAAS R&D
Budget and Policy Program from OMB and appropriations documents; the GDP
deflator, GDP, and total discretionary budget authority benchmarks come from
OMB Historical Tables 10.1, 10.2, and 8.1. Dollar figures are millions of
current dollars unless deflated.

Guiding question
----------------
Headline R&D dollars conflate four different things: price change, the size of
the economy, the size of the budget R&D competes inside, and what the money
actually buys. This script shows that **nominal R&D spending alone gives a
misleading picture of federal R&D investment**, and that correcting it needs a
deflator, a denominator, AND a breakdown.

  Section 1  Load, reshape, validate.
  Section 2  Long-run trend in current and constant dollars.      [fig 1]
  Section 3  Where the trajectory changed, and by how much.       [fig 2]
  Section 4  R&D against GDP and the discretionary budget.        [fig 3]
  Section 5  Composition by character of work.              [figs 4, 5, 6]
  Section 6  Defense vs nondefense, levels and share.             [fig 7]
  Section 7  Synthesis: what the aggregate cannot say.         [figs 8, 9]

Structure
---------
Section 0 is generic. `load_wide_sheet` / `tidy_wide_sheet` handle any
"metadata columns + one column per fiscal year" sheet - the layout of the
Totals, Type, Field, and Funder sheets here and of every other workbook in
`data/`. Later sections can add sources and reuse `to_constant`,
`detect_breakpoints`, `segment_table`, and the plotting helpers unchanged.

Run:  python main.py
"""

from __future__ import annotations

# --- Standard library -------------------------------------------------------
import itertools
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

# --- Scientific stack -------------------------------------------------------
import numpy as np
import pandas as pd

# --- Plotting ---------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns

# --- Statistics -------------------------------------------------------------
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


# ===========================================================================
# SECTION 0 - CONFIGURATION AND REUSABLE WORKFLOW
# ===========================================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"

DATA_FILES = {
    "historical": DATA_DIR / "Historical Data.xlsx",
    "agencies": DATA_DIR / "Agencies.xlsx",
    "dod_st": DATA_DIR / "DODST.xlsx",
    "doe": DATA_DIR / "DOE.xlsx",
    "def_nondef": DATA_DIR / "DefNon.xlsx",
    "discretionary": DATA_DIR / "Disc1.xlsx",
    "nih_budget": DATA_DIR / "NIHbud.xlsx",
}

BASE_YEAR = 2017                 # constant-dollar base used by the AAAS file
FY_START, FY_END = 1976, 2024    # enacted appropriations only
FY_REQUEST = (2025, 2026)        # FY2026 President's Budget Request
MIN_SEGMENT = 5                  # shortest regime the breakpoint search accepts

# --- Colour ----------------------------------------------------------------
# Eight validated categorical slots on a light surface, plus chart chrome.
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
       "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SLOT = {1: "#2a78d6",   # blue
        2: "#eb6834",   # orange
        3: "#1baf7a",   # aqua
        5: "#e87ba4",   # magenta
        7: "#4a3aa7",   # violet
        8: "#e34948"}   # red

# Colour is assigned to MEANING, once, and never reassigned. Two families run
# through the analysis and are never drawn in the same figure, so they may
# share slots:
#
#   Family A - how the same dollars are MEASURED (sections 2-4, 7)
#   Family B - what the dollars are FOR          (sections 5-7)
#
# Within a family every entity has its own slot, so no figure ever repeats a
# colour. The aggregate is never a hue: it is ink, dashed, so a reader can
# always tell the headline from a component.
PALETTE = {
    # Family A
    "current": SLOT[2], "constant": SLOT[1], "gdp_share": SLOT[8],
    "disc_share": SLOT[3],
    # Family B
    "Basic": SLOT[1], "Applied": SLOT[2], "Developmental": SLOT[3],
    "defense": SLOT[7], "nondefense": SLOT[5],
    # Roles
    "aggregate": INK["primary"], "fit": INK["primary"],
    "up": SLOT[1], "down": SLOT[8],    # diverging sign (blue<->red, never green/red)
}

TYPE_LABELS = {"Basic": "Basic research", "Applied": "Applied research",
               "Developmental": "Development"}


def apply_chart_style() -> None:
    """Install the chart chrome once: recessive grid and axes, ink-coloured text."""
    sns.set_theme(style="white")
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 9.5, "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": INK["grid"], "grid.linewidth": 0.7,
        "text.color": INK["primary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def finish_axes(ax, *, title=None, subtitle=None, ylabel=None, xlabel=None,
                yfmt=None, xtick_step=None):
    """Apply the shared axis treatment: title block, spines, formatters."""
    if title:
        # Reserve room above the axes so the subtitle cannot collide with it.
        ax.set_title(title, pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9, color=INK["secondary"])
    ax.set_ylabel(ylabel or "")
    ax.set_xlabel(xlabel or "")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK["axis"])
    ax.spines["bottom"].set_color(INK["axis"])
    ax.set_axisbelow(True)
    if yfmt:
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    if xtick_step:
        ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
    return ax


# `D` is an escaped dollar sign: matplotlib reads an unescaped pair as math mode.
D = r"\$"
fmt_billions = lambda v, _=None: f"{D}{v / 1e3:,.0f}B"
fmt_pct2 = lambda v, _=None: f"{v:.2f}%"
fmt_pct0 = lambda v, _=None: f"{v:.0f}%"
fmt_plain = lambda v, _=None: f"{v:,.0f}"


def save_figure(fig, name: str) -> Path:
    """Write a figure to figures/<name>.png and report the path."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved  figures/{path.name}")
    return path


def label_series_end(ax, series: pd.Series, color: str, text: str, *, dx: int = 8):
    """Direct-label a line at its right-hand end, in the series' own colour."""
    ax.annotate(text, (series.index[-1], series.iloc[-1]),
                textcoords="offset points", xytext=(dx, 0), ha="left",
                va="center", fontsize=9.5, fontweight="semibold", color=color)


def mark_knots(ax, knots, *, y, labels: bool = True):
    """Draw the breakpoint years as vertical rules with boxed year labels."""
    for year in knots:
        ax.axvline(year, color=INK["secondary"], lw=1.2, ls=(0, (2, 2)), alpha=0.8)
        if labels:
            ax.annotate(f"FY{year}", (year, y), ha="center", va="center",
                        fontsize=9.5, color=INK["primary"], fontweight="semibold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=INK["surface"],
                                  edgecolor=INK["axis"], linewidth=0.8))


# --- Generic loaders -------------------------------------------------------

def year_columns(df: pd.DataFrame) -> list:
    """Return the columns of `df` that are fiscal years, in ascending order."""
    years = []
    for col in df.columns:
        try:
            value = int(float(col))
        except (TypeError, ValueError):
            continue
        if 1900 <= value <= 2100:
            years.append(col)
    return sorted(years, key=lambda c: int(float(c)))


def load_wide_sheet(file_key: str, sheet: str, *, header: int = 0) -> pd.DataFrame:
    """Load a wide 'one column per fiscal year' sheet as-is."""
    path = DATA_FILES[file_key]
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    return pd.read_excel(path, sheet_name=sheet, header=header)


def tidy_wide_sheet(df: pd.DataFrame, id_cols=None, *, value_name="value") -> pd.DataFrame:
    """Melt a year-wide frame into tidy rows of (id columns..., fy, value).

    The canonical long format for any new data source added later.
    """
    years = year_columns(df)
    if id_cols is None:
        id_cols = [c for c in df.columns if c not in years]
    tidy = df.melt(id_vars=id_cols, value_vars=years,
                   var_name="fy", value_name=value_name)
    tidy["fy"] = tidy["fy"].astype(float).astype(int)
    tidy[value_name] = pd.to_numeric(tidy[value_name], errors="coerce")
    return tidy.dropna(subset=[value_name]).sort_values(["fy", *id_cols], ignore_index=True)


def pick_series(df: pd.DataFrame, *, label_col=None, **match) -> pd.Series:
    """Extract one year-indexed series from a wide frame by matching metadata."""
    mask = pd.Series(True, index=df.index)
    if label_col is not None:
        mask &= df.iloc[:, 0].astype(str).str.strip() == label_col
    for col, value in match.items():
        mask &= df[col].astype(str).str.strip() == str(value)
    rows = df.loc[mask]
    if rows.empty:
        raise KeyError(f"no row matched label={label_col!r} {match}")
    series = rows.iloc[0][year_columns(df)].astype(float)
    series.index = [int(float(y)) for y in series.index]
    return series.rename(label_col or "series")


# --- Deflation and growth statistics ---------------------------------------

def to_constant(current: pd.Series, deflator: pd.Series) -> pd.Series:
    """Convert a current-dollar series to constant BASE_YEAR dollars."""
    return (current / deflator.reindex(current.index)).rename(
        f"{current.name} (FY{BASE_YEAR}$)")


def cagr(series: pd.Series, start: int | None = None, end: int | None = None) -> float:
    """Compound annual growth rate, percent per year, between two years."""
    start = start if start is not None else int(series.index.min())
    end = end if end is not None else int(series.index.max())
    if end - start <= 0 or series[start] <= 0:
        return np.nan
    return 100 * ((series[end] / series[start]) ** (1 / (end - start)) - 1)


def index_to_base(series: pd.Series, base_year: int) -> pd.Series:
    """Rescale a series so that `base_year` equals 100."""
    return (100 * series / series[base_year]).rename(series.name)


def loglinear_trend(series: pd.Series) -> dict:
    """OLS of log(value) on year: exponential trend rate with inference."""
    reg = stats.linregress(series.index.values.astype(float), np.log(series.values))
    return {"cagr_pct": 100 * (np.exp(reg.slope) - 1),
            "se_pct": 100 * reg.stderr * np.exp(reg.slope),
            "r2": reg.rvalue ** 2, "p": reg.pvalue}


def fit_piecewise(series: pd.Series, knots, *, log: bool = True) -> dict:
    """Fit a continuous piecewise-linear trend; return the fit and its slopes.

    `log=True` models log(value), so each segment slope is a compound growth
    rate in percent per year - the right form for a dollar series that
    compounds. `log=False` models the value directly, so slopes are units per
    year; that is the right form for a SHARE, which is bounded by 0 and 100
    and does not compound.
    """
    x = series.index.values.astype(float)
    y = np.log(series.values) if log else series.values.astype(float)
    design = np.column_stack([np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in knots])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    slopes = np.cumsum(beta[1:])          # slope of each successive segment
    return {
        "knots": list(knots), "log": log, "slopes": slopes,
        "rates": [100 * (np.exp(v) - 1) for v in slopes] if log else list(slopes),
        "rate_unit": "%/yr" if log else "pp/yr",
        "sse": float(((y - fitted) ** 2).sum()), "n_params": design.shape[1],
        "fitted": pd.Series(np.exp(fitted) if log else fitted,
                            index=series.index, name="fitted"),
    }


def detect_breakpoints(series: pd.Series, n_breaks: int = 2,
                       min_segment: int = MIN_SEGMENT, *, log: bool = True) -> dict:
    """Grid-search the knot years that minimise SSE of a piecewise fit.

    The model is continuous at the knots, so each knot is a change in the RATE
    of change - a bend in the trajectory, not a jump in level.
    """
    years = series.index.values.astype(int)
    best = None
    for combo in itertools.combinations(range(min_segment, len(years) - min_segment),
                                        n_breaks):
        if any(b - a < min_segment for a, b in zip(combo, combo[1:])):
            continue
        fit = fit_piecewise(series, [float(years[i]) for i in combo], log=log)
        if best is None or fit["sse"] < best["sse"]:
            best = fit
    if best is None:
        raise ValueError("series too short for the requested number of breaks")
    return best


def segment_table(series: pd.Series, knots, *, log: bool = True) -> pd.DataFrame:
    """Per-regime descriptive statistics for the periods defined by `knots`.

    `log=True` describes a dollar series in growth rates; `log=False`
    describes a share in percentage points.
    """
    edges = [int(series.index.min())] + [int(k) for k in knots] + [int(series.index.max())]
    growth = series.pct_change() * 100
    rows = []
    for start, end in zip(edges, edges[1:]):
        window = series.loc[start:end]
        slope = stats.linregress(window.index.values.astype(float),
                                 np.log(window.values) if log else window.values)
        row = {"period": f"{start}-{end}", "years": end - start,
               "start": window[start], "end": window[end]}
        if log:
            yoy = growth.loc[start + 1:end]
            row |= {"total_chg_%": 100 * (window[end] / window[start] - 1),
                    "cagr_%/yr": cagr(window),
                    "trend_%/yr": 100 * (np.exp(slope.slope) - 1),
                    "sd_yoy_%": yoy.std(ddof=1), "yrs_down": int((yoy < 0).sum())}
        else:
            row |= {"change_pp": window[end] - window[start],
                    "trend_pp/yr": slope.slope,
                    "min": window.min(), "max": window.max()}
        row["trend_r2"] = slope.rvalue ** 2
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


def chow_f_test(series: pd.Series, knots, *, log: bool = True) -> dict:
    """F test of the piecewise model against a single straight trend."""
    restricted = fit_piecewise(series, [], log=log)
    full = fit_piecewise(series, knots, log=log)
    q = full["n_params"] - restricted["n_params"]
    df_resid = len(series) - full["n_params"]
    f_stat = ((restricted["sse"] - full["sse"]) / q) / (full["sse"] / df_resid)
    return {"F": f_stat, "df": (q, df_resid),
            "p": 1 - stats.f.cdf(f_stat, q, df_resid),
            "sse_reduction_%": 100 * (1 - full["sse"] / restricted["sse"])}


def opposite_direction_years(total: pd.Series, part: pd.Series) -> pd.Index:
    """Years where a component moved against the aggregate it belongs to.

    The direct measure of how much an aggregate can hide: when the signs
    disagree, the headline reports the opposite of what happened to that part.
    """
    both = pd.DataFrame({"total": total.diff(), "part": part.diff()}).dropna()
    return both[np.sign(both["total"]) != np.sign(both["part"])].index


# --- Containers and console helpers ----------------------------------------

@dataclass
class Totals:
    """National R&D aggregates and their benchmarks, year-indexed."""
    current: pd.Series        # total federal R&D, $M current
    constant: pd.Series       # total federal R&D, $M FY2017
    defense: pd.Series        # defense R&D, $M FY2017
    nondefense: pd.Series     # nondefense R&D, $M FY2017
    deflator: pd.Series       # GDP deflator, BASE_YEAR = 1.0
    gdp: pd.Series            # GDP in FY, $B
    discretionary: pd.Series  # total discretionary budget authority, $B
    arra: pd.Series           # FY2009 Recovery Act R&D, $M current

    @property
    def pct_gdp(self) -> pd.Series:
        return (100 * self.current / (self.gdp * 1_000)).rename("R&D as % of GDP")

    @property
    def pct_discretionary(self) -> pd.Series:
        return (100 * self.current / (self.discretionary * 1_000)).rename(
            "R&D as % of discretionary BA")

    @property
    def defense_share(self) -> pd.Series:
        return (100 * self.defense / (self.defense + self.nondefense)).rename(
            "Defense share")

    def enacted(self) -> "Totals":
        """Restrict every series to FY1976-2024, dropping the request years."""
        return Totals(*[getattr(self, f).loc[FY_START:FY_END]
                        for f in self.__dataclass_fields__])


@dataclass
class TypeMix:
    """Federal R&D split by character of work, in constant BASE_YEAR dollars."""
    real: pd.DataFrame        # columns Basic / Applied / Developmental

    @property
    def total(self) -> pd.Series:
        return self.real.sum(axis=1).rename("All types")

    @property
    def share(self) -> pd.DataFrame:
        return 100 * self.real.div(self.total, axis=0)

    @property
    def research(self) -> pd.Series:
        return self.real[["Basic", "Applied"]].sum(axis=1).rename("Research")


def header(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def note(text: str) -> None:
    print(textwrap.fill(text, 78, initial_indent="  ", subsequent_indent="  "))


# ===========================================================================
# SECTION 1 - LOAD, RESHAPE, AND VALIDATE
# ===========================================================================

def load_totals() -> Totals:
    """Build the `Totals` container from the Historical Data workbook."""
    header("SECTION 1 - DATA LOADING AND VALIDATION")

    raw = load_wide_sheet("historical", "Totals")
    years = year_columns(raw)
    print(f"  Totals sheet: {raw.shape[0]} rows x {len(years)} fiscal years "
          f"(FY{int(float(years[0]))}-FY{int(float(years[-1]))})")
    print(f"  metrics present: {sorted(raw['Metric'].dropna().unique())}")

    get = lambda label, metric: pick_series(raw, label_col=label, Metric=metric)
    totals = Totals(
        current=(get("Defense", "Current Dollars")
                 + get("Nondefense", "Current Dollars")).rename("Total federal R&D"),
        constant=(get("Defense", f"Constant {BASE_YEAR}")
                  + get("Nondefense", f"Constant {BASE_YEAR}")).rename("Total federal R&D"),
        defense=get("Defense", f"Constant {BASE_YEAR}").rename("Defense R&D"),
        nondefense=get("Nondefense", f"Constant {BASE_YEAR}").rename("Nondefense R&D"),
        deflator=pick_series(raw, label_col="GDP Deflator").dropna(),
        gdp=pick_series(raw, label_col="GDP in FY").dropna(),
        discretionary=pick_series(
            raw, label_col="Total Discretionary Budget Authority").dropna(),
        arra=(pick_series(raw, label_col="ARRA Defense", Metric="Current Dollars").fillna(0)
              + pick_series(raw, label_col="ARRA NonDefense",
                            Metric="Current Dollars").fillna(0)),
    )

    max_err = float(((to_constant(totals.current, totals.deflator) - totals.constant).abs()
                     / totals.constant).max())
    print(f"\n  CHECK deflation: current/deflator reproduces the workbook's "
          f"constant-dollar series (max relative error {max_err:.1e})")
    assert max_err < 1e-9, "constant-dollar series does not match the deflator"
    print(f"  CHECK base year: deflator[FY{BASE_YEAR}] = {totals.deflator[BASE_YEAR]:.4f}")
    print(f"  CHECK coverage: {totals.current.loc[FY_START:FY_END].isna().sum()} "
          f"missing years in FY{FY_START}-FY{FY_END}")
    note(f"CAVEAT FY{FY_REQUEST[0]}-FY{FY_REQUEST[1]} come from the FY2026 President's "
         "Budget Request, not enacted appropriations. Every statistic below is "
         f"estimated on FY{FY_START}-FY{FY_END}; request years appear only as dashed "
         "segments in figure 1.")
    note(f"CAVEAT FY2009 Recovery Act R&D (${totals.arra[2009]:,.0f}M) is reported "
         "separately by AAAS and excluded from these totals, so the 2009-10 "
         "stimulus surge is understated.")
    return totals


def load_type_mix(t: Totals) -> TypeMix:
    """Load the Type sheet and reconcile it against the Section 1 aggregate."""
    raw = load_wide_sheet("historical", "Type")
    mix = TypeMix(real=pd.DataFrame({
        kind: pick_series(raw, label_col="All", Type=kind, Metric=f"Constant {BASE_YEAR}")
        for kind in TYPE_LABELS}).loc[FY_START:FY_END])

    # The workbook's "Research" line should be basic + applied; confirm before
    # relying on the three categories being exhaustive and non-overlapping.
    stated = pick_series(raw, label_col="All", Type="Research",
                         Metric=f"Constant {BASE_YEAR}").loc[FY_START:FY_END]
    err = float((stated - mix.research).abs().max())
    print(f"  CHECK additivity: Research == Basic + Applied (max abs error ${err:,.3f}M)")
    assert err < 1e-6, "Research is not the sum of basic and applied"

    gap = 100 * (mix.total / t.enacted().constant - 1)
    print(f"  CHECK reconciliation: Type-sheet totals run {gap.mean():+.1f}% against the "
          f"Section 1 aggregate (range {gap.min():+.1f}% to {gap.max():+.1f}%)")
    note("The two sheets rest on different OMB reporting bases, so type totals do "
         "not equal the Section 1 totals. Shares are therefore computed WITHIN the "
         "Type sheet, never against the Section 1 aggregate.")
    return mix


# ===========================================================================
# SECTION 2 - LONG-TERM TREND: CURRENT VS CONSTANT DOLLARS
# ===========================================================================

def analyze_long_term_trend(t: Totals) -> None:
    header("SECTION 2 - LONG-TERM TREND IN CURRENT AND CONSTANT DOLLARS")
    e = t.enacted()

    print(pd.DataFrame({"current $M": e.current,
                        f"constant FY{BASE_YEAR} $M": e.constant,
                        "deflator": e.deflator})
          .loc[[FY_START, 1990, 2000, 2010, 2020, FY_END]].to_string())

    nominal = e.current[FY_END] / e.current[FY_START]
    real = e.constant[FY_END] / e.constant[FY_START]
    price = e.deflator[FY_END] / e.deflator[FY_START]
    print(f"\n  FY{FY_START} -> FY{FY_END}")
    print(f"    current dollars   ${e.current[FY_START]:,.0f}M -> ${e.current[FY_END]:,.0f}M"
          f"   {100 * (nominal - 1):+,.0f}%  ({cagr(e.current):+.2f}%/yr)")
    print(f"    constant dollars  ${e.constant[FY_START]:,.0f}M -> ${e.constant[FY_END]:,.0f}M"
          f"   {100 * (real - 1):+,.0f}%  ({cagr(e.constant):+.2f}%/yr)")
    # The nominal multiple factors exactly into a real multiple times a price
    # multiple - the cleanest statement of how much of the headline is inflation.
    print(f"    the headline factors exactly: {nominal:.2f}x nominal = "
          f"{real:.2f}x real x {price:.2f}x prices, so "
          f"{100 * np.log(price) / np.log(nominal):.0f}% of the nominal growth is "
          "price change, not added research capacity")

    peak = int(e.constant.idxmax())
    print(f"\n  real peak: FY{peak} at ${e.constant[peak]:,.0f}M; FY{FY_END} sits "
          f"{100 * (e.constant[FY_END] / e.constant[peak] - 1):+.1f}% below it after "
          f"{FY_END - peak} years")
    for label, series in [("current", e.current), ("constant", e.constant)]:
        trend = loglinear_trend(series)
        print(f"  log-linear trend, {label:<9} {trend['cagr_pct']:+.2f} "
              f"+/- {trend['se_pct']:.2f} %/yr   (R2={trend['r2']:.3f})")

    plot_current_vs_constant(t)


def plot_current_vs_constant(t: Totals) -> None:
    """FIG 1 - two panels on separate axes; never one chart with two scales."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    top = max(t.current.max(), t.constant.max()) * 1.15

    for ax, series, role, panel in [
        (axes[0], t.current, "current", "Current dollars"),
        (axes[1], t.constant, "constant", f"Constant FY{BASE_YEAR} dollars"),
    ]:
        color = PALETTE[role]
        enacted = series.loc[:FY_END]
        ax.plot(enacted.index, enacted.values, color=color, lw=2.0)
        ax.fill_between(enacted.index, 0, enacted.values, color=color, alpha=0.10, lw=0)
        # The request years are drawn dashed so they cannot be read as enacted.
        request = series.loc[FY_END:]
        ax.plot(request.index, request.values, color=color, lw=2.0,
                ls=(0, (3, 2)), alpha=0.85)
        for year in (FY_START, FY_END):
            ax.scatter([year], [enacted[year]], s=28, color=color, zorder=5,
                       edgecolor=INK["surface"], linewidth=2)
            ax.annotate(f"{D}{enacted[year] / 1e3:,.0f}B", (year, enacted[year]),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, fontweight="semibold", color=INK["primary"])
        finish_axes(ax, title=panel, ylabel="Total federal R&D",
                    yfmt=fmt_billions, xtick_step=10)
        ax.set_ylim(0, top)

    fig.suptitle("Total federal R&D by fiscal year, current and constant dollars",
                 x=0.005, ha="left", fontsize=14, fontweight="semibold", y=1.05)
    fig.text(0.005, 0.99, "FY1976-FY2026, both panels on the same y-axis range: "
             "nominal R&D grew eightfold, real R&D roughly doubled. Dashed segments "
             "are the FY2026 budget request, not enacted appropriations.",
             ha="left", fontsize=9.5, color=INK["secondary"])
    fig.tight_layout()
    save_figure(fig, "01_current_vs_constant")


# ===========================================================================
# SECTION 3 - WHERE THE TRAJECTORY CHANGED
# ===========================================================================

def analyze_trajectory_changes(t: Totals) -> dict:
    header("SECTION 3 - TWO TRAJECTORY CHANGES, DETECTED AND QUANTIFIED")
    real = t.enacted().constant

    note("Method: fit a continuous piecewise log-linear trend to real R&D and "
         "grid-search the two knot years that minimise residual sum of squares. "
         "Because the fit is continuous, each knot is a change in the growth "
         "RATE - a bend in the trajectory, not a one-year jump.")

    fit = detect_breakpoints(real, n_breaks=2)
    knots = [int(k) for k in fit["knots"]]
    test = chow_f_test(real, fit["knots"])
    print(f"\n  detected breakpoints: FY{knots[0]} and FY{knots[1]}")
    print("  segment trend rates (real): "
          + "  ->  ".join(f"{r:+.2f}%/yr" for r in fit["rates"]))
    print(f"  F test vs a single trend: F({test['df'][0]},{test['df'][1]}) = "
          f"{test['F']:.1f}, p = {test['p']:.2e}; residual variance falls "
          f"{test['sse_reduction_%']:.0f}%")
    print(f"\n  Regime statistics (constant FY{BASE_YEAR} dollars):")
    print(segment_table(real, fit["knots"]).to_string())

    for label, rate in zip(
            [f"FY{FY_START}-{knots[0]}  Steady post-Vietnam expansion",
             f"FY{knots[0]}-{knots[1]}  Surge: NIH doubling, post-9/11 defense R&D",
             f"FY{knots[1]}-{FY_END}  Plateau and real erosion"], fit["rates"]):
        print(f"    {label}  ({rate:+.2f}%/yr real)")

    note(f"The FY{knots[0]} bend ACCELERATED real growth by "
         f"{fit['rates'][1] - fit['rates'][0]:+.1f} points per year; the FY{knots[1]} "
         f"bend REVERSED it by {fit['rates'][2] - fit['rates'][1]:+.1f} points per year. "
         "The second is the larger break, and it is invisible in nominal dollars, "
         f"which kept rising at {cagr(t.enacted().current, knots[1], FY_END):+.2f}%/yr "
         "across the same stretch.")

    plot_trajectory(t, fit, knots)
    return {"fit": fit, "knots": knots}


def plot_trajectory(t: Totals, fit: dict, knots: list) -> None:
    """FIG 2 - level with its piecewise fit, over year-over-year change."""
    real = t.enacted().constant
    yoy = (real.pct_change() * 100).dropna()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.6, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.12})
    bounds = [FY_START] + knots + [FY_END]

    ax_top.plot(real.index, real.values, color=PALETTE["constant"], lw=2.0,
                label="Actual")
    ax_top.plot(fit["fitted"].index, fit["fitted"].values, color=PALETTE["fit"],
                lw=1.6, ls=(0, (4, 2)), alpha=0.75, label="Piecewise log-linear fit")
    # Alternate regimes get a neutral wash - grey, not a hue, so the shading
    # delimits periods without implying a category.
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        if i % 2:
            for ax in (ax_top, ax_bot):
                ax.axvspan(start, end, color=INK["grid"], alpha=0.55, lw=0, zorder=0)
    mark_knots(ax_top, knots, y=real.min() * 0.22)
    # Each rate sits on its own segment of the fitted line, so the reader never
    # has to match a label in a header strip back to a period.
    for (start, end), rate in zip(zip(bounds, bounds[1:]), fit["rates"]):
        mid = (start + end) // 2
        # Anchor above whichever is higher there, the fit or the series itself.
        ax_top.annotate(f"{rate:+.1f}%/yr",
                        (mid, max(fit["fitted"][mid], real.loc[start:end].max())),
                        textcoords="offset points", xytext=(0, 14), ha="center",
                        va="bottom", fontsize=10.5, fontweight="semibold",
                        color=INK["primary"],
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=INK["surface"],
                                  edgecolor="none", alpha=0.85))
    finish_axes(ax_top, ylabel="Total federal R&D", yfmt=fmt_billions)
    ax_top.set_ylim(0, real.max() * 1.22)
    # Lower left: the only quadrant the series, the knot labels and the rate
    # band all leave empty.
    ax_top.legend(loc="lower left", ncols=2)

    ax_bot.bar(yoy.index, yoy.values, width=0.68, zorder=3,
               color=[PALETTE["up"] if v >= 0 else PALETTE["down"] for v in yoy.values])
    ax_bot.axhline(0, color=INK["axis"], lw=1.0)
    for start, end in zip(bounds, bounds[1:]):
        ax_bot.hlines(yoy.loc[start + 1:end].mean(), start, end,
                      color=INK["primary"], lw=1.8, zorder=4)
    for year in knots:
        ax_bot.axvline(year, color=INK["secondary"], lw=1.2, ls=(0, (2, 2)), alpha=0.8)
    finish_axes(ax_bot, ylabel="Year-over-year real change",
                yfmt=lambda v, _: f"{v:+.0f}%", xtick_step=5)
    ax_bot.set_xlim(FY_START - 1, FY_END + 1)

    fig.suptitle("Total federal R&D and annual real change, by fiscal year",
                 x=0.005, ha="left", fontsize=14, fontweight="semibold", y=1.03)
    fig.text(0.005, 0.985, f"Constant FY{BASE_YEAR} dollars. Shaded bands are the "
             f"three growth regimes found by the piecewise fit; black bars in the "
             "lower panel are each regime's mean annual change.",
             ha="left", fontsize=9.5, color=INK["secondary"])
    save_figure(fig, "02_trajectory_breakpoints")


# ===========================================================================
# SECTION 4 - BENCHMARK COMPARISON
# ===========================================================================

def analyze_benchmarks(t: Totals, knots: list) -> None:
    header("SECTION 4 - R&D AGAINST GDP AND THE DISCRETIONARY BUDGET")
    e = t.enacted()

    note("Constant dollars correct for prices but not for scale. Two denominators "
         "answer what prices cannot: % of GDP asks how much national output is "
         "committed to federal research, and % of discretionary budget authority "
         "asks how R&D fared against the pot appropriators actually divide each year.")

    print(pd.DataFrame({"R&D current $M": e.current, "GDP $B": e.gdp,
                        "Discretionary BA $B": e.discretionary,
                        "% of GDP": e.pct_gdp,
                        "% of discretionary": e.pct_discretionary})
          .loc[[FY_START, 1990, 2000, knots[1], 2010, 2020, FY_END]].to_string())

    for name, series in [("% of GDP", e.pct_gdp),
                         ("% of discretionary BA", e.pct_discretionary)]:
        print(f"\n  {name}: {series[FY_START]:.2f}% (FY{FY_START}) -> "
              f"{series[FY_END]:.2f}% (FY{FY_END})   "
              f"{series[FY_END] - series[FY_START]:+.2f}pp "
              f"({100 * (series[FY_END] / series[FY_START] - 1):+.0f}% relative)")
        print(f"    peak FY{int(series.idxmax())} at {series.max():.2f}%; "
              f"trough FY{int(series.idxmin())} at {series.min():.2f}%")

    print(f"\n  Real growth, FY{FY_START}-FY{FY_END}:")
    for label, series in [("Federal R&D", e.constant),
                          ("GDP", to_constant(e.gdp, e.deflator)),
                          ("Discretionary BA", to_constant(e.discretionary, e.deflator))]:
        print(f"    {label:<18} {cagr(series):+.2f}%/yr "
              f"({100 * (series[FY_END] / series[FY_START] - 1):+,.0f}% total)")
    note("R&D grew more slowly in real terms than both denominators, which is why "
         "the shares fall even as the dollars rise. The economy outran the research "
         "budget; so did the rest of discretionary spending.")

    r, p = stats.pearsonr(e.constant.values, e.pct_gdp.values)
    print(f"\n  Pearson r between real R&D dollars and R&D % of GDP: "
          f"r = {r:+.3f} (p = {p:.2e})")
    note("The negative correlation is the point: the two measures move in opposite "
         "directions, so a rising dollar figure carries no information about whether "
         "research is gaining or losing ground.")

    plot_benchmarks(t, knots)


def plot_benchmarks(t: Totals, knots: list) -> None:
    """FIG 3 - the two denominators, each on its own axis."""
    e = t.enacted()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    for ax, series, role, panel, yfmt in [
        (axes[0], e.pct_gdp, "gdp_share", "R&D as % of GDP", fmt_pct2),
        (axes[1], e.pct_discretionary, "disc_share",
         "R&D as % of discretionary budget authority", fmt_pct0),
    ]:
        # Lines only: neither panel is zero-based, so an area fill would
        # overstate magnitude.
        ax.plot(series.index, series.values, color=PALETTE[role], lw=2.0)
        for year in knots:
            ax.axvline(year, color=INK["muted"], lw=1.0, ls=(0, (2, 2)), alpha=0.6)
        for year in (FY_START, FY_END):
            ax.scatter([year], [series[year]], s=26, color=PALETTE[role], zorder=5,
                       edgecolor=INK["surface"], linewidth=2)
        finish_axes(ax, title=panel,
                    subtitle=f"FY{FY_START} {series[FY_START]:.2f}%  ->  "
                             f"FY{FY_END} {series[FY_END]:.2f}%",
                    ylabel=series.name, yfmt=yfmt, xtick_step=12)

    fig.suptitle("Federal R&D as a share of GDP and of discretionary budget authority",
                 x=0.004, ha="left", fontsize=14, fontweight="semibold", y=1.06)
    fig.text(0.004, 1.0, f"FY{FY_START}-FY{FY_END}. Real dollars doubled, but R&D lost "
             "ground on both denominators. Each panel has its own axis - the measures "
             f"share a direction, not a unit. Dotted lines are the FY{knots[0]} and "
             f"FY{knots[1]} breakpoints.",
             ha="left", fontsize=9.5, color=INK["secondary"])
    fig.tight_layout()
    save_figure(fig, "03_benchmarks")


# ===========================================================================
# SECTION 5 - COMPOSITION BY CHARACTER OF WORK
# ===========================================================================

def analyze_type_mix(t: Totals, knots: list) -> TypeMix:
    header("SECTION 5 - BASIC RESEARCH, APPLIED RESEARCH, AND DEVELOPMENT")
    mix = load_type_mix(t)
    real, share = mix.real, mix.share

    print(f"\n  Real funding by character of work (constant FY{BASE_YEAR} $M):")
    print(real.rename(columns=TYPE_LABELS)
          .loc[[FY_START, 1986, 2000, knots[1], 2010, 2020, FY_END]].to_string())
    print("\n  Share of the total (%):")
    print(share.rename(columns=TYPE_LABELS)
          .loc[[FY_START, 1986, 2000, knots[1], 2010, 2020, FY_END]].round(1).to_string())

    print(f"\n  FY{FY_START} -> FY{FY_END} by type:")
    for kind, label in TYPE_LABELS.items():
        level, pct = real[kind], share[kind]
        print(f"    {label:<18} ${level[FY_START]:>8,.0f}M -> ${level[FY_END]:>8,.0f}M  "
              f"({100 * (level[FY_END] / level[FY_START] - 1):+6.0f}%, "
              f"{cagr(level):+.2f}%/yr real)   share {pct[FY_START]:4.1f}% -> "
              f"{pct[FY_END]:4.1f}%  ({pct[FY_END] - pct[FY_START]:+.1f}pp)")

    dev, basic = share["Developmental"], share["Basic"]
    print("\n  IMPORTANT CHANGE - the portfolio inverted its centre of gravity.")
    print(f"    Development fell from {dev[FY_START]:.1f}% of federal R&D to "
          f"{dev[FY_END]:.1f}% ({dev[FY_END] - dev[FY_START]:+.1f}pp) while basic "
          f"research rose {basic[FY_START]:.1f}% -> {basic[FY_END]:.1f}% "
          f"({basic[FY_END] - basic[FY_START]:+.1f}pp).")
    print(f"    Research (basic + applied) grew "
          f"{100 * (mix.research[FY_END] / mix.research[FY_START] - 1):+.0f}% real against "
          f"development's "
          f"{100 * (real['Developmental'][FY_END] / real['Developmental'][FY_START] - 1):+.0f}% "
          f"- a {cagr(mix.research) - cagr(real['Developmental']):+.2f} point gap in "
          f"annual real growth sustained for {FY_END - FY_START} years.")

    dev_fit = detect_breakpoints(dev, n_breaks=2, log=False)
    dev_knots = [int(k) for k in dev_fit["knots"]]
    dev_test = chow_f_test(dev, dev_fit["knots"], log=False)
    print(f"\n  TURNING POINT - development's share, breakpoints FY{dev_knots[0]} "
          f"and FY{dev_knots[1]}")
    print("    segment slopes: "
          + "  ->  ".join(f"{r:+.2f}pp/yr" for r in dev_fit["rates"]))
    print(f"    F({dev_test['df'][0]},{dev_test['df'][1]}) = {dev_test['F']:.1f}, "
          f"p = {dev_test['p']:.2e}; residual variance falls "
          f"{dev_test['sse_reduction_%']:.0f}%")
    print(segment_table(dev, dev_fit["knots"], log=False).to_string())
    peak = int(real["Developmental"].idxmax())
    print(f"    Development's real dollars peaked in FY{peak} at "
          f"${real['Developmental'][peak]:,.0f}M and are "
          f"{100 * (real['Developmental'][FY_END] / real['Developmental'][peak] - 1):+.1f}% "
          f"lower in FY{FY_END}.")

    cut = knots[1]
    print(f"\n  CONTRASTING PATTERN - what the flat post-FY{cut} aggregate contains")
    for kind, label in TYPE_LABELS.items():
        print(f"      {label:<18} {100 * (real[kind][FY_END] / real[kind][cut] - 1):+6.1f}%   "
              f"({cagr(real[kind], cut, FY_END):+.2f}%/yr)")
    print(f"      {'ALL TYPES':<18} "
          f"{100 * (mix.total[FY_END] / mix.total[cut] - 1):+6.1f}%   "
          f"({cagr(mix.total, cut, FY_END):+.2f}%/yr)")
    note(f"The aggregate reads as stagnation, but applied research and development "
         f"pulled "
         f"{cagr(real['Applied'], cut, FY_END) - cagr(real['Developmental'], cut, FY_END):.1f} "
         "points per year apart underneath it. A flat total is not a dormant "
         "portfolio - it is an actively reallocated one, and the reallocation is "
         "the policy decision the headline conceals.")

    plot_type_levels(mix)
    plot_type_shares(mix, dev_knots)
    plot_type_indexed(mix, knots)
    return mix


def plot_type_levels(mix: TypeMix) -> None:
    """FIG 4 - real dollars by character of work."""
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for kind, label in TYPE_LABELS.items():
        series = mix.real[kind]
        ax.plot(series.index, series.values, color=PALETTE[kind], lw=2.0, label=label)
        label_series_end(ax, series, PALETTE[kind], label)
    finish_axes(ax, title="Federal R&D by character of work, by fiscal year",
                subtitle=f"Constant FY{BASE_YEAR} dollars, FY{FY_START}-FY{FY_END}. "
                         "Development still buys the most; basic research grew three "
                         "times faster.",
                ylabel="R&D funding", yfmt=fmt_billions, xtick_step=10)
    ax.set_xlim(FY_START, FY_END + 8)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "04_type_levels")


def plot_type_shares(mix: TypeMix, dev_knots: list) -> None:
    """FIG 5 - composition over time as a 100% stacked band."""
    share = mix.share
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    base = pd.Series(0.0, index=share.index)
    for kind in ["Developmental", "Applied", "Basic"]:
        top = base + share[kind]
        ax.fill_between(share.index, base, top, color=PALETTE[kind],
                        alpha=0.85, lw=0, label=TYPE_LABELS[kind])
        # A 2px surface-coloured gap keeps adjacent bands from reading as one.
        ax.plot(share.index, top, color=INK["surface"], lw=2.0, zorder=4)
        mid = (base + top) / 2
        for year, ha, dx in [(FY_START, "left", 6), (FY_END, "right", -6)]:
            ax.annotate(f"{share[kind][year]:.0f}%", (year, mid[year]), ha=ha,
                        va="center", fontsize=10, fontweight="semibold",
                        color=INK["surface"], xytext=(dx, 0),
                        textcoords="offset points")
        base = top
    for year in dev_knots:
        ax.axvline(year, color=INK["surface"], lw=1.4, ls=(0, (3, 2)), zorder=5)
    finish_axes(ax, title="Share of federal R&D by character of work, by fiscal year",
                subtitle=f"Percent of the Type-sheet total, FY{FY_START}-FY{FY_END}; dashed "
                         f"lines are the FY{dev_knots[0]} and FY{dev_knots[1]} breakpoints "
                         "in development's share",
                ylabel="Share of total R&D", yfmt=fmt_pct0, xtick_step=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(FY_START, FY_END)
    ax.grid(visible=False)
    ax.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    save_figure(fig, "05_type_shares")


def plot_type_indexed(mix: TypeMix, knots: list) -> None:
    """FIG 6 - indexed growth; the contrast a level chart cannot show."""
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.axhline(100, color=INK["axis"], lw=1.2)
    for year in knots:
        ax.axvline(year, color=INK["muted"], lw=1.0, ls=(0, (2, 2)), alpha=0.6)
    for kind, label in TYPE_LABELS.items():
        series = index_to_base(mix.real[kind], FY_START)
        ax.plot(series.index, series.values, color=PALETTE[kind], lw=2.0, label=label)
        label_series_end(ax, series, PALETTE[kind], f"{series[FY_END]:.0f}")
    total = index_to_base(mix.total, FY_START)
    ax.plot(total.index, total.values, color=PALETTE["aggregate"], lw=2.0,
            ls=(0, (4, 2)), label="All types (the headline)")
    label_series_end(ax, total, PALETTE["aggregate"], f"{total[FY_END]:.0f}")
    finish_axes(ax, title=f"Federal R&D by character of work, indexed to FY{FY_START} = 100",
                subtitle=f"Constant FY{BASE_YEAR} dollars; the headline total is an average "
                         "of three very different histories. Dotted lines are "
                         f"the FY{knots[0]} and FY{knots[1]} aggregate breakpoints",
                ylabel=f"Index (FY{FY_START} = 100)", yfmt=fmt_plain, xtick_step=10)
    ax.set_xlim(FY_START, FY_END + 4)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    save_figure(fig, "06_type_indexed")


# ===========================================================================
# SECTION 6 - DEFENSE VS NONDEFENSE
# ===========================================================================

def analyze_defense_share(t: Totals) -> dict:
    header("SECTION 6 - DEFENSE VS NONDEFENSE, LEVELS AND SHARE")
    e = t.enacted()
    combined, share = e.constant, e.defense_share

    note("Levels answer 'how much'; the share answers 'who the portfolio is FOR'. "
         "Because the two components sum to the aggregate, every move in this "
         "share is a reallocation the aggregate cannot show by construction.")

    print("\n  Defense share of federal R&D (%):")
    print(share.loc[[FY_START, 1980, 1986, 1990, 1995, 2000, 2005,
                     2010, 2017, FY_END]].round(1).to_string())
    print(f"\n  range: max {share.max():.1f}% in FY{int(share.idxmax())}, "
          f"min {share.min():.1f}% in FY{int(share.idxmin())}, "
          f"swing {share.max() - share.min():.1f}pp")

    fit = detect_breakpoints(share, n_breaks=2, log=False)
    knots = [int(k) for k in fit["knots"]]
    test = chow_f_test(share, fit["knots"], log=False)
    print(f"\n  TURNING POINT - breakpoints FY{knots[0]} and FY{knots[1]}")
    print("    segment slopes: " + "  ->  ".join(f"{r:+.2f}pp/yr" for r in fit["rates"]))
    print(f"    F({test['df'][0]},{test['df'][1]}) = {test['F']:.1f}, p = {test['p']:.2e}; "
          f"residual variance falls {test['sse_reduction_%']:.0f}%")
    print(segment_table(share, fit["knots"], log=False).to_string())

    peak = int(share.idxmax())
    print(f"\n  IMPORTANT CHANGE - the FY{peak} defense peak and its unwinding")
    print(f"    The buildup carried defense from {share[1980]:.1f}% of federal R&D in "
          f"FY1980 to {share[peak]:.1f}% in FY{peak} ({share[peak] - share[1980]:+.1f}pp "
          f"in {peak - 1980} years); the post-Cold War drawdown reversed it to "
          f"{share[1995]:.1f}% by FY1995 ({share[1995] - share[peak]:+.1f}pp in "
          f"{1995 - peak} years).")
    print(f"    In real dollars over FY{peak}-FY1995, defense fell "
          f"{100 * (e.defense[1995] / e.defense[peak] - 1):+.0f}% while nondefense rose "
          f"{100 * (e.nondefense[1995] / e.nondefense[peak] - 1):+.0f}%.")

    print("\n  CONTRASTING PATTERN - the components move while the total stays calm")
    against = opposite_direction_years(combined, e.defense)
    print(f"    In {len(against)} of {len(combined) - 1} year-pairs, defense R&D moved "
          "OPPOSITE to the total.")
    early = share.loc[FY_START:2000]
    print(f"    Over FY{FY_START}-FY2000 the aggregate grew a smooth "
          f"{cagr(combined, FY_START, 2000):+.2f}%/yr real, yet the defense share "
          f"traversed {early.max() - early.min():.1f}pp inside that window "
          f"({early.min():.1f}% to {early.max():.1f}%).")
    r, p = stats.pearsonr(e.defense.diff().dropna(), e.nondefense.diff().dropna())
    print(f"    Annual real changes in the two components correlate at r = {r:+.3f} "
          f"(p = {p:.3f}): weak but real, and r2 = {r ** 2:.2f} means "
          f"{100 * (1 - r ** 2):.0f}% of each component's movement is unexplained by "
          "the other. The aggregate sums two largely separate budget processes.")

    plot_defense(t, share, fit, knots)
    return {"share": share, "knots": knots}


def plot_defense(t: Totals, share: pd.Series, fit: dict, knots: list) -> None:
    """FIG 7 - the two levels, and the share they imply."""
    e = t.enacted()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 4.9))

    for series, role, label in [(e.defense, "defense", "Defense"),
                                (e.nondefense, "nondefense", "Nondefense")]:
        ax_l.plot(series.index, series.values, color=PALETTE[role], lw=2.0)
        label_series_end(ax_l, series, PALETTE[role], label)
    gap = e.defense - e.nondefense
    finish_axes(ax_l, title=f"Constant FY{BASE_YEAR} dollars",
                subtitle=f"Gap peaked at {D}{gap.max() / 1e3:.0f}B in FY{int(gap.idxmax())}",
                ylabel="R&D funding", yfmt=fmt_billions, xtick_step=10)
    ax_l.set_xlim(FY_START, FY_END + 6)
    ax_l.set_ylim(0, None)

    band = share.max() + 5.0
    ax_r.axhline(50, color=INK["axis"], lw=1.2)
    ax_r.plot(share.index, share.values, color=PALETTE["defense"], lw=2.0)
    ax_r.plot(fit["fitted"].index, fit["fitted"].values, color=PALETTE["fit"],
              lw=1.6, ls=(0, (4, 2)), alpha=0.75)
    mark_knots(ax_r, knots, y=share.min() - 2.4)
    for i, ((start, end), rate) in enumerate(zip(
            zip([FY_START] + knots, knots + [FY_END]), fit["rates"])):
        ax_r.annotate(f"{rate:+.2f}pp/yr", ((start + end) / 2, band - 2.8 * (i % 2)),
                      ha="center", va="center", fontsize=9.5, fontweight="semibold",
                      color=INK["primary"])
    for year, dx, ha in [(int(share.idxmax()), -10, "right"),
                         (int(share.idxmin()), 10, "left")]:
        ax_r.scatter([year], [share[year]], s=32, color=PALETTE["defense"], zorder=5,
                     edgecolor=INK["surface"], linewidth=2)
        ax_r.annotate(f"FY{year}  {share[year]:.1f}%", (year, share[year]),
                      textcoords="offset points", xytext=(dx, 0), ha=ha, va="center",
                      fontsize=9.5, fontweight="semibold", color=INK["primary"])
    finish_axes(ax_r, title="Defense share of total federal R&D",
                subtitle=f"A {share.max() - share.min():.0f}-point swing around an even split",
                ylabel="Defense share of federal R&D", yfmt=fmt_pct0, xtick_step=10)
    ax_r.set_ylim(share.min() - 5, band + 2.5)
    ax_r.set_xlim(FY_START, FY_END)

    fig.suptitle("Defense and nondefense R&D: levels and defense share, by fiscal year",
                 x=0.004, ha="left", fontsize=14, fontweight="semibold", y=1.06)
    fig.text(0.004, 1.0, f"Constant FY{BASE_YEAR} dollars, FY{FY_START}-FY{FY_END}. The "
             "dashed line on the right is the piecewise linear fit to the share.",
             ha="left", fontsize=9.5, color=INK["secondary"])
    fig.tight_layout()
    save_figure(fig, "07_defense_levels_and_share")


# ===========================================================================
# SECTION 7 - SYNTHESIS: WHAT THE AGGREGATE CANNOT SAY
# ===========================================================================

def synthesize(t: Totals, mix: TypeMix, defense: dict, knots: list) -> None:
    header("SECTION 7 - SYNTHESIS: WHY R&D DOLLARS ALONE MISLEAD")
    e, cut = t.enacted(), knots[1]

    # --- Part 1: the same dollars on four yardsticks ------------------------
    measures = {"Current dollars": e.current,
                f"Constant FY{BASE_YEAR} dollars": e.constant,
                "% of discretionary BA": e.pct_discretionary,
                "% of GDP": e.pct_gdp}
    indexed = pd.DataFrame({k: index_to_base(v, FY_START) for k, v in measures.items()})
    print(f"  All four measures indexed to FY{FY_START} = 100:")
    print(indexed.loc[[FY_START, 1990, 2000, cut, 2010, 2020, FY_END]].to_string())
    print(f"\n  FY{FY_END} index value (FY{FY_START} = 100):")
    for name in measures:
        value = indexed[name][FY_END]
        print(f"    {name:<28} {value:7.1f}   "
              f"({'grew' if value > 100 else 'shrank'} {abs(value - 100):.0f}%)")

    note(f"The same appropriations read {indexed['Current dollars'][FY_END]:.0f} on one "
         f"yardstick and {indexed['% of GDP'][FY_END]:.0f} on another - a "
         f"{indexed['Current dollars'][FY_END] - indexed['% of GDP'][FY_END]:.0f}-point "
         "spread. Three corrections separate them:")
    print("    1. PRICES    Of the "
          f"{100 * (e.current[FY_END] / e.current[FY_START] - 1):+,.0f}% nominal rise, "
          f"only {100 * (e.constant[FY_END] / e.constant[FY_START] - 1):+,.0f}% is real.")
    print("    2. SCALE     The economy outgrew the research budget: "
          f"{e.pct_gdp[FY_START]:.2f}% of GDP -> {e.pct_gdp[FY_END]:.2f}%.")
    print("    3. PRIORITY  Within the budget appropriators control, R&D slipped from "
          f"{e.pct_discretionary[FY_START]:.1f}% to {e.pct_discretionary[FY_END]:.1f}%.")

    # --- Part 2: the fourth correction the denominators cannot make ---------
    note("A fourth correction is beyond any denominator: WHAT the money buys. "
         "Sections 5-6 show that two portfolios with the same total can be doing "
         "entirely different things. Three findings follow.")

    print(f"\n  1. STAGNATION IS RESHUFFLING. Since FY{cut} the real total moved "
          f"{100 * (mix.total[FY_END] / mix.total[cut] - 1):+.1f}%, which reads as "
          "nothing happening. Underneath:")
    for kind, label in TYPE_LABELS.items():
        print(f"       {label:<18} "
              f"{100 * (mix.real[kind][FY_END] / mix.real[kind][cut] - 1):+6.1f}%")
    moved = (mix.real.loc[FY_END] - mix.real.loc[cut]).abs().sum()
    net = abs(mix.total[FY_END] - mix.total[cut])
    print(f"       Gross reallocation across types is ${moved:,.0f}M against a net "
          f"change of ${net:,.0f}M - a churn-to-net ratio of {moved / net:.1f}x.")

    print("\n  2. THE PARTS CONTRADICT THE WHOLE. Years a component moved opposite "
          "to its aggregate:")
    checks = contradiction_checks(e, mix)
    rates = []
    for label, part, whole, _ in checks:
        years = opposite_direction_years(whole, part)
        rate = 100 * len(years) / (len(whole) - 1)
        rates.append(rate)
        print(f"       {label:<18} {len(years):>2} of {len(whole) - 1} years ({rate:4.1f}%)")
    note(f"Between {min(rates):.0f}% and {max(rates):.0f}% of years - depending on the "
         "component - the headline's direction is the wrong sign for that part of "
         "the portfolio. A reader tracking only the total would have drawn the "
         "opposite conclusion about it in those years.")

    print("\n  3. THE SUM IS CALMER THAN THE PARTS THAT DRIVE IT. Standard deviation "
          "of annual real change:")
    volatility = {"TOTAL (the headline)": e.constant, "Defense R&D": e.defense,
                  "Nondefense R&D": e.nondefense, "Basic research": mix.real["Basic"],
                  "Applied research": mix.real["Applied"],
                  "Development": mix.real["Developmental"]}
    vols = {}
    for label, series in volatility.items():
        yoy = series.pct_change().dropna() * 100
        vols[label] = yoy.std(ddof=1)
        print(f"       {label:<22} {vols[label]:5.2f}pp   (worst {yoy.min():+6.1f}%, "
              f"best {yoy.max():+6.1f}%)")
    headline = vols["TOTAL (the headline)"]
    rougher = sum(1 for k, v in vols.items() if v > headline)
    note(f"Aggregation is a smoothing operation: {rougher} of {len(vols) - 1} components "
         f"swing harder than the headline, led by development at {vols['Development']:.1f}pp "
         f"against the total's {headline:.1f}pp. Offsetting moves cancel inside the sum, "
         "so the headline understates how disruptive a year was for the programs "
         "living in one component. (Applied research is the exception, and is "
         "steadier than the total.)")

    share = defense["share"]
    print("\n  TAKEAWAY")
    print(f"    The headline says federal R&D doubled in real terms over FY{FY_START}-"
          f"FY{FY_END}. The same data says development's share fell "
          f"{mix.share['Developmental'][FY_END] - mix.share['Developmental'][FY_START]:+.0f}pp, "
          f"basic research's rose "
          f"{mix.share['Basic'][FY_END] - mix.share['Basic'][FY_START]:+.0f}pp, and the "
          f"defense share traversed {share.max() - share.min():.0f}pp between "
          f"FY{int(share.idxmax())} and FY{int(share.idxmin())}.")
    note("None of those facts can be recovered from the aggregate, and each changes "
         "the answer to 'what did federal R&D investment do?'. The headline is not "
         "wrong - it is underdetermined. It needs a deflator, a denominator, AND a "
         "breakdown before it supports a claim.")

    plot_measure_divergence(indexed, knots)
    plot_contradictions(e, mix)


def contradiction_checks(e: Totals, mix: TypeMix) -> list:
    """(label, component, its aggregate, colour) for every part/whole pair.

    Defined once and consumed by both the console table and figure 9, so the
    two can never disagree about which pairs are compared.
    """
    return [("Development", mix.real["Developmental"], mix.total, PALETTE["Developmental"]),
            ("Applied research", mix.real["Applied"], mix.total, PALETTE["Applied"]),
            ("Basic research", mix.real["Basic"], mix.total, PALETTE["Basic"]),
            ("Nondefense R&D", e.nondefense, e.constant, PALETTE["nondefense"]),
            ("Defense R&D", e.defense, e.constant, PALETTE["defense"])]


def plot_measure_divergence(indexed: pd.DataFrame, knots: list) -> None:
    """FIG 8 - the headline chart: four measures on one common index scale."""
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    roles = ["current", "constant", "disc_share", "gdp_share"]

    ax.axhline(100, color=INK["axis"], lw=1.2)
    for year in knots:
        ax.axvline(year, color=INK["muted"], lw=1.0, ls=(0, (2, 2)), alpha=0.6)
    ax.fill_between(indexed.index, indexed["% of GDP"], indexed["Current dollars"],
                    color=INK["muted"], alpha=0.06, lw=0)
    for (name, series), role in zip(indexed.items(), roles):
        ax.plot(series.index, series.values, color=PALETTE[role], lw=2.0, label=name)
        label_series_end(ax, series, PALETTE[role], f"{series[FY_END]:.0f}")

    spread = indexed.loc[FY_END].max() - indexed["% of GDP"][FY_END]
    finish_axes(ax, title=f"Four measures of federal R&D, indexed to FY{FY_START} = 100",
                subtitle=f"The same budget read four ways; they diverge by {spread:.0f} index "
                         "points. The band is the gap between the most and least "
                         "flattering yardstick.",
                ylabel=f"Index (FY{FY_START} = 100)", yfmt=fmt_plain, xtick_step=5)
    ax.set_xlim(FY_START, FY_END + 4)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    save_figure(fig, "08_measure_divergence")


def plot_contradictions(e: Totals, mix: TypeMix) -> None:
    """FIG 9 - how often each component moves against its own aggregate."""
    checks = contradiction_checks(e, mix)
    labels = [c[0] for c in checks]
    values = [100 * len(opposite_direction_years(c[2], c[1])) / (len(c[2]) - 1)
              for c in checks]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars = ax.barh(labels, values, color=[c[3] for c in checks], height=0.55, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.0f}%", (value + 0.8, bar.get_y() + bar.get_height() / 2),
                    ha="left", va="center", fontsize=10.5, fontweight="semibold",
                    color=INK["primary"])
    ax.grid(axis="x", color=INK["grid"], lw=0.7)
    ax.grid(axis="y", visible=False)
    finish_axes(ax, title="Years each component moved opposite its own aggregate",
                subtitle=f"Percent of FY{FY_START}-FY{FY_END} years; the aggregate is a "
                         "summary, not a description",
                xlabel="Years moving against the aggregate")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_pct0))
    ax.set_xlim(0, max(values) * 1.22)
    fig.tight_layout()
    save_figure(fig, "09_contradictions")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    apply_chart_style()
    header("U.S. FEDERAL R&D FUNDING - EXPLORATORY DATA ANALYSIS")
    note(f"Source: {DATA_FILES['historical'].name} (AAAS Historical Trends in "
         f"Federal R&D). Figures written to {FIG_DIR.name}/.")

    totals = load_totals()
    analyze_long_term_trend(totals)
    breaks = analyze_trajectory_changes(totals)
    analyze_benchmarks(totals, breaks["knots"])
    mix = analyze_type_mix(totals, breaks["knots"])
    defense = analyze_defense_share(totals)
    synthesize(totals, mix, defense, breaks["knots"])

    header("DONE")
    note(f"9 figures in {FIG_DIR.name}/. Later sections can load any entry in "
         "DATA_FILES with load_wide_sheet()/tidy_wide_sheet() and reuse "
         "to_constant(), detect_breakpoints(), segment_table(), chow_f_test(), "
         "and the plotting helpers unchanged.")


if __name__ == "__main__":
    main()
