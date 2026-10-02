# U.S. Federal R&D Funding — Exploratory Data Analysis

An EDA of federal R&D funding, FY1976–FY2024, built on the AAAS *Historical Trends in
Federal R&D* data file. It argues one thing: **the headline federal R&D number is
underdetermined.** Read on its own it supports almost any story, because it hides price
change, the size of the economy, the size of the budget it competes inside, what the
money buys, and which agency moved.

Two scripts, nine + four figures, every number reproducible from the files in `data/`.

---

## Quick start

```bash
python3 -m pip install pandas numpy matplotlib seaborn scipy openpyxl
python3 main.py        # sections 1–7, writes figures/01–09
python3 drivers.py     # hypotheses 1–3, writes figures/10–13
```

Both scripts print their full analysis to stdout and write PNGs to `figures/`. They are
deterministic — no sampling, no random seeds — so a rerun reproduces the console output
and the figures byte-for-byte on the same library versions.

Verified on:

| | |
|---|---|
| Python | 3.9.6 (macOS) |
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| scipy | 1.13.1 |
| openpyxl | 3.1.5 |

`drivers.py` imports from `main.py`, so run it from the repo root and keep both files
together.

---

## Data

All files are in `data/`, downloaded from the AAAS R&D Budget and Policy Program. None
are modified — every transformation happens in code.

| File | Contents | Dollar base |
|---|---|---|
| `Historical Data.xlsx` | 8 sheets: Totals, Agency, Type, Function, Field, Funder, Discipline | constant FY2017 |
| `Agencies.xlsx` | Total R&D by agency, FY1976–2025 | constant FY2024 |
| `DODST.xlsx` | DOD Science & Technology (6.1/6.2/6.3), FY1990–2025 | constant FY2024 |
| `DOE.xlsx` | DOE R&D by office, FY1990–2025 | constant FY2024 |
| `DefNon.xlsx` | Defense/nondefense/total, FY1976–2023 | constant FY2022 |
| `Disc1.xlsx` | Federal research by discipline, 1970–2021 | constant FY2022 |
| `NIHbud.xlsx` | NIH budgets by institute, FY1990–2025 | constant FY2024 |

`Historical Data.xlsx` is the primary source. Its internal series are compiled by AAAS
from OMB and appropriations documents; the GDP deflator, GDP, and total discretionary
budget authority come from OMB Historical Tables 10.1, 10.2 and 8.1 respectively.

**The mixed dollar bases are deliberate and harmless.** `main.py` works entirely in
constant FY2017 dollars. `drivers.py` pulls agency-level series from workbooks based in
FY2024 — but every test there runs on log differences or on shares, and a fixed
rescaling cancels in both. Levels are never mixed across bases.

### Three caveats that change how results should be read

1. **FY2025–FY2026 are a budget request, not appropriations.** They come from the FY2026
   President's Budget Request. Every statistic is estimated on FY1976–FY2024; the request
   years appear only as dashed segments in figure 01.
2. **FY2009 Recovery Act R&D ($18,739M) is excluded.** AAAS reports it separately, so the
   2009–10 stimulus surge is understated throughout.
3. **The Type sheet does not reconcile to the Totals sheet.** It runs about −4.5% against
   it (range −11.3% to −0.5%) because the two rest on different OMB reporting bases.
   Character-of-work shares are therefore computed *within* the Type sheet, never against
   the Section 1 aggregate.

A fourth, specific to `drivers.py`: `Agencies.xlsx` reports an +85% real rise over
FY1976–FY2024 against `main.py`'s +102%, because it uses the post-FY2017 R&D definition
under which late-stage defense testing left the R&D total. Shares and growth rates are
unaffected.

### Validation the code performs at load

`main.py` asserts these on every run and will stop if any fails:

- Recomputing `current / deflator` reproduces the workbook's constant-dollar series
  (max relative error 1.9e-16).
- `deflator[FY2017] == 1.0000`.
- Zero missing years in FY1976–FY2024.
- The Type sheet's `Research` row equals `Basic + Applied` exactly.

---

## Method

Two choices drive everything and are worth stating up front.

**Breakpoints are found, not chosen.** Regime changes come from a continuous piecewise
linear fit whose knot years are grid-searched to minimise residual sum of squares, with a
five-year minimum segment. Because the fit is continuous at the knots, each knot is a
change in the *rate* of change — a bend, not a level jump. Dollar series are modelled in
logs, so slopes are compound growth rates; shares are modelled directly, so slopes are
percentage points per year. Every fit is tested against a single-trend null with an F test.

**Relationships are tested on growth, not levels.** Almost any two upward-trending series
correlate above r = 0.8. Federal R&D, GDP, the budget, industry R&D and university R&D all
rise over fifty years. `drivers.py` therefore reports every correlation twice — once on
levels, once on annual real growth — and treats only the second as evidence. Candidates
are pre-specified from a stated mechanism rather than screened from the whole workbook.

---

## Findings

### Part one — `main.py`: the headline is underdetermined

**The same appropriations read four different ways.** Indexed to FY1976 = 100, FY2024 is
880 in current dollars, 202 in constant dollars, 85 as a share of discretionary budget
authority, and 56 as a share of GDP. An 824-point spread from one budget.

| Measure | FY1976 | FY2024 | Change |
|---|---|---|---|
| Current dollars | $22,055M | $194,106M | +780% |
| Constant FY2017 dollars | $76,740M | $154,876M | +102% |
| % of GDP | 1.23% | 0.69% | −44% |
| % of discretionary BA | 12.56% | 10.72% | −15% |

The nominal multiple factors exactly: **8.80× nominal = 2.02× real × 4.36× prices**, so
68% of the headline growth is price change. Real R&D peaked in FY2010 and sits 7.2% below
that peak. R&D grew at +1.47%/yr real while GDP grew +2.72%/yr and discretionary BA
+1.81%/yr — which is why the shares fall while the dollars rise. Real dollars and share of
GDP correlate at **r = −0.61** (p = 2.9e-06): the two measures move in opposite directions.

**Two trajectory changes, FY2000 and FY2005.** Real growth ran +1.55%/yr → **+6.03%/yr**
→ **−0.54%/yr**. F(2,45) = 34.9, p = 7.1e-10; residual variance falls 61%. The FY2005 bend
is the larger break and is invisible in nominal dollars, which kept rising at +2.08%/yr
across the same stretch.

**The portfolio inverted its centre of gravity.**

| Character of work | FY1976 → FY2024 real | CAGR | Share change |
|---|---|---|---|
| Basic research | +317% | +3.02%/yr | 11.9% → 23.8% |
| Applied research | +138% | +1.83%/yr | 24.8% → 28.3% |
| Development | +58% | +0.95%/yr | 63.3% → 47.9% |

Development's share breaks at FY2012 and FY2019 (−0.32 → −1.78 → +1.60 pp/yr; F(2,45) =
10.2, p = 2.2e-04). Since FY2005 the real total moved −3.3% — but applied research rose
+20.4% while development fell −16.4%. Gross reallocation is $23.2B against $5.1B net, a
**4.6× churn-to-net ratio**. A flat total is an actively reallocated portfolio, not a
dormant one.

**Defense share swung 23.7 points** — 46.7% in FY1980 to 68.9% in FY1986, back to 53.3% by
FY1995. Breakpoints FY1987 and FY1992 (+2.32 → −2.67 → −0.20 pp/yr; F(2,45) = 50.2,
p = 3.5e-12). Over FY1976–2000 the aggregate grew a smooth +1.71%/yr real while the share
traversed 22.2pp inside that same window.

**Components routinely contradict the whole.** Between 10% and 35% of years — depending on
the component — the headline's direction is the wrong sign for that part of the portfolio.
Four of five components are more volatile than the headline (development 9.2pp vs the
total's 5.4pp); applied research is the one exception.

### Part two — `drivers.py`: what the number actually tracks

Three hypotheses, each with its test and stakes stated in the script output.

**H1 — Federal R&D is an appropriations phenomenon, not an economic one.**

| Candidate | r (levels) | r (growth) | p |
|---|---|---|---|
| Discretionary budget | +0.93 | **+0.56** | <0.001 |
| GDP | +0.88 | +0.13 | 0.38 |
| University R&D | +0.83 | +0.24 | 0.11 |
| Industry R&D | +0.78 | −0.08 | 0.61 |

Mean level correlation 0.85; mean growth correlation 0.21. Only the budget survives
detrending. There is **no evidence of crowding out or crowding in** — year to year,
federal and industry R&D are unrelated. Elasticity to the budget: total β = 0.65,
defense β = 0.85 (indistinguishable from 1), nondefense β = 0.39 (decisively below 1).

> Excluded by construction: the Function sheet's budget-function columns (050, 550, 251)
> are federal R&D split by function, not outside variables. Defense R&D against function
> 050 correlates at r = 0.98 on growth — that measures the partition, not a relationship.

**H2 — The budget's pull is concentrated, not shared.** Only 3 of 12 agencies respond at
all (DOD 0.85, HHS 0.57, NIH 0.55); the other nine have coefficients indistinguishable
from zero. **DOD supplies 71% of the aggregate's year-to-year variance on 51% of its
funding.** Corollary: EPA −62%, NASA −28%, DOT −25% and Interior −21% in real terms, none
of them budget-sensitive — they lost steadily inside a total that grew.

**H3 — The cycle reaches development, not research.** DOD S&T has β = 0.36 to DOD's own
R&D and no link to the budget at all (p = 0.17). Inside S&T, FY1990→2024: basic +28%,
applied +39%, advanced technology development **−16%**. The research-over-development
shift repeats inside the single agency that dominates the total, so it is not an artefact
of mixing agencies.

**Together:** the headline is a budget-negotiation outcome dominated by one agency's
procurement cycle. Reading it as a measure of national research effort attributes to
science a series that is mostly about something else.

---

## Figures

Regenerated on every run; `figures/` can be deleted safely.

| | Figure | From |
|---|---|---|
| 01 | Total federal R&D, current and constant dollars | `main.py` §2 |
| 02 | Total federal R&D and annual real change | §3 |
| 03 | R&D as a share of GDP and of discretionary BA | §4 |
| 04 | Federal R&D by character of work | §5 |
| 05 | Share of federal R&D by character of work | §5 |
| 06 | Character of work, indexed to FY1976 = 100 | §5 |
| 07 | Defense/nondefense levels and defense share | §6 |
| 08 | Four measures of federal R&D, indexed | §7 |
| 09 | Years each component moved opposite its aggregate | §7 |
| 10 | Correlation with four candidates, levels vs growth | `drivers.py` H1 |
| 11 | Budget elasticity by agency | H2 |
| 12 | Funding share vs variance share by agency | H2 |
| 13 | DOD research tiers, indexed to FY1990 = 100 | H3 |

---

## Code structure

### `main.py` (1,236 lines)

Section 0 is deliberately generic infrastructure; sections 1–7 are the analysis.

- `load_wide_sheet` / `tidy_wide_sheet` — handle any "metadata columns + one column per
  fiscal year" sheet. Covers Totals, Type, Field and Funder.
- `to_constant`, `cagr`, `index_to_base`, `loglinear_trend` — deflation and growth.
- `fit_piecewise(series, knots, *, log=True)` — continuous piecewise fit. `log=True` for
  dollars (slopes are %/yr), `log=False` for shares (pp/yr).
- `detect_breakpoints` — grid-searches knots minimising SSE.
- `segment_table` — per-regime statistics, adapting its columns to `log=`.
- `chow_f_test` — piecewise vs single trend.
- `opposite_direction_years` — years a component moved against its aggregate.
- `apply_chart_style`, `finish_axes`, `save_figure`, `label_series_end`, `mark_knots`.

### `drivers.py` (639 lines)

Imports all of the above from `main.py`, so the two cannot drift apart on colour or style.
Adds:

- `load_aaas_workbook(name)` — reads the six standalone workbooks, whose layout
  `main.py`'s loaders do not cover. Finds the year row by taking whichever of the first
  rows parses the most four-digit years, so it tolerates labels like `2025 Request`.
- `funder_series`, `growth`, `paired_correlation`, `elasticity`.
- `hypothesis()` / `finding()` — the reporting scaffold that keeps claim, test, stakes and
  result explicit in the output.

### Colour

One palette, keyed by meaning, defined once in `main.py` and extended in `drivers.py`.
Two families run through the analysis and never share a figure, so they may share slots:

- **Family A** — how the dollars are *measured*: current (orange), constant (blue),
  % of GDP (red), % of discretionary (aqua).
- **Family B** — what the dollars are *for*: basic (blue), applied (orange), development
  (aqua), defense (violet), nondefense (magenta).

The aggregate is never a hue — it is ink, dashed — so the headline is always
distinguishable from a component. Sign uses the blue↔red diverging pair, never green/red.
Figure titles describe the axes; findings live in subtitles.

---

## Known gaps

- **`DOE.xlsx`, `NIHbud.xlsx` and `Disc1.xlsx` are unused.** All three load with
  `load_aaas_workbook()` unchanged. NIH institute dispersion was checked during
  exploration and mostly restated H3; the other two are untouched.
- **Figure 12 is the weakest of the four in `drivers.py`.** Its finding (71% vs 51%) is a
  single sentence and could fold into H2's text.
- **No formal time-series inference.** Elasticities are OLS on log differences with
  standard errors assuming independent residuals. No unit-root test, no Newey–West
  correction, no lag structure beyond the contemporaneous term. Lags of 0–4 years were
  checked informally for the federal→industry relationship and none was significant, but
  this is not in the committed code.
- **Correlation, not causation.** Every relationship reported is co-movement. The budget
  elasticity in particular is partly mechanical — R&D is a component of the discretionary
  budget it is regressed on.
