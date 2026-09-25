# FERC Energy Infrastructure Update (EIU) — transmission completions, 2009–2025

Source: FERC Office of Energy Projects, *Energy Infrastructure Update* (a.k.a. Monthly
Infrastructure Report / MIR), "Transmission Projects Completed" table.

Underlying data in that table is NOT collected by FERC. It is derived from NERC
Electricity Supply and Demand (ES&D) and *U.S. Electric Transmission Projects* ©
The C Three Group, LLC. The reports state: "The data may be subject to update."

## The revision problem

Transmission line-miles are back-filled for years after first publication, because
vendor confirmation of in-service dates lags energization. Contemporaneous figures are
systematically low.

FERC quantified this once, in the Dec 2011 report:
> "In 2010 approximately 55 miles of transmission projects have not been confirmed as
> placed into service. In 2011, approximately 1,277 miles have been unconfirmed."

i.e. at the Y+1 reporting date, year Y is essentially complete (55 mi outstanding),
while the current year is missing ~1,277 mi.

## Extraction rule used here

For each year Y, take the "January – December Y Cumulative" column from the
**December Y+1** report — not from the December Y report.

Validated three ways:
1. Mid-year reports also carry a full prior-calendar-year column (not a partial-year
   comparison), so ~12 vintages exist per year and the back-fill trajectory is visible.
2. Convergence is observed within Y+1, mostly in Q1:
   - Jan–Dec 2018: Mar-19 1,281.0 → Jun-19 1,281.0 → Sep-19 1,438.1 → Dec-19 1,438.1
   - Jan–Dec 2019: Dec-19 699.9 → Mar-20 908.7 → Jun-20 911.2 → Dec-20 915.6
3. Matches FERC's own unconfirmed-mileage note above.

Naive summation of the contemporaneous December-Y figures for 2010–2025 gives
16,826 mi vs 27,668 mi on this rule — a 39% undercount.

## Caveats

- **No "<= 230 kV" aggregate exists.** The table has rows for 230 / 345 / 500 kV only
  (the Dec 2010 report adds a 765 kV row, 0.0 for both 2009 and 2010). Sub-230 kV
  projects (138 kV, 161 kV, …) appear in the narrative highlights but are not in the
  table and not in its totals. The `mi_230kV` column is the 230 kV class, not "<=230".
- **2025 has no Y+1 vintage.** The EIU series ends with December 2025 (revised
  2026-04-08); no 2026 monthly reports were published (slugs probed, all 404). The 2025
  row is therefore contemporaneous and undercounted by an expected 25–60%.
- **Uplift factors are unstable**: median 1.52x, but 2021 = 7.09x, 2024 = 4.59x,
  2022 = 1.00x (Dec 2023 restated 2022 unchanged — worth a second look).
- **2013 is an outlier**: 3,663.5 mi at 345 kV in a single year. Verify before citing.
- The "(Revised …)" filename labels do not guarantee the transmission table was
  refreshed. Dec 2020 and Dec 2021 are both labelled "Revised 9-28-2022" yet disagree
  on Jan–Dec 2020 (896.6 vs 1,133.3 mi).

## MW-mile

`gwmi_*` columns apply fixed per-circuit thermal ratings supplied by the user:
230 kV = 221 MW, 345 kV = 1198 MW, 500 kV = 2572 MW. These are assumptions, not
FERC data.

Result: 27,191 GW-mile over 2009–2025.
- 2009–2024 (16 yr, drops undercounted 2025): 26.62 TW-mi, **mean 1,664 GW-mi/yr**
- 2009–2025 with 2025 scaled 1.52x (17 yr): 27.49 TW-mi, mean 1,617 GW-mi/yr
- 2009–2025 as printed (17 yr): 27.19 TW-mi, mean 1,599 GW-mi/yr

Share of MW-mile: 230 kV 9.8% / 345 kV 65.2% / 500 kV 25.0%
Share of miles:   230 kV 40.7% / 345 kV 50.3% / 500 kV  9.0%

Year-to-year spread is 46x (2023: 129 GW-mi; 2013: 5,910 GW-mi). Median year
(2009–2024) is 1,491 GW-mi, below the mean — distribution is right-skewed by 2013.

## Access note

`www.ferc.gov` HTML pages are behind a Cloudflare JS challenge (HTTP 403). The mirror
host **`cms.ferc.gov` is not**, and `https://cms.ferc.gov/staff-reports-and-papers`
lists all 175 EIU entries (2010–2025) inline, including the archived ones. It does rate
limit — retry with backoff. PDFs under `/sites/default/files/` are reachable on either
host.

## Files

- `ferc_eiu_transmission_2009_2025.csv` — the dataset, one row per year, with source
  report and vintage for every figure
- `eiu_index.tsv` — all 175 EIU reports (title -> URL), 2010–2025
- `pdf/december/YYYY.pdf` — the 16 December reports the dataset is built from.
  Filename year = the report's own month, so `2020.pdf` is the source for year 2019.
- `pdf/monthly/` — Mar/Jun/Sep 2019 and Mar/Jun 2020, used for the convergence check

Extract tables with `pdftotext -table` (xpdf). `-layout` scrambles these columns.
