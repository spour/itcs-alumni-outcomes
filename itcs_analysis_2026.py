#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings("ignore")
DATA_PATH    = "Where are Trainees Now_ - Raw Data.csv"
FIG_DIR      = "figures"
CLEANED_PATH = "itcs_cleaned_data.csv"
os.makedirs(FIG_DIR, exist_ok=True)
HISTORICAL = pd.DataFrame({
    "year":                [2016, 2017, 2018, 2019, 2020],
    "n_total":             [  11,   45,   41,   40,   47],
    "n_grad":              [  11,   34,   30,   19,   12],
    "Industry":            [  64,   62,   43,   63,   42],
    "Academia/Healthcare": [  27,   18,   43,   26,   42],
    "Non-profit":          [   0,    3,    3,    5,    8],
    "Government":          [   0,    3,    3,    5,    0],
    "Unknown":             [   9,   14,    8,    0,    8],
}).set_index("year")
HISTORICAL_TOTAL = {
    "n_total": 184, "n_grad": 106,
    "Industry": 55, "Academia/Healthcare": 28,
    "Non-profit": 2, "Government": 1, "Unknown": 14,
}
SECTOR_ORDER = [
    "Industry", "Academia", "Further Schooling",
    "Non-profit", "Government", "Unemployed", "Unknown",
]
PALETTE = {
    "Industry":          "#1565C0",
    "Academia":          "#E65100",
    "Further Schooling": "#00838F",
    "Non-profit":        "#6A1B9A",
    "Government":        "#AD1457",
    "Unemployed":        "#BF360C",
    "Unknown":           "#757575",
}
POSITION_ORDER = ["MSc", "MHSc", "PhD", "Postdoc", "PharmD", "Other", "Unknown"]
POSITION_PALETTE = {
    "MSc":     "#43A047",
    "MHSc":    "#00ACC1",
    "PhD":     "#1E88E5",
    "Postdoc": "#FB8C00",
    "PharmD":  "#8E24AA",
    "Other":   "#6D4C41",
    "Unknown": "#90A4AE",
}
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
})
pct_fmt = FuncFormatter(lambda x, _: "{:.0f}%".format(x))
raw = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
BASE_COLS = [
    "collector", "name", "linkedin", "year", "position",
    "graduated", "sector", "role", "company",
    "had_internship", "internship_industry", "internship_nonprofit",
    "comments",
]
raw.columns = BASE_COLS + [
    "col_{:02d}".format(i) for i in range(1, raw.shape[1] - 12)
]
print("=" * 70)
print("ITCS 2026 RETROSPECTIVE — QC REPORT")
print("=" * 70)
print("\nRaw rows in CSV: {}".format(len(raw)))
blank_mask = (
    (raw["name"].str.strip() == "") &
    (raw["position"].str.strip().str.upper() == "FALSE")
)
print("Blank template rows dropped : {}".format(blank_mask.sum()))
df = raw[~blank_mask].copy()
df["year"] = pd.to_numeric(df["year"].str.strip(), errors="coerce")
bad_year = df["year"].isna()
print("Rows with non-numeric year  : {}".format(bad_year.sum()))
df = df[~bad_year].copy()
df["year"] = df["year"].astype(int)
NOT_FOUND_STRINGS = ["cannot find", "could not find", "can't find", "cant find"]
def is_not_found(row):
    li = str(row["linkedin"]).strip().lower()
    nm = str(row["name"]).strip().lower()
    pos = str(row["position"]).strip().upper()
    grad = str(row["graduated"]).strip().upper()
    return (
        any(s in li for s in NOT_FOUND_STRINGS) or
        any(s in nm for s in NOT_FOUND_STRINGS) or
        (pos == "FALSE" and grad == "FALSE" and li in ("", "false"))
    )
df["not_found"] = df.apply(is_not_found, axis=1)
print("'Not found' rows flagged    : {}".format(df["not_found"].sum()))
ghost_mask = (
    (df["position"].str.strip().str.upper() == "FALSE") &
    (df["graduated"].str.strip().str.upper() == "FALSE") &
    (df["linkedin"].str.strip().str.lower().isin(["", "false"]))
)
print("Ghost rows excluded          : {}".format(ghost_mask.sum()))
df = df[~ghost_mask].copy()
def parse_graduated(v):
    v = str(v).strip().upper()
    if v in ("TRUE", "YES", "1"):  return True
    if v in ("FALSE", "NO", "0"): return False
    return None
df["graduated_bool"] = df["graduated"].apply(parse_graduated)
def map_position(v):
    v = str(v).strip().lower()
    if v.startswith("mhsc"):   return "MHSc"
    if v.startswith("msc"):    return "MSc"
    if v.startswith("masc"):   return "MSc"
    if v.startswith("master"): return "MSc"
    if v.startswith("phd"):    return "PhD"
    if v.startswith("postdoc"):return "Postdoc"
    if v.startswith("pharmd"): return "PharmD"
    if v in ("false", ""):     return "Unknown"
    if v.startswith("other"):  return "Other"
    return "Other"
df["position_std"] = df["position"].apply(map_position)
df.loc[df["position"].str.strip().str.upper() == "FALSE", "position_std"] = "Unknown"
SECTOR_RAW_MAP = {
    "industry":            "Industry",
    "academia":            "Academia",
    "academia/healthcare": "Academia",
    "non-profit":          "Non-profit",
    "nonprofit":           "Non-profit",
    "government":          "Government",
    "unemployed":          "Unemployed",
    "other":               "_other_",
    "":                    "Unknown",
}
NONPROFIT_COMPANY_KW = (
    "parkinson", "cancer care", "sickkids", "sick kids", "obi", "ontario brain",
    "science collective", "rmic", "regenerative medicine institute",
    "mitacs", "cihr", "nserc", "hospital for sick", "sunnybrook",
    "unity health", "uhn ", "university health", "st. michael", "st michael",
    "women's college", "mount sinai", "trillium", "cancer research",
    "heart and stroke", "alzheimer", "ms society", "diabetes canada",
)
def map_sector_raw(v):
    v = str(v).strip().lower()
    return SECTOR_RAW_MAP.get(v, "Unknown")
df["sector_raw"] = df["sector"].apply(map_sector_raw)
FS_ROLE_KW = [
    "jd candidate", "jd student", "jd ", "law school", "law student",
    "medical student", "md candidate", "md student", "dds", "dental school",
]
UNIV_KW = ("university", "uoft", "u of t", "college", "school", "medicine",
           "cornell", "mcmaster", "queens", "western", "mcgill", "toronto",
           "dalhousie", "alberta", "ottawa", "montreal", "medical",
           "harvard", "mit ", "columbia", "stanford", "oxford", "cambridge")
def detect_further_schooling(row):
    role    = str(row["role"]).strip().lower()
    company = str(row["company"]).strip().lower()
    pos     = str(row["position"]).strip().lower()
    sec     = row["sector_raw"]
    if sec not in ("Academia", "_other_", "Unknown"):
        return False
    if any(kw in role for kw in FS_ROLE_KW):
        return True
    if role == "md" and any(w in company for w in UNIV_KW):
        return True
    if (any(pos.startswith(p) for p in ("msc", "mhsc", "masc", "master")) and
            role in ("phd", "phd student", "phd candidate") and
            any(w in company for w in UNIV_KW)):
        return True
    return False
df["further_schooling"] = df.apply(detect_further_schooling, axis=1)
UNEMPLOYED_ROLE_KW = [
    "no current position", "no position", "not employed",
    "not working", "unemployed", "seeking", "between roles",
    "no current role", "idk", "seems unemployed", "looking for",
]
def is_other_nonprofit(row):
    if row["sector_raw"] != "_other_":
        return False
    company = str(row["company"]).strip().lower()
    return any(kw in company for kw in NONPROFIT_COMPANY_KW)
def detect_unemployed_other(row):
    if row["sector_raw"] != "_other_":
        return False
    if is_other_nonprofit(row):
        return False
    role    = str(row["role"]).strip().lower()
    company = str(row["company"]).strip().lower()
    if role in ("idk", "n/a", "na", "", "unknown") and \
       company in ("idk", "n/a", "na", "", "unknown", "unemployed"):
        return True
    return any(kw in role or kw in company for kw in UNEMPLOYED_ROLE_KW)
df["other_is_unemployed"] = df.apply(detect_unemployed_other, axis=1)
df["other_is_nonprofit"] = df.apply(is_other_nonprofit, axis=1)
def resolve_sector(row):
    if row["not_found"] and str(row["sector"]).strip() in ("", "FALSE", "false"):
        return "Unknown"
    sec = row["sector_raw"]
    if row["further_schooling"]:
        return "Further Schooling"
    if sec == "_other_":
        if row["other_is_nonprofit"]:
            return "Non-profit"
        if row["other_is_unemployed"]:
            return "Unemployed"
        return "Unknown"
    return sec
df["sector_std"] = df.apply(resolve_sector, axis=1)
UNPAID_TOKENS    = ("unpaid", "not paid", "no pay", "volunteer (", "voluntary")
UNDERGRAD_TOKENS = ("undergrad co-op", "undergrad coop", "undergrad intern",
                    "undergrad?", "undergrad only", ", undergrad", "undergrad")
YES_TOKENS       = ("yes", "mitacs", "benchsci", "shift health", "j&j",
                    "intern", "co-op", "coop", "practicum", "fellow",
                    "halohealth", "analyst", "associate", "coordinator",
                    "consultant", "manager", "eversana", "iqvia", "roche",
                    "sanofi", "pfizer", "astrazeneca", "gsk", "novartis",
                    "jnj", "obi", "deep genomics", "geneseeq", "xtalks",
                    "reverba", "biolytical", "rpi", "basl", "arc ",
                    "xenon", "griffith", "sqi", "bloomburton",
                    "ministry of", "health canada")
def map_internship(row):
    if row["position_std"] == "Postdoc":
        return "No"
    v = str(row["had_internship"]).strip().lower()
    if v in ("no", "no?", "false", "0", "", "none"):
        return "No"
    if any(t in v for t in UNPAID_TOKENS):
        if "paid and unpaid" in v or "paid," in v:
            return "Yes"
        return "No"
    if any(t in v for t in UNDERGRAD_TOKENS):
        if any(g in v for g in (" grad", "paid", "industry", "pharma", "biotech")):
            return "Yes"
        return "No"
    if v in ("unknown", "n/a", "na", "maybe?", "maybe", "no?", "yes?"):
        return "Unknown"
    if any(t in v for t in YES_TOKENS):
        return "Yes"
    if len(v) > 3:
        return "Yes"
    return "Unknown"
df["internship_std"] = df.apply(map_internship, axis=1)
def clean_url(url):
    u = str(url).strip().lower()
    if any(s in u for s in NOT_FOUND_STRINGS):
        return ""
    m = re.search(r"linkedin\.com/in/([^/?#\s]+)", u)
    if m:
        return "linkedin.com/in/" + m.group(1)
    return ""
def norm_name(name):
    return str(name).strip().lower()
df["url_key"]  = df["linkedin"].apply(clean_url)
df["name_key"] = df["name"].apply(norm_name)
def canonical_id(row):
    u = row["url_key"]
    if u:
        return u
    n = row["name_key"]
    return n if n else "__blank__"
df["canon_id"] = df.apply(canonical_id, axis=1)
grad_best = df.groupby("canon_id")["graduated_bool"].apply(
    lambda s: True if True in s.values else (
        False if False in s.values else None)
)
sector_best = df[df["graduated_bool"] == True].groupby("canon_id")["sector_std"].first()
df_sorted = df.sort_values("year", ascending=True)
df_dedup  = df_sorted.drop_duplicates(subset="canon_id", keep="first").copy()
df_dedup  = df_dedup[df_dedup["canon_id"] != "__blank__"].copy()
df_dedup["graduated_bool"] = df_dedup["canon_id"].map(grad_best)
needs_sector_update = (
    df_dedup["graduated_bool"] == True
) & (df_dedup["sector_std"] == "Unknown") & (df_dedup["canon_id"].isin(sector_best.index))
df_dedup.loc[needs_sector_update, "sector_std"] = \
    df_dedup.loc[needs_sector_update, "canon_id"].map(sector_best)
df_dedup = df_dedup.reset_index(drop=True)
n_raw        = len(df)
n_dedup      = len(df_dedup)
n_removed    = n_raw - n_dedup
print("\nParticipation-years before dedup : {}".format(n_raw))
print("Unique canonical IDs removed     : {}".format(n_removed))
print("Participation-years after dedup  : {}".format(n_dedup))
duped_ids = df_sorted[df_sorted.duplicated(subset="canon_id", keep="first")][
    ["name", "year", "canon_id"]
].sort_values("year")
if len(duped_ids) > 0:
    print("\nDuplicate entries removed (kept first appearance):")
    for _, row in duped_ids.iterrows():
        first_yr = df_sorted[df_sorted["canon_id"] == row["canon_id"]]["year"].min()
        print("  {} ({}) — first appeared: {}".format(
            row["name"], row["year"], first_yr))
print("\n" + "-" * 70)
print("COHORT SUMMARY (de-duplicated, 2021–2025)")
print("-" * 70)
cohort_stats = df_dedup.groupby("year").apply(lambda g: pd.Series({
    "n_total":        len(g),
    "n_graduated":    g["graduated_bool"].eq(True).sum(),
    "n_in_training":  g["graduated_bool"].eq(False).sum(),
    "n_unknown_grad": g["graduated_bool"].isna().sum(),
    "n_not_found":    g["not_found"].sum(),
})).reset_index()
print(cohort_stats.to_string(index=False))
pos_pivot = (
    df_dedup.groupby(["year", "position_std"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[c for c in POSITION_ORDER if c in df_dedup["position_std"].unique()],
                     fill_value=0)
)
print("\nTrainee type breakdown by cohort year:")
print(pos_pivot.to_string())
grad_df = df_dedup[df_dedup["graduated_bool"] == True].copy()
print("\n" + "-" * 70)
print("SECTOR ANALYSIS (graduates only, 2021–2025, de-duplicated)")
print("-" * 70)
print("Total graduates: {}".format(len(grad_df)))
sector_counts = (
    grad_df.groupby(["year", "sector_std"])
           .size()
           .unstack(fill_value=0)
           .reindex(columns=SECTOR_ORDER, fill_value=0)
)
sector_pct = sector_counts.div(sector_counts.sum(axis=1), axis=0).mul(100)
print("\nSector counts by cohort (graduates):")
print(sector_counts.to_string())
print("\nSector % by cohort (graduates):")
print(sector_pct.round(1).to_string())
tot_counts = sector_counts.sum()
tot_pct    = (tot_counts / tot_counts.sum() * 100).round(1)
print("\n2021–2025 combined (n={} graduates):".format(int(tot_counts.sum())))
for sec in SECTOR_ORDER:
    print("  {:<20} {:>4}  ({:.1f}%)".format(
        sec, int(tot_counts[sec]), float(tot_pct[sec])))
print("\n" + "-" * 70)
print("INTERNSHIP ANALYSIS (paid only, graduates, 2021–2025)")
print("-" * 70)
intern_sub  = grad_df[grad_df["internship_std"].isin(["Yes", "No"])]
intern_counts = (
    intern_sub.groupby(["internship_std", "sector_std"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=SECTOR_ORDER, fill_value=0)
)
intern_pct = intern_counts.div(intern_counts.sum(axis=1), axis=0).mul(100)
print(intern_counts.to_string())
print("\nAs %:")
print(intern_pct.round(1).to_string())
clean_cols = {
    "collector":       "collector",
    "name":            "name",
    "year":            "cohort_year",
    "position_std":    "trainee_type",
    "graduated_bool":  "graduated",
    "sector_std":      "sector",
    "role":            "current_role",
    "company":         "current_company",
    "internship_std":  "paid_internship",
    "not_found":       "not_found",
    "canon_id":        "canonical_id",
}
clean_df = df_dedup[[c for c in clean_cols]].rename(columns=clean_cols)
clean_df.to_csv(CLEANED_PATH, index=False)
print("\nCleaned dataset saved to: {}".format(CLEANED_PATH))
print("  Rows: {}".format(len(clean_df)))
avail_pos = [p for p in POSITION_ORDER if p in pos_pivot.columns]
fig, ax = plt.subplots(figsize=(9, 5))
bottom  = np.zeros(len(pos_pivot))
for pos in avail_pos:
    vals = pos_pivot[pos].values.astype(float)
    ax.bar(pos_pivot.index, vals, bottom=bottom,
           label=pos, color=POSITION_PALETTE.get(pos, "#90A4AE"),
           edgecolor="white", width=0.6)
    bottom += vals
for yr, tot in cohort_stats.set_index("year")["n_total"].items():
    ax.text(yr, bottom[list(pos_pivot.index).index(yr)] + 0.5,
            str(tot), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Cohort year")
ax.set_ylabel("Number of participants")
ax.set_title("ITCS Participation by Cohort Year and Trainee Type (2021–2025)\n"
             "(de-duplicated: each person counted in first cohort year only)")
ax.set_xticks(pos_pivot.index)
ax.set_ylim(0, pos_pivot.sum(axis=1).max() * 1.18)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.legend(title="Position", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_cohort_composition.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nFig 1 saved.")
fig, ax = plt.subplots(figsize=(10, 6))
bottom  = np.zeros(len(sector_pct))
for sec in SECTOR_ORDER:
    if sec not in sector_pct.columns:
        continue
    vals = sector_pct[sec].values
    bars = ax.bar(sector_pct.index, vals, bottom=bottom,
                  label=sec, color=PALETTE[sec], edgecolor="white", width=0.65)
    if sec == "Industry":
        for rect, val, bot in zip(bars, vals, bottom):
            if val >= 5:
                ax.text(rect.get_x() + rect.get_width() / 2.0,
                        bot + val / 2.0,
                        "{:.0f}%".format(val),
                        ha="center", va="center",
                        color="white", fontweight="bold", fontsize=11)
    bottom += vals
row_sums = sector_counts.sum(axis=1)
for yr, n in row_sums.items():
    ax.text(yr, -6, "n={}".format(int(n)),
            ha="center", va="top", fontsize=9, color="#444")
ax.set_ylim(-12, 110)
ax.set_xlabel("Cohort year")
ax.set_ylabel("% of graduates")
ax.set_title("Employment Sector by Cohort Year — ITCS Alumni 2021–2025\n"
             "(graduates only)")
ax.set_xticks(sector_pct.index)
ax.yaxis.set_major_formatter(pct_fmt)
legend_handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in SECTOR_ORDER]
ax.legend(handles=legend_handles, title="Sector",
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_sector_by_cohort.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Fig 2 saved.")
hist_yrs  = HISTORICAL.index.values
hist_ind  = HISTORICAL["Industry"].values
hist_grad = HISTORICAL["n_grad"].values
new_yrs  = sector_pct.index.values
new_ind  = sector_pct.get("Industry", pd.Series(0, index=sector_pct.index)).values
new_grad = row_sums.values
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(20, color="#888", linestyle=":", lw=1.8,
           label="Benchmark: ~20% industry — UofT PhD graduates 2000–2015\n"
                 "(Reithmeier et al. 2019, 10,000 PhDs Project, PLoS ONE)")
ax.axhline(25, color="#aaa", linestyle=(0, (3, 6)), lw=1.4,
           label="Benchmark: ~25% industry — UofT life sciences PhDs\n"
                 "(Reithmeier et al. 2019, life sciences subset)")
ax.plot(hist_yrs, hist_ind, "o-", color="#1565C0", lw=2.5, ms=9,
        label="2016–2020 (Sealey et al. 2021, Jan 2021 snapshot)")
for yr, pct, n in zip(hist_yrs, hist_ind, hist_grad):
    ax.annotate("{:.0f}%\n(n={})".format(pct, n), (yr, pct),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8.5, color="#1565C0")
ax.plot(new_yrs, new_ind, "s--", color="#E65100", lw=2.5, ms=9,
        label="2021–2025 (this analysis, 2025–2026 snapshot)")
for yr, pct, n in zip(new_yrs, new_ind, new_grad):
    ax.annotate("{:.0f}%\n(n={})".format(pct, n), (yr, pct),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8.5, color="#E65100")
ax.scatter([2020.6], [55], color="#1565C0", marker="D", s=120, zorder=5)
ax.annotate("55% overall\n2016–2020\n(n=106)", (2020.6, 55),
            textcoords="offset points", xytext=(22, 0), ha="left", fontsize=8.5,
            color="#1565C0",
            arrowprops=dict(arrowstyle="-", color="#1565C0", lw=0.8))
overall_ind_b = float(tot_pct.get("Industry", 0))
ax.scatter([2023.0], [overall_ind_b], color="#E65100", marker="D", s=120, zorder=5)
ax.annotate("{:.0f}% overall\n2021–2025\n(n={})".format(
                overall_ind_b, int(tot_counts.sum())),
            (2023.0, overall_ind_b),
            textcoords="offset points", xytext=(22, 0), ha="left", fontsize=8.5,
            color="#E65100",
            arrowprops=dict(arrowstyle="-", color="#E65100", lw=0.8))
ax.set_xlim(2015.5, 2025.9)
ax.set_ylim(0, 96)
ax.set_xlabel("Cohort year")
ax.set_ylabel("% of graduates in Industry")
ax.set_title("Industry Employment Rate — ITCS Alumni 2016–2025\n"
             "(graduates only; benchmarks from 10,000 PhDs Project, Reithmeier et al. 2019)")
ax.set_xticks(list(hist_yrs) + list(new_yrs))
ax.yaxis.set_major_formatter(pct_fmt)
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_industry_trend.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Fig 3 saved.")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, group, label in zip(
    axes, ["Yes", "No"],
    ["Had paid industry internship", "No paid industry internship"],
):
    if group not in intern_pct.index:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(label)
        continue
    vals   = intern_pct.loc[group].reindex(SECTOR_ORDER, fill_value=0)
    colors = [PALETTE[s] for s in SECTOR_ORDER]
    n_grp  = int(intern_counts.loc[group].sum())
    bars   = ax.bar(range(len(SECTOR_ORDER)), vals.values,
                    color=colors, edgecolor="white", width=0.65)
    ax.set_title("{}\n(n={})".format(label, n_grp), fontsize=11)
    ax.set_xticks(range(len(SECTOR_ORDER)))
    ax.set_xticklabels(SECTOR_ORDER, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(pct_fmt)
    for bar, val in zip(bars, vals.values):
        if val >= 3:
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.5,
                    "{:.0f}%".format(val),
                    ha="center", va="bottom", fontsize=9)
axes[0].set_ylabel("% of graduates")
fig.suptitle("Effect of Paid Industry Internship on Employment Sector\n"
             "ITCS Graduates 2021–2025 (paid internships only; postdocs excluded)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_internship_effect.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Fig 4 saved.")
analysis_pos = [p for p in ["MSc", "MHSc", "PhD", "Postdoc"]
                if p in grad_df["position_std"].values]
pos_sector = (
    grad_df[grad_df["position_std"].isin(analysis_pos)]
    .groupby(["position_std", "sector_std"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=analysis_pos, columns=SECTOR_ORDER, fill_value=0)
)
pos_sector_pct = pos_sector.div(pos_sector.sum(axis=1), axis=0).mul(100)
fig, ax = plt.subplots(figsize=(9, 5))
bottom = np.zeros(len(pos_sector_pct))
for sec in SECTOR_ORDER:
    if sec not in pos_sector_pct.columns:
        continue
    vals = pos_sector_pct[sec].values
    bars = ax.bar(range(len(pos_sector_pct)), vals, bottom=bottom,
                  color=PALETTE[sec], label=sec, edgecolor="white", width=0.55)
    if sec == "Industry":
        for rect, val, bot in zip(bars, vals, bottom):
            if val >= 5:
                ax.text(rect.get_x() + rect.get_width() / 2.0,
                        bot + val / 2.0,
                        "{:.0f}%".format(val),
                        ha="center", va="center",
                        color="white", fontweight="bold", fontsize=12)
    bottom += vals
for i, (pos, n) in enumerate(zip(pos_sector_pct.index, pos_sector.sum(axis=1))):
    ax.text(i, -6, "n={}".format(int(n)), ha="center", fontsize=9.5, color="#444")
ax.set_xlim(-0.5, len(pos_sector_pct) - 0.5)
ax.set_ylim(-9, 112)
ax.set_xticks(range(len(pos_sector_pct)))
ax.set_xticklabels(pos_sector_pct.index, fontsize=12)
ax.set_xlabel("Trainee type")
ax.set_ylabel("% of graduates")
ax.set_title("Employment Sector by Trainee Type — ITCS Alumni 2021–2025\n"
             "(graduates only; MHSc tracked separately from MSc)")
ax.yaxis.set_major_formatter(pct_fmt)
legend_handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in SECTOR_ORDER]
ax.legend(handles=legend_handles, title="Sector",
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_sector_by_position.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Fig 5 saved.")
A_sec_pct = {
    "Industry":          HISTORICAL_TOTAL["Industry"],
    "Academia":          HISTORICAL_TOTAL["Academia/Healthcare"],
    "Non-profit":        HISTORICAL_TOTAL["Non-profit"],
    "Government":        HISTORICAL_TOTAL["Government"],
    "Further Schooling": 0,
    "Unemployed":        0,
    "Unknown":           HISTORICAL_TOTAL["Unknown"],
}
B_sec_pct = {s: float(tot_pct.get(s, 0)) for s in SECTOR_ORDER}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
datasets = [
    ("2016–2020\n(Sealey et al. 2021)", A_sec_pct, HISTORICAL_TOTAL["n_grad"],
     "#1565C0"),
    ("2021–2025\n(this analysis)", B_sec_pct, int(tot_counts.sum()),
     "#E65100"),
]
for ax, (title, data, n_grad, color) in zip(axes, datasets):
    vals   = [data.get(s, 0) for s in SECTOR_ORDER]
    colors = [PALETTE[s] for s in SECTOR_ORDER]
    bars   = ax.bar(range(len(SECTOR_ORDER)), vals, color=colors,
                    edgecolor="white", width=0.65)
    ax.set_title("{}\n(n={} graduates)".format(title, n_grad),
                 fontsize=11, color=color, fontweight="bold")
    ax.set_xticks(range(len(SECTOR_ORDER)))
    ax.set_xticklabels(SECTOR_ORDER, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_formatter(pct_fmt)
    for bar, val in zip(bars, vals):
        if val >= 2:
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.5,
                    "{:.0f}%".format(val),
                    ha="center", va="bottom", fontsize=10)
axes[0].set_ylabel("% of graduates")
fig.suptitle(
    "ITCS Alumni Employment Outcomes: 2016–2020 vs 2021–2025\n"
    "(graduates only; note 2016–2020 did not track Unemployed/Further Schooling separately)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_cohort_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Fig 6 saved.")
print("\n" + "=" * 70)
print("SUMMARY — ITCS 2026 RETROSPECTIVE")
print("=" * 70)
print("\nCohorts 2021–2025 (de-duplicated, first year per person)")
print("  Participation-years tracked : {}".format(n_dedup))
print("  Unique people (by canonical ID): {}".format(n_dedup))
print("  Graduates analysed          : {}".format(len(grad_df)))
print("  Still in training           : {}".format(
    (df_dedup["graduated_bool"] == False).sum()))
n_ind = int((grad_df["sector_std"] == "Industry").sum())
n_fs  = int((grad_df["sector_std"] == "Further Schooling").sum())
n_un  = int((grad_df["sector_std"] == "Unemployed").sum())
total_g = len(grad_df)
print("\n  Industry employment rate    : {}/{} = {:.1f}%".format(
    n_ind, total_g, 100.0 * n_ind / total_g if total_g else 0))
print("  Further Schooling           : {}/{} = {:.1f}%".format(
    n_fs, total_g, 100.0 * n_fs / total_g if total_g else 0))
print("  Unemployed                  : {}/{} = {:.1f}%".format(
    n_un, total_g, 100.0 * n_un / total_g if total_g else 0))
print("\n  For comparison:")
print("  Industry rate (Sealey 2021) : 58/106 = 55.0% (cohorts 2016–2020)")
print("  UofT PhD baseline (10k PhDs): ~20% industry (Reithmeier et al. 2019)")
print("  UofT life sci. subset       : ~25% industry (Reithmeier et al. 2019)")
n_intern_yes = int((grad_df["internship_std"] == "Yes").sum())
n_intern_no  = int((grad_df["internship_std"] == "No").sum())
print("\n  Paid internship — Yes       : {} ({:.0f}% of graduates)".format(
    n_intern_yes, 100.0 * n_intern_yes / total_g if total_g else 0))
print("  Paid internship — No        : {} ({:.0f}% of graduates)".format(
    n_intern_no, 100.0 * n_intern_no / total_g if total_g else 0))
print("\n  MHSc trainees (first year)  : {}".format(
    (df_dedup["position_std"] == "MHSc").sum()))
print("\nFigures in ./{}/".format(FIG_DIR))
for f in sorted(f for f in os.listdir(FIG_DIR) if f.startswith("fig")):
    print("  " + f)
print("\nDone.")
