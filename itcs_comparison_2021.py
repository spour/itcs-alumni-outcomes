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
DATA_PATH = "Where are Trainees Now_ - Raw Data.csv"
FIG_DIR   = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
pct_fmt = FuncFormatter(lambda x, _: "{:.0f}%".format(x))
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
})
A_COHORT = pd.DataFrame({
    "year":    [2016, 2017, 2018, 2019, 2020],
    "n_total": [  11,   45,   41,   40,   47],
    "n_grad":  [  11,   34,   30,   19,   12],
    "Master":  [   0,   12,   19,   11,   19],
    "PhD":     [  11,   25,   18,   23,   17],
    "Postdoc": [   0,    6,    2,    2,    8],
    "PharmD":  [   0,    1,    0,    0,    0],
    "Other":   [   0,    1,    2,    4,    3],
    "Industry":            [64, 62, 43, 63, 42],
    "Academia/Healthcare": [27, 18, 43, 26, 42],
    "Non-profit":          [ 0,  3,  3,  5,  8],
    "Government":          [ 0,  3,  3,  5,  0],
    "Other_sec":           [ 0,  0,  0,  0,  0],
    "Unknown":             [ 9, 14,  8,  0,  8],
}).set_index("year")
A_TOTAL = {
    "n_total": 184, "n_grad": 106,
    "Master": 61, "PhD": 94, "Postdoc": 18, "PharmD": 1, "Other": 10,
    "Industry": 55, "Academia/Healthcare": 28,
    "Non-profit": 2, "Government": 1, "Other_sec": 0, "Unknown": 14,
}
SECTOR_ORDER_B = [
    "Industry", "Academia", "Further Schooling",
    "Non-profit", "Government", "Unemployed", "Unknown",
]
SECTOR_ORDER_A = [
    "Industry", "Academia/Healthcare", "Non-profit", "Government", "Unknown",
]
PALETTE = {
    "Industry":             "#1565C0",
    "Academia":             "#E65100",
    "Academia/Healthcare":  "#E65100",
    "Further Schooling":    "#00838F",
    "Non-profit":           "#6A1B9A",
    "Government":           "#AD1457",
    "Unemployed":           "#BF360C",
    "Unknown":              "#757575",
}
POS_ORDER   = ["MSc", "MHSc", "PhD", "Postdoc", "PharmD", "Other", "Unknown"]
POS_PALETTE = {
    "MSc":     "#43A047", "MHSc": "#00ACC1", "PhD": "#1E88E5",
    "Postdoc": "#FB8C00", "PharmD": "#8E24AA", "Other": "#6D4C41",
    "Unknown": "#90A4AE",
}
ERA_COLORS = {"2016–2020": "#1565C0", "2021–2025": "#E65100"}
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
blank_mask = (raw["name"].str.strip() == "") & \
             (raw["position"].str.strip().str.upper() == "FALSE")
df = raw[~blank_mask].copy()
df["year"] = pd.to_numeric(df["year"].str.strip(), errors="coerce")
df = df[df["year"].notna()].copy()
df["year"] = df["year"].astype(int)
NOT_FOUND_STRINGS = ["cannot find", "could not find", "can't find", "cant find"]
def is_not_found(row):
    li = str(row["linkedin"]).strip().lower()
    pos = str(row["position"]).strip().upper()
    grad = str(row["graduated"]).strip().upper()
    return (
        any(s in li for s in NOT_FOUND_STRINGS) or
        (pos == "FALSE" and grad == "FALSE" and li in ("", "false"))
    )
df["not_found"] = df.apply(is_not_found, axis=1)
ghost_mask = (
    (df["position"].str.strip().str.upper() == "FALSE") &
    (df["graduated"].str.strip().str.upper() == "FALSE") &
    (df["linkedin"].str.strip().str.lower().isin(["", "false"]))
)
df = df[~ghost_mask].copy()
def parse_grad(v):
    v = str(v).strip().upper()
    if v in ("TRUE", "YES", "1"):  return True
    if v in ("FALSE", "NO", "0"): return False
    return None
df["graduated_bool"] = df["graduated"].apply(parse_grad)
def map_pos(v):
    v = str(v).strip().lower()
    if v.startswith("mhsc"):    return "MHSc"
    if v.startswith("msc"):     return "MSc"
    if v.startswith("masc"):    return "MSc"
    if v.startswith("master"):  return "MSc"
    if v.startswith("phd"):     return "PhD"
    if v.startswith("postdoc"): return "Postdoc"
    if v.startswith("pharmd"):  return "PharmD"
    if v in ("false", ""):      return "Unknown"
    return "Other"
df["position_std"] = df["position"].apply(map_pos)
df.loc[df["position"].str.strip().str.upper() == "FALSE", "position_std"] = "Unknown"
SECTOR_RAW_MAP = {
    "industry": "Industry", "academia": "Academia",
    "academia/healthcare": "Academia", "non-profit": "Non-profit",
    "nonprofit": "Non-profit", "government": "Government",
    "unemployed": "Unemployed", "other": "_other_", "": "Unknown",
}
NONPROFIT_COMPANY_KW = (
    "parkinson", "cancer care", "sickkids", "sick kids", "obi",
    "ontario brain", "science collective", "rmic",
    "regenerative medicine institute", "mitacs", "cihr", "nserc",
    "hospital for sick", "sunnybrook", "unity health", "uhn ",
    "university health", "st. michael", "st michael",
    "women's college", "mount sinai", "trillium", "cancer research",
    "heart and stroke", "alzheimer", "ms society", "diabetes canada",
)
UNEMPLOYED_ROLE_KW = [
    "no current position", "no position", "not employed",
    "not working", "unemployed", "seeking", "between roles",
    "no current role", "idk", "seems unemployed", "looking for",
]
FS_ROLE_KW = [
    "jd candidate", "jd student", "jd ", "law school", "law student",
    "medical student", "md candidate", "md student", "dds", "dental school",
]
UNIV_KW = ("university", "uoft", "u of t", "college", "school", "medicine",
           "cornell", "mcmaster", "queens", "western", "mcgill", "toronto",
           "dalhousie", "alberta", "ottawa", "montreal", "medical",
           "harvard", "mit ", "columbia", "stanford", "oxford", "cambridge")
df["sector_raw"] = df["sector"].apply(
    lambda v: SECTOR_RAW_MAP.get(str(v).strip().lower(), "Unknown"))
def detect_further_schooling(row):
    role = str(row["role"]).strip().lower()
    company = str(row["company"]).strip().lower()
    pos = str(row["position"]).strip().lower()
    sec = row["sector_raw"]
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
def is_other_nonprofit(row):
    if row["sector_raw"] != "_other_":
        return False
    return any(kw in str(row["company"]).strip().lower() for kw in NONPROFIT_COMPANY_KW)
def detect_unemployed_other(row):
    if row["sector_raw"] != "_other_":
        return False
    if is_other_nonprofit(row):
        return False
    role = str(row["role"]).strip().lower()
    company = str(row["company"]).strip().lower()
    if role in ("idk", "n/a", "na", "", "unknown") and \
       company in ("idk", "n/a", "na", "", "unknown", "unemployed"):
        return True
    return any(kw in role or kw in company for kw in UNEMPLOYED_ROLE_KW)
df["further_schooling"]   = df.apply(detect_further_schooling, axis=1)
df["other_is_nonprofit"]  = df.apply(is_other_nonprofit, axis=1)
df["other_is_unemployed"] = df.apply(detect_unemployed_other, axis=1)
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
UNPAID_TOKENS = ("unpaid", "not paid", "no pay", "volunteer (", "voluntary")
UNDERGRAD_TOKENS = ("undergrad co-op", "undergrad coop", "undergrad intern",
                    "undergrad?", "undergrad only", ", undergrad", "undergrad")
YES_TOKENS = ("yes", "mitacs", "benchsci", "shift health", "j&j", "intern",
              "co-op", "coop", "practicum", "fellow", "halohealth", "analyst",
              "associate", "coordinator", "consultant", "manager", "eversana",
              "iqvia", "roche", "sanofi", "pfizer", "astrazeneca", "gsk",
              "novartis", "jnj", "obi", "xenon", "griffith", "sqi",
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
    m = re.search(r"linkedin\.com/in/([^/?#\s]+)", u)
    if m:
        return "linkedin.com/in/" + m.group(1)
    return ""
df["url_key"]  = df["linkedin"].apply(clean_url)
df["name_key"] = df["name"].str.strip().str.lower()
def canonical_id(row):
    u = row["url_key"]
    if u:
        return u
    n = row["name_key"]
    return n if n else "__blank__"
df["canon_id"] = df.apply(canonical_id, axis=1)
grad_best = df.groupby("canon_id")["graduated_bool"].apply(
    lambda s: True if True in s.values else (
        False if False in s.values else None))
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
grad_df = df_dedup[df_dedup["graduated_bool"] == True].copy()
B_cohort_n    = df_dedup.groupby("year").size()
B_cohort_grad = grad_df.groupby("year").size()
B_sector_counts = (
    grad_df.groupby(["year", "sector_std"]).size()
    .unstack(fill_value=0)
    .reindex(columns=SECTOR_ORDER_B, fill_value=0)
)
B_sector_pct = B_sector_counts.div(B_sector_counts.sum(axis=1), axis=0).mul(100)
B_ind_pct    = B_sector_pct.get("Industry", pd.Series(0, index=B_sector_pct.index))
B_pos = (
    df_dedup.groupby(["year", "position_std"]).size()
    .unstack(fill_value=0)
    .reindex(columns=POS_ORDER, fill_value=0)
)
B_TOTAL_n    = int(len(df_dedup))
B_TOTAL_grad = int(len(grad_df))
B_sec_totals  = B_sector_counts.sum()
B_sec_pct_tot = (B_sec_totals / B_sec_totals.sum() * 100).round(1)
B_pos_totals  = B_pos.sum()
B_overall_ind = float(B_sec_pct_tot.get("Industry", 0))
print("=" * 70)
print("ITCS COMPARISON: 2016–2020 vs 2021–2025")
print("=" * 70)
print("\nPeriod A (2016–2020, Sealey et al. 2021, Jan 2021 snapshot):")
print("  Participation-years : {}".format(A_TOTAL["n_total"]))
print("  Graduates analysed  : {}".format(A_TOTAL["n_grad"]))
print("  Industry %          : {}%".format(A_TOTAL["Industry"]))
print("\nPeriod B (2021–2025, this analysis, 2025–2026 snapshot):")
print("  Participation-years : {}".format(B_TOTAL_n))
print("  Graduates analysed  : {}".format(B_TOTAL_grad))
print("  Industry %          : {:.1f}%".format(B_overall_ind))
print("\nNote: Period A reports 'first employment after graduation';")
print("Period B reports 'current role' (may reflect career changes for 2021–2023).")
print("Period A used 'Academia/Healthcare' combined;")
print("Period B separates Academia, Further Schooling, and Unemployed.")
print("\nSector comparison:")
print("{:<26} {:>12} {:>12}".format("Sector", "2016–2020", "2021–2025"))
print("-" * 52)
sector_map_compare = [
    ("Industry", "Industry"),
    ("Academia/Healthcare", "Academia"),
    ("Non-profit", "Non-profit"),
    ("Government", "Government"),
    ("Further Schooling (new)", "Further Schooling"),
    ("Unemployed (new)", "Unemployed"),
    ("Unknown", "Unknown"),
]
for label_a, sec_b in sector_map_compare:
    a_pct = A_TOTAL.get(label_a.replace(" (new)", ""),
                         A_TOTAL.get(label_a.replace(" (new)", "") + "_sec", 0))
    b_pct = float(B_sec_pct_tot.get(sec_b, 0))
    if "(new)" in label_a:
        print("{:<26} {:>12} {:>11.1f}%".format(label_a, "not tracked", b_pct))
    else:
        delta = b_pct - a_pct
        print("{:<26} {:>11}% {:>11.1f}%  ({:+.1f} pp)".format(
            label_a, a_pct, b_pct, delta))
fig, ax = plt.subplots(figsize=(12, 5.5))
all_years   = list(A_COHORT.index) + list(B_cohort_n.index)
all_n_total = list(A_COHORT["n_total"]) + list(B_cohort_n.values)
all_n_grad  = list(A_COHORT["n_grad"]) + list(
    B_cohort_grad.reindex(B_cohort_n.index, fill_value=0).values)
colors = ([ERA_COLORS["2016–2020"]] * 5 +
          [ERA_COLORS["2021–2025"]] * len(B_cohort_n))
bars = ax.bar(all_years, all_n_total, color=colors, edgecolor="white", width=0.7)
for yr, n_tot, n_gr, bar in zip(all_years, all_n_total, all_n_grad, bars):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6, str(n_tot),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2, "{}↑".format(n_gr),
            ha="center", va="center", fontsize=8.5, color="white")
ax.set_xlabel("Cohort year")
ax.set_ylabel("Number of participants")
ax.set_title("ITCS Program Growth 2016–2025\n"
             "(bar height = total participants; white number = graduates;\n"
             " 2021–2025 de-duplicated: each person counted in first cohort year only)")
ax.set_xticks(all_years)
ax.set_ylim(0, max(all_n_total) * 1.28)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
legend_handles = [
    mpatches.Patch(color=ERA_COLORS["2016–2020"],
                   label="2016–2020 (Sealey et al. 2021, n={})".format(A_TOTAL["n_total"])),
    mpatches.Patch(color=ERA_COLORS["2021–2025"],
                   label="2021–2025 (this analysis, n={})".format(B_TOTAL_n)),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figC1_program_growth.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nFig C1 saved.")
A_pos_vals = {p: A_TOTAL.get(p, 0) for p in POS_ORDER}
A_pos_vals["MHSc"] = 0
B_pos_vals = {p: int(B_pos_totals.get(p, 0)) for p in POS_ORDER}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, label, pos_d, n_tot, color, note in [
    (axes[0], "2016–2020\n(Sealey et al. 2021)", A_pos_vals,
     A_TOTAL["n_total"], ERA_COLORS["2016–2020"], "(MHSc not tracked separately)"),
    (axes[1], "2021–2025\n(this analysis, de-duplicated)", B_pos_vals,
     B_TOTAL_n, ERA_COLORS["2021–2025"], "(MHSc tracked separately from MSc)"),
]:
    present = [(p, pos_d.get(p, 0)) for p in POS_ORDER if pos_d.get(p, 0) > 0]
    if not present:
        continue
    labels_p, sizes_p = zip(*present)
    pct_p    = [100.0 * s / n_tot for s in sizes_p]
    colors_p = [POS_PALETTE[p] for p in labels_p]
    wedges, texts, autotexts = ax.pie(
        sizes_p, labels=None, colors=colors_p, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    legend_labels = ["{} (n={}, {:.0f}%)".format(l, s, p)
                     for l, s, p in zip(labels_p, sizes_p, pct_p)]
    ax.legend(wedges, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=8.5, ncol=2)
    ax.set_title("{}\n(n={} total)\n{}".format(label, n_tot, note),
                 fontsize=11, fontweight="bold", color=color, pad=10)
fig.suptitle("Trainee Type Composition — 5-Year Period Comparison\n"
             "ITCS 2016–2025",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figC2_trainee_type_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Fig C2 saved.")
A_sec = {
    "Industry":  A_TOTAL["Industry"],
    "Academia":  A_TOTAL["Academia/Healthcare"],
    "Non-profit": A_TOTAL["Non-profit"],
    "Government": A_TOTAL["Government"],
    "Unknown":    A_TOTAL["Unknown"],
    "Further Schooling": 0,
    "Unemployed": 0,
}
B_sec = {s: float(B_sec_pct_tot.get(s, 0)) for s in SECTOR_ORDER_B}
fig, ax = plt.subplots(figsize=(10, 6.5))
x = np.array([0, 1])
bot_a = 0.0
bot_b = 0.0
for sec in SECTOR_ORDER_B:
    va = A_sec.get(sec, 0)
    vb = B_sec.get(sec, 0)
    if va == 0 and vb == 0:
        continue
    ax.bar([x[0]], [va], bottom=[bot_a], width=0.55,
           color=PALETTE[sec], edgecolor="white")
    ax.bar([x[1]], [vb], bottom=[bot_b], width=0.55,
           color=PALETTE[sec], edgecolor="white", label=sec)
    for xpos, val, bot in [(x[0], va, bot_a), (x[1], vb, bot_b)]:
        if val >= 3:
            ax.text(xpos, bot + val / 2, "{:.0f}%".format(val),
                    ha="center", va="center", color="white",
                    fontweight="bold", fontsize=11)
    bot_a += va
    bot_b += vb
for sec in ["Industry", "Non-profit", "Unknown"]:
    va = A_sec.get(sec, 0)
    vb = B_sec.get(sec, 0)
    if va == 0 and vb == 0:
        continue
    y_a = sum(A_sec.get(s, 0) for s in SECTOR_ORDER_B[:SECTOR_ORDER_B.index(sec)]) + va / 2
    y_b = sum(B_sec.get(s, 0) for s in SECTOR_ORDER_B[:SECTOR_ORDER_B.index(sec)]) + vb / 2
    delta = vb - va
    if abs(delta) >= 2:
        ax.annotate(
            "{:+.0f} pp".format(delta),
            xy=(x[1] - 0.29, y_b), xytext=(x[0] + 0.29, y_a),
            fontsize=8.5, color="#333", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
        )
note = ("Note: 2021–2025 includes 'Further Schooling'\n"
        "and 'Unemployed' as new categories not tracked\n"
        "in the 2016–2020 snapshot.")
ax.text(0.98, 0.97, note, transform=ax.transAxes, fontsize=8.5,
        va="top", ha="right", color="#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#ccc", alpha=0.9))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 115)
ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["2016–2020\n(Sealey et al. 2021)\nn={} graduates".format(A_TOTAL["n_grad"]),
     "2021–2025\n(this analysis)\nn={} graduates".format(B_TOTAL_grad)],
    fontsize=11)
ax.set_ylabel("% of graduates")
ax.set_title("Employment Sector — 5-Year Period Comparison\n"
             "(2016–2020: 'Academia/Healthcare' combined; "
             "2021–2025: split into sub-categories)", fontsize=11)
ax.yaxis.set_major_formatter(pct_fmt)
legend_handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in SECTOR_ORDER_B]
ax.legend(handles=legend_handles, title="Sector",
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figC3_sector_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Fig C3 saved.")
all_yrs  = list(A_COHORT.index) + list(B_sector_pct.index)
all_ind  = list(A_COHORT["Industry"]) + list(B_ind_pct.values)
all_acad = (list(A_COHORT["Academia/Healthcare"]) +
            list(B_sector_pct.get("Academia", pd.Series(0, index=B_sector_pct.index)).values))
all_np   = (list(A_COHORT["Non-profit"]) +
            list(B_sector_pct.get("Non-profit", pd.Series(0, index=B_sector_pct.index)).values))
all_unk  = (list(A_COHORT["Unknown"]) +
            list(B_sector_pct.get("Unknown", pd.Series(0, index=B_sector_pct.index)).values))
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
fig.suptitle("Employment Sector Trends — ITCS Alumni 2016–2025\n"
             "(Period A = Sealey et al. 2021 values; "
             "Period B 'Academia' ≈ 2016–2020 'Academia/Healthcare')",
             fontsize=11, fontweight="bold")
datasets = [
    (axes[0, 0], "Industry",              all_ind,  PALETTE["Industry"]),
    (axes[0, 1], "Academia(/Healthcare)", all_acad, PALETTE["Academia"]),
    (axes[1, 0], "Non-profit",            all_np,   PALETTE["Non-profit"]),
    (axes[1, 1], "Unknown",               all_unk,  PALETTE["Unknown"]),
]
for ax, label, vals, color in datasets:
    a_vals = vals[:5]
    b_vals = vals[5:]
    ax.plot(list(A_COHORT.index), a_vals, "o-", color=color, lw=2.5, ms=8,
            label="2016–2020 (Period A)")
    ax.plot(list(B_sector_pct.index), b_vals, "s--", color=color, lw=2.5, ms=8,
            alpha=0.8, label="2021–2025 (Period B)")
    avg_a = np.mean(a_vals)
    avg_b = np.mean(b_vals)
    ax.axhline(avg_a, color=color, ls=":", lw=1.4, alpha=0.6)
    ax.axhline(avg_b, color=color, ls="-.", lw=1.4, alpha=0.6)
    ax.text(2020.4, avg_a + 1, "A avg\n{:.0f}%".format(avg_a),
            color=color, fontsize=8, alpha=0.8)
    ax.text(2020.4, avg_b - 4, "B avg\n{:.0f}%".format(avg_b),
            color=color, fontsize=8, alpha=0.8)
    ax.axvspan(2015.5, 2020.5, alpha=0.05, color=ERA_COLORS["2016–2020"])
    ax.axvspan(2020.5, 2025.5, alpha=0.05, color=ERA_COLORS["2021–2025"])
    ax.set_title(label, fontsize=12, color=color, fontweight="bold")
    ax.yaxis.set_major_formatter(pct_fmt)
    ymax = max(max(a_vals) if a_vals else 0, max(b_vals) if b_vals else 0)
    ax.set_ylim(-2, ymax + 18)
    ax.set_xticks(all_yrs)
    ax.tick_params(axis="x", rotation=45)
    if ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("Cohort year")
    if ax in [axes[0, 0], axes[1, 0]]:
        ax.set_ylabel("% of graduates")
axes[0, 0].legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figC4_sector_trends.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Fig C4 saved.")
A_for_delta = {
    "Industry":          A_TOTAL["Industry"],
    "Academia":          A_TOTAL["Academia/Healthcare"],
    "Non-profit":        A_TOTAL["Non-profit"],
    "Government":        A_TOTAL["Government"],
    "Further Schooling": 0,
    "Unemployed":        0,
    "Unknown":           A_TOTAL["Unknown"],
}
deltas = {}
for s in SECTOR_ORDER_B:
    a = float(A_for_delta.get(s, 0))
    b = float(B_sec_pct_tot.get(s, 0))
    deltas[s] = round(b - a, 1)
fig, ax = plt.subplots(figsize=(10, 5.5))
xs     = range(len(SECTOR_ORDER_B))
vals_d = [deltas[s] for s in SECTOR_ORDER_B]
colors = [PALETTE[s] for s in SECTOR_ORDER_B]
bars = ax.bar(xs, vals_d, color=colors, edgecolor="white", width=0.65)
ax.axhline(0, color="#333", lw=1.2)
for i, (bar, val) in enumerate(zip(bars, vals_d)):
    ypos = bar.get_height() + 0.4 if val >= 0 else bar.get_height() - 1.2
    va   = "bottom" if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            "{:+.1f} pp".format(val),
            ha="center", va=va, fontsize=10.5, fontweight="bold",
            color=PALETTE[SECTOR_ORDER_B[i]])
ax.set_xticks(xs)
ax.set_xticklabels(SECTOR_ORDER_B, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Change in percentage points (2021–2025 minus 2016–2020)")
ax.set_title("Change in Employment Sector Distribution\n"
             "ITCS Alumni: 2021–2025 vs 2016–2020 (graduates only)", fontsize=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:+.0f} pp".format(x)))
note = ("Note: 'Further Schooling' and 'Unemployed' were not\n"
        "tracked in 2016–2020; their positive bars reflect\n"
        "new tracking methodology, not real changes.\n"
        "'Unknown' decrease = improved LinkedIn coverage.")
ax.text(0.98, 0.97, note, transform=ax.transAxes, fontsize=8.5,
        va="top", ha="right", color="#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#ccc", alpha=0.9))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figC5_sector_delta.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Fig C5 saved.")
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.axvspan(2015.6, 2020.4, alpha=0.07, color=ERA_COLORS["2016–2020"], label="_nolegend_")
ax.axvspan(2020.6, 2025.4, alpha=0.07, color=ERA_COLORS["2021–2025"], label="_nolegend_")
ax.plot(list(A_COHORT.index), list(A_COHORT["Industry"]),
        "o-", color=ERA_COLORS["2016–2020"], lw=2.5, ms=10,
        label="2016–2020 (Sealey et al. 2021, first employment after graduation)")
ax.plot(list(B_sector_pct.index), list(B_ind_pct.values),
        "s--", color=ERA_COLORS["2021–2025"], lw=2.5, ms=10,
        label="2021–2025 (this analysis, current role at time of data collection)")
avg_a = A_COHORT["Industry"].mean()
avg_b = B_ind_pct.mean()
ax.axhline(avg_a, color=ERA_COLORS["2016–2020"], lw=1.5, ls=":",
           label="Period A avg: {:.0f}%".format(avg_a))
ax.axhline(avg_b, color=ERA_COLORS["2021–2025"], lw=1.5, ls="-.",
           label="Period B avg: {:.0f}%".format(avg_b))
ax.axhline(20, color="#888", ls=":", lw=1.8,
           label="UofT PhD benchmark: ~20% industry — Reithmeier et al. 2019\n"
                 "(10,000 PhDs Project, all disciplines, 2000–2015 graduates)")
A_n = A_COHORT["n_grad"].values
B_n = B_sector_counts.sum(axis=1).values
for yr, pct, n in zip(list(A_COHORT.index), list(A_COHORT["Industry"]), A_n):
    ax.annotate("{:.0f}%\n(n={})".format(pct, n), (yr, pct),
                textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=8.5, color=ERA_COLORS["2016–2020"])
for yr, pct, n in zip(list(B_sector_pct.index), list(B_ind_pct.values), B_n):
    ax.annotate("{:.0f}%\n(n={})".format(pct, n), (yr, pct),
                textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=8.5, color=ERA_COLORS["2021–2025"])
ax.scatter([2018.0], [A_TOTAL["Industry"]], marker="D", s=160,
           color=ERA_COLORS["2016–2020"], zorder=5)
ax.annotate("Overall A: {}%\n(n={} grads)".format(
                A_TOTAL["Industry"], A_TOTAL["n_grad"]),
            (2018.0, A_TOTAL["Industry"]),
            textcoords="offset points", xytext=(-48, -30),
            fontsize=8.5, color=ERA_COLORS["2016–2020"],
            arrowprops=dict(arrowstyle="->", color=ERA_COLORS["2016–2020"], lw=0.8))
ax.scatter([2023.0], [B_overall_ind], marker="D", s=160,
           color=ERA_COLORS["2021–2025"], zorder=5)
ax.annotate("Overall B: {:.0f}%\n(n={} grads)".format(B_overall_ind, B_TOTAL_grad),
            (2023.0, B_overall_ind),
            textcoords="offset points", xytext=(32, -30),
            fontsize=8.5, color=ERA_COLORS["2021–2025"],
            arrowprops=dict(arrowstyle="->", color=ERA_COLORS["2021–2025"], lw=0.8))
ax.set_xlim(2015.3, 2025.7)
ax.set_ylim(0, 98)
ax.set_xticks(all_yrs)
ax.set_xlabel("Cohort year")
ax.set_ylabel("% of graduates in Industry")
ax.set_title("Industry Employment Rate — ITCS Alumni 2016–2025 (Unified View)\n"
             "vs UofT PhD Benchmark (Reithmeier et al. 2019)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(pct_fmt)
ax.text(2018, 5, "Period A", ha="center", color=ERA_COLORS["2016–2020"],
        fontsize=10, alpha=0.6, style="italic")
ax.text(2023, 5, "Period B", ha="center", color=ERA_COLORS["2021–2025"],
        fontsize=10, alpha=0.6, style="italic")
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figC6_industry_10yr_unified.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Fig C6 saved.")
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print("\n{:<30} {:>12} {:>12} {:>14}".format(
    "Metric", "2016–2020", "2021–2025", "Change"))
print("-" * 70)
rows = [
    ("Participation-years",
     str(A_TOTAL["n_total"]), str(B_TOTAL_n),
     "{:+d} ({:+.0f}%)".format(
         B_TOTAL_n - A_TOTAL["n_total"],
         100*(B_TOTAL_n - A_TOTAL["n_total"])/A_TOTAL["n_total"])),
    ("Graduates analysed",
     str(A_TOTAL["n_grad"]), str(B_TOTAL_grad),
     "{:+d}".format(B_TOTAL_grad - A_TOTAL["n_grad"])),
    ("Avg cohort size",
     "{:.0f}".format(A_TOTAL["n_total"]/5),
     "{:.0f}".format(B_TOTAL_n/5),
     "{:+.0f}".format(B_TOTAL_n/5 - A_TOTAL["n_total"]/5)),
    ("% Industry",
     "{}%".format(A_TOTAL["Industry"]),
     "{:.0f}%".format(B_overall_ind),
     "{:+.0f} pp".format(B_overall_ind - A_TOTAL["Industry"])),
    ("% Academia(/Healthcare)",
     "{}%".format(A_TOTAL["Academia/Healthcare"]),
     "{:.0f}%".format(float(B_sec_pct_tot.get("Academia", 0))),
     "{:+.0f} pp (methodology change)".format(
         float(B_sec_pct_tot.get("Academia", 0)) -
         A_TOTAL["Academia/Healthcare"])),
    ("% Non-profit",
     "{}%".format(A_TOTAL["Non-profit"]),
     "{:.0f}%".format(float(B_sec_pct_tot.get("Non-profit", 0))),
     "{:+.0f} pp".format(
         float(B_sec_pct_tot.get("Non-profit", 0)) - A_TOTAL["Non-profit"])),
    ("% Further Schooling (new)",
     "not tracked",
     "{:.0f}%".format(float(B_sec_pct_tot.get("Further Schooling", 0))), ""),
    ("% Unemployed (new)",
     "not tracked",
     "{:.0f}%".format(float(B_sec_pct_tot.get("Unemployed", 0))), ""),
    ("% Unknown",
     "{}%".format(A_TOTAL["Unknown"]),
     "{:.0f}%".format(float(B_sec_pct_tot.get("Unknown", 0))),
     "{:+.0f} pp (better coverage)".format(
         float(B_sec_pct_tot.get("Unknown", 0)) - A_TOTAL["Unknown"])),
]
for label, va, vb, delta in rows:
    print("{:<30} {:>12} {:>12} {:>14}".format(label, va, vb, delta))
print("\nTrainee type mix:")
for p, pa in [("MSc+MHSc", "Master"), ("PhD", "PhD"), ("Postdoc", "Postdoc")]:
    na = A_TOTAL.get(pa, 0)
    if p == "MSc+MHSc":
        nb = int(B_pos_totals.get("MSc", 0)) + int(B_pos_totals.get("MHSc", 0))
    else:
        nb = int(B_pos_totals.get(p, 0))
    pct_a = 100.0 * na / A_TOTAL["n_total"]
    pct_b = 100.0 * nb / B_TOTAL_n if B_TOTAL_n else 0
    print("  {:<10} A: {} ({:.0f}%)   B: {} ({:.0f}%)   Δ{:+.0f} pp".format(
        p, na, pct_a, nb, pct_b, pct_b - pct_a))
print("    Note: Period B MHSc tracked separately within 'MSc+MHSc' total.")
print("\nFigures written to ./{}/:".format(FIG_DIR))
for f in sorted(f for f in os.listdir(FIG_DIR) if f.startswith("figC")):
    print("  " + f)
print("\nDone.")
