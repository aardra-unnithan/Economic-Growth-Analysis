#!/usr/bin/env python
# coding: utf-8

# In[177]:


# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path


# In[178]:


# =============================================================
# Part 1 — Data Cleaning
# Countries: India (IND), Singapore (SGP), Niger (NER)
# Variables: GDP per capita (2015 USD), FDI (% GDP),
#            Population growth (%), Gross capital formation (% GDP)
# Output:   A panel dataset with Year as index and columns formatted
#           as Country_Indicator (e.g., India_GDPpc2015)
# =============================================================

# Define analysis window and country codes
YEAR_START, YEAR_END = 1980, 2024
keep_codes = ["IND", "SGP", "NER"]
code_to_name = {"IND": "India", "SGP": "Singapore", "NER": "Niger"}

# ------------------------------
# GDP per capita (constant 2015 USD)
# ------------------------------
gdp_raw = pd.read_csv("C:/Users/ardra/OneDrive/Desktop/MACRO PROJECT/GDP_per_capita_main.csv", skiprows=4)
# Filter by indicator and countries
gdp_raw = gdp_raw[gdp_raw["Indicator Code"] == "NY.GDP.PCAP.KD"]
gdp_raw = gdp_raw[gdp_raw["Country Code"].isin(keep_codes)]
# Keep only valid year columns (avoid extra unnamed columns)
gdp_years = [c for c in gdp_raw.columns if re.fullmatch(r"\d{4}", str(c))]
# Reshape to Year as index and countries as columns
gdp = gdp_raw.set_index("Country Code")[gdp_years].T
gdp.index = gdp.index.astype(int)
gdp = gdp.loc[YEAR_START:YEAR_END]
# Rename columns with country name + variable
gdp.columns = [f"{code_to_name[c]}_GDPpc2015" for c in gdp.columns]

# ------------------------------
# FDI, net inflows (% of GDP)
# ------------------------------
fdi_raw = pd.read_csv(r"C:/Users/ardra/OneDrive/Desktop/MACRO PROJECT/FDI inflow.csv", skiprows=4)
fdi_raw = fdi_raw[fdi_raw["Indicator Code"] == "BX.KLT.DINV.WD.GD.ZS"]
fdi_raw = fdi_raw[fdi_raw["Country Code"].isin(keep_codes)]
fdi_years = [c for c in fdi_raw.columns if re.fullmatch(r"\d{4}", str(c))]
fdi = fdi_raw.set_index("Country Code")[fdi_years].T
fdi.index = fdi.index.astype(int)
fdi = fdi.loc[YEAR_START:YEAR_END]
fdi.columns = [f"{code_to_name[c]}_FDI_pctGDP" for c in fdi.columns]

# ------------------------------
# Population growth (annual %)
# ------------------------------
pop_raw = pd.read_csv(r"C:/Users/ardra/OneDrive/Desktop/MACRO PROJECT/Population_Growth.csv", skiprows=4)
pop_raw = pop_raw[pop_raw["Indicator Code"] == "SP.POP.GROW"]
pop_raw = pop_raw[pop_raw["Country Code"].isin(keep_codes)]
pop_years = [c for c in pop_raw.columns if re.fullmatch(r"\d{4}", str(c))]
pop = pop_raw.set_index("Country Code")[pop_years].T
pop.index = pop.index.astype(int)
pop = pop.loc[YEAR_START:YEAR_END]
pop.columns = [f"{code_to_name[c]}_PopGrowth" for c in pop.columns]

# ------------------------------
# Gross capital formation (% of GDP)
# ------------------------------
gcf_raw = pd.read_csv(r"C:/Users/ardra/OneDrive/Desktop/MACRO PROJECT/Gross capital formation.csv", skiprows=4)
gcf_raw = gcf_raw[gcf_raw["Indicator Code"] == "NE.GDI.TOTL.ZS"]
gcf_raw = gcf_raw[gcf_raw["Country Code"].isin(keep_codes)]
gcf_years = [c for c in gcf_raw.columns if re.fullmatch(r"\d{4}", str(c))]
gcf = gcf_raw.set_index("Country Code")[gcf_years].T
gcf.index = gcf.index.astype(int)
gcf = gcf.loc[YEAR_START:YEAR_END]
gcf.columns = [f"{code_to_name[c]}_GCF_pctGDP" for c in gcf.columns]

# ===== Trade Openness (% of GDP) =====
trade_raw = pd.read_csv("C:/Users/ardra/OneDrive/Desktop/MACRO PROJECT/Trade_per_GDP.csv", skiprows=4)

# Keep only Trade (% of GDP) indicator
trade_raw = trade_raw[trade_raw["Indicator Code"] == "NE.TRD.GNFS.ZS"]

# Keep only selected countries
trade_raw = trade_raw[trade_raw["Country Code"].isin(keep_codes)]

# Extract valid year columns
trade_years = [c for c in trade_raw.columns if re.fullmatch(r"\d{4}", str(c))]

# Reshape into Year × Country
trade = trade_raw.set_index("Country Code")[trade_years].T
trade.index = trade.index.astype(int)
trade = trade.loc[YEAR_START:YEAR_END]

# Rename columns to Country_TradePctGDP
trade.columns = [f"{code_to_name[c]}_Trade_pctGDP" for c in trade.columns]


# ------------------------------
# Merge all variables into one panel dataset
# ------------------------------
panel = gdp.join(fdi, how="outer").join(pop, how="outer").join(gcf, how="outer")
panel.index.name = "Year"
panel = panel.sort_index()

# ------------------------------
# Output checks
# ------------------------------
print("Panel shape:", panel.shape)# dimensions of the panel
panel_part2 = panel.join(trade, how="outer")
print(panel_part2.head(10))                     # preview first 10 rows
print("Missing values per column:\n", panel.isna().sum())  # missing data summary

# Display full panel (Jupyter will render as table)
panel

panel_part2 = panel.copy()   # save as Part 2 dataset


# In[179]:


# Step 1: Select GDP per capita columns from the panel
gdp_cols = [c for c in panel.columns if c.endswith("_GDPpc2015")]

# Step 2: Compute annual log growth = 100 * [ln(Y_t) - ln(Y_{t-1})]
gdp_growth = np.log(panel[gdp_cols].astype(float)).diff() * 100

# Step 3: Rename columns clearly with "Annual" tag
gdp_growth.columns = [c.replace("_GDPpc2015", "_Annual_GDPpcGrowth") for c in gdp_growth.columns]

# Step 4: Keep only the three countries (Singapore, India, Niger) in order
gdp_growth = gdp_growth[[
    "Singapore_Annual_GDPpcGrowth",
    "India_Annual_GDPpcGrowth",
    "Niger_Annual_GDPpcGrowth"
]]

# Step 5: Set index name
gdp_growth.index.name = "Year"

# Preview the first 10 rows of the new table
print(gdp_growth.head(10))


# In[180]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: drop the first row (1980) since diff() makes it NaN
gdp_growth_clean = gdp_growth.dropna(how="all")

# Compute stats per country
stats = (
    gdp_growth_clean.agg(["mean", "std", "min", "max"]).T
      .rename(columns={
          "mean": "avg_growth_pct",
          "std":  "volatility_sd",
          "min":  "worst_year_pct",
          "max":  "best_year_pct"
      })
)

# Add tidy country names
stats.insert(0, "country", stats.index.str.replace("_Annual_GDPpcGrowth", "", regex=False))

# Rank (1 = highest)
stats["rank_fastest"]   = stats["avg_growth_pct"].rank(ascending=False).astype(int)
stats["rank_volatile"]  = stats["volatility_sd"].rank(ascending=False).astype(int)

# Round for display
print(stats.round(2))


# In[181]:


# =============================================================
# Part 2 — Visualize Growth Trajectories
# =============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

# Plot each country’s annual GDP per capita growth
plt.plot(gdp_growth.index, gdp_growth["Singapore_Annual_GDPpcGrowth"], label="Singapore")
plt.plot(gdp_growth.index, gdp_growth["India_Annual_GDPpcGrowth"], label="India")
plt.plot(gdp_growth.index, gdp_growth["Niger_Annual_GDPpcGrowth"], label="Niger")

# Labels & formatting
plt.title("Annual GDP per Capita Growth (%) — 1980–2024", fontsize=14, weight="bold")
plt.xlabel("Year")
plt.ylabel("Annual Growth Rate (%)")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # reference line
plt.legend()
plt.grid(alpha=0.3)

plt.show()


# In[182]:


fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True)

# Left: Annual GDP per capita growth
axes[0].plot(gdp_growth.index, gdp_growth["Singapore_Annual_GDPpcGrowth"], label="Singapore")
axes[0].plot(gdp_growth.index, gdp_growth["India_Annual_GDPpcGrowth"], label="India")
axes[0].plot(gdp_growth.index, gdp_growth["Niger_Annual_GDPpcGrowth"], label="Niger")
axes[0].set_title("Annual GDP per Capita Growth (%)")
axes[0].set_ylabel("Growth Rate (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: FDI inflows (% GDP)
axes[1].plot(panel.index, panel["Singapore_FDI_pctGDP"], label="Singapore")
axes[1].plot(panel.index, panel["India_FDI_pctGDP"], label="India")
axes[1].plot(panel.index, panel["Niger_FDI_pctGDP"], label="Niger")
axes[1].set_title("FDI Inflows (% of GDP)")
axes[1].set_ylabel("FDI (% of GDP)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("FDI and Economic Growth Trajectories", fontsize=14, weight="bold")
plt.show()



# In[183]:


plt.figure(figsize=(12,6))
plt.plot(panel.index, panel_part2["Singapore_GDPpc2015"], label="Singapore")
plt.plot(panel.index, panel_part2["India_GDPpc2015"], label="India")
plt.plot(panel.index, panel_part2["Niger_GDPpc2015"], label="Niger")

plt.yscale("log")  # log scale to show relative divergence clearly
plt.title("GDP per Capita (constant 2015 US$) — 1980–2024", fontsize=14, weight="bold")
plt.xlabel("Year")
plt.ylabel("GDP per Capita (log scale)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[184]:


for c in ["Singapore","India","Niger"]:
    fig, ax1 = plt.subplots(figsize=(10,5))

    # Left y-axis = GDP growth
    ax1.plot(gdp_growth.index, gdp_growth[f"{c}_Annual_GDPpcGrowth"], color="tab:blue", label="GDPpc Growth")
    ax1.set_ylabel("GDP per capita growth (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    # Right y-axis = GCF
    ax2 = ax1.twinx()
    ax2.plot(panel.index, panel[f"{c}_GCF_pctGDP"], color="tab:orange", linestyle="--", label="GCF")
    ax2.set_ylabel("Gross capital formation (% of GDP)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(f"{c}: GDP per Capita Growth vs Gross Capital Formation (1980–2024)", fontsize=13, weight="bold")
    fig.tight_layout()
    plt.show()


# In[185]:


# =============================================================
# Part 2 — One figure with all variables (India, Singapore, Niger)
# 3x2 grid: GDPpc growth + FDI + GCF + Pop growth + Trade (+ optional slot)
# =============================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- choose countries & variable suffixes (match your column names) ---
countries = ["India", "Singapore", "Niger"]
vars_cfg = [
    ("gdp_growth",      "Annual GDP per Capita Growth (%)",  "growth_pct"),           # from gdp_growth df
    ("FDI_pctGDP",      "FDI Inflows (% of GDP)",            "percent"),              # from panel
    ("GCF_pctGDP",      "Gross Capital Formation (% of GDP)","percent"),
    ("PopGrowth",       "Population Growth (%)",              "percent"),
    ("Trade_pctGDP",    "Trade Openness (% of GDP)",         "percent")
]

# --- x-axis (year) helper: use 'year' col if present, else index ---
def get_x(df):
    if isinstance(df, pd.DataFrame) and "year" in df.columns:
        return df["year"].values
    return df.index

x_panel  = get_x(panel)
x_grow   = get_x(gdp_growth)

# --- figure layout (3x2). last axis is optional/empty if you keep 5 panels ---
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=False)
axes = axes.flat

# ---- (1) GDP per capita growth (from gdp_growth df) ----
ax = axes[0]
for c in countries:
    col = f"{c}_Annual_GDPpcGrowth"
    if col in gdp_growth.columns:
        y = pd.to_numeric(gdp_growth[col], errors="coerce")
        mask = ~pd.isna(y)
        ax.plot(x_grow[mask], y[mask], label=c)
ax.set_title(vars_cfg[0][1]); ax.set_ylabel("Percent"); ax.grid(alpha=0.3)

# ---- Remaining variables from panel (lines for each country) ----
for ax, (suffix, title, ylab) in zip(axes[1:], vars_cfg[1:]):
    for c in countries:
        col = f"{c}_{suffix}"
        if col in panel.columns:
            y = pd.to_numeric(panel[col], errors="coerce")
            mask = ~pd.isna(y)
            ax.plot(x_panel[mask], y[mask], label=c)
    ax.set_title(title)
    ax.set_ylabel("Percent")
    ax.grid(alpha=0.3)

# If you end up with only 5 subplots, hide the 6th:
if len(vars_cfg) == 5 and len(axes) > 5:
    axes[-1].axis("off")

# --- shared X label + legend ---
for ax in axes:
    ax.set_xlabel("Year")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=10)
plt.suptitle("Part 2: Growth and Key Drivers — India, Singapore, Niger", fontsize=16, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[186]:


plt.figure(figsize=(8,6))
plt.scatter(panel["Singapore_FDI_pctGDP"], gdp_growth["Singapore_Annual_GDPpcGrowth"], label="Singapore", alpha=0.6)
plt.scatter(panel["India_FDI_pctGDP"], gdp_growth["India_Annual_GDPpcGrowth"], label="India", alpha=0.6)
plt.scatter(panel["Niger_FDI_pctGDP"], gdp_growth["Niger_Annual_GDPpcGrowth"], label="Niger", alpha=0.6)

plt.title("FDI vs Economic Growth", fontsize=14, weight="bold")
plt.xlabel("FDI Inflows (% of GDP)")
plt.ylabel("Annual GDP per Capita Growth (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[188]:


# =============================================================
# Part 3 — India: Growth Trends and Drivers (2x2 subplot figure)
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- pick x-axis as 'year' if available, else index
def get_x(df):
    if isinstance(df, pd.DataFrame) and "year" in df.columns:
        return df["year"].values
    return df.index

# --- ensure we have India's GDPpc growth series
# PRIORITY: use gdp_growth["India_Annual_GDPpcGrowth"] if it exists
# FALLBACK: compute from panel["India_GDPpc2015"] if available
if 'gdp_growth' in globals() and isinstance(gdp_growth, pd.DataFrame) \
   and "India_Annual_GDPpcGrowth" in gdp_growth.columns:
    x_growth = get_x(gdp_growth)
    ind_gdp_growth = gdp_growth["India_Annual_GDPpcGrowth"]
else:
    # compute from level if present
    if "India_GDPpc2015" in panel.columns:
        s = pd.to_numeric(panel["India_GDPpc2015"], errors="coerce")
        ind_gdp_growth = np.log(s).diff() * 100
        x_growth = get_x(panel)
    else:
        raise KeyError("Neither gdp_growth['India_Annual_GDPpcGrowth'] nor panel['India_GDPpc2015'] found.")

# --- helper to safely plot a column if it exists
def plot_if_exists(ax, df, col, title, ylab):
    if col in df.columns:
        ax.plot(get_x(df), pd.to_numeric(df[col], errors="coerce"))
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        return True
    else:
        ax.text(0.5, 0.5, f"Missing: {col}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return False

fig, axes = plt.subplots(2, 2, figsize=(14,10), sharex=False)

# ---- (1) Annual GDP per capita growth ----
axes[0,0].plot(x_growth, ind_gdp_growth, label="India")
axes[0,0].set_title("Annual GDP per Capita Growth (%)")
axes[0,0].set_ylabel("Growth Rate (%)")
axes[0,0].axhline(0, linewidth=0.8, linestyle="--")
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# ---- (2) FDI inflows (% of GDP) ----
# adjust the column name here to match your panel exactly
plot_if_exists(
    axes[0,1],
    panel,
    "India_FDI_pctGDP",          # e.g., "India_fdi_pct_gdp" if that’s your naming
    "FDI Inflows (% of GDP)",
    "FDI (% of GDP)"
)

# ---- (3) Gross Capital Formation (% of GDP) ----
plot_if_exists(
    axes[1,0],
    panel,
    "India_GCF_pctGDP",          # e.g., "India_gcf_pct_gdp"
    "Gross Capital Formation (% of GDP)",
    "GCF (% of GDP)"
)

# ---- (4) Population Growth (%) ----
plot_if_exists(
    axes[1,1],
    panel,
    "India_PopGrowth",           # e.g., "India_pop_growth_pct"
    "Population Growth (%)",
    "Population Growth (%)"
)

plt.suptitle("India: Growth Performance and Key Drivers", fontsize=16, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[189]:


#---------------------------------------------
# PART3
#---------------------------------------------

# India's annual GDP per capita growth
india_growth = gdp_growth["India_Annual_GDPpcGrowth"]

# Define periods
periods = {
    "1960–1990 (Pre-reforms)": (1960, 1990),
    "1991–2007 (Post-reforms)": (1991, 2007),
    "2008–2024 (Globalization & Crises)": (2008, 2024)
}

# Compute averages
india_avg_growth = {
    label: india_growth.loc[start:end].mean().round(2)
    for label, (start, end) in periods.items()
}

print("Average GDP per capita growth rates for India by period:")
for p, v in india_avg_growth.items():
    print(f"{p}: {v}%")


# In[190]:


# =============================================================
# Part 3 — India: Growth Trends and Drivers (2x2 subplot figure)
# =============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14,10), sharex=True)

# ---- (1) Annual GDP per capita growth ----
axes[0,0].plot(gdp_growth.index, gdp_growth["India_Annual_GDPpcGrowth"], color="blue", label="India")
axes[0,0].set_title("Annual GDP per Capita Growth (%)")
axes[0,0].set_ylabel("Growth Rate (%)")
axes[0,0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# ---- (2) FDI inflows (% of GDP) ----
axes[0,1].plot(panel.index, panel["India_FDI_pctGDP"], color="green", label="India")
axes[0,1].set_title("FDI Inflows (% of GDP)")
axes[0,1].set_ylabel("FDI (% of GDP)")
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# ---- (3) Gross Capital Formation (% of GDP) ----
axes[1,0].plot(panel.index, panel["India_GCF_pctGDP"], color="purple", label="India")
axes[1,0].set_title("Gross Capital Formation (% of GDP)")
axes[1,0].set_ylabel("GCF (% of GDP)")
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# ---- (4) Population Growth (%) ----
axes[1,1].plot(panel.index, panel["India_PopGrowth"], color="orange", label="India")
axes[1,1].set_title("Population Growth (%)")
axes[1,1].set_ylabel("Population Growth (%)")
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

# ---- Global formatting ----
plt.suptitle("India: Growth Performance and Key Drivers (1960–2024)", fontsize=16, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[192]:


# =============================================================
# India — GDP per capita growth vs Trade Openness (2-panel figure)
# =============================================================
import matplotlib.pyplot as plt

# --- Select series (assumes you already created gdp_growth and panel) ---
g_series = gdp_growth["India_Annual_GDPpcGrowth"]
t_series = panel["India_Trade_pctGDP"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ---- (1) Annual GDP per capita growth ----
axes[0].plot(g_series.index, g_series, label="India")
axes[0].set_title("India: Annual GDP per Capita Growth (%)")
axes[0].set_ylabel("Growth (%)")
axes[0].axhline(0, linewidth=0.8, linestyle="--")
axes[0].grid(alpha=0.3)
# period markers
for year in [1991, 2008]:
    axes[0].axvline(year, linewidth=0.8, linestyle="--")

# ---- (2) Trade openness (% of GDP) ----
axes[1].plot(t_series.index, t_series, label="India")
axes[1].set_title("India: Trade Openness (% of GDP)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Trade (% of GDP)")
axes[1].grid(alpha=0.3)
# period markers
for year in [1991, 2008]:
    axes[1].axvline(year, linewidth=0.8, linestyle="--")

plt.suptitle("India — Growth and Trade Integration (1960–2024)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[202]:


# ============================================================
# Build clean OECD panel (2000–2024) with no duplicates
# country | year | gdp_growth | investment_rate | employment_rate | education_tertiary | population_growth
# ============================================================

import pandas as pd
import os

# -----------------------------
# PATHS — EDIT ME
# -----------------------------
PATH_GDP_RAW = r"C:\Users\ardra\Downloads\OECD_GDP_rate.csv"
PATH_GCF_RAW = r"C:\Users\ardra\Downloads\OECD_GCFC %GDP.csv"
PATH_EMP_RAW = r"C:\Users\ardra\Downloads\OECD_ Employment.csv"
PATH_EDU_RAW = r"C:\Users\ardra\Downloads\Education attainment_oecd.csv"

# Tidy population growth you created earlier (WB)
PATH_POP_TIDY = r"C:\Users\ardra\OneDrive\Desktop\population oecd_OECD38_PopGrowth_tidy_2000_2024.csv"

OUT_PATH = r"C:\Users\ardra\OneDrive\Desktop\OECD_panel_final_clean.csv"

# -----------------------------
# SETTINGS
# -----------------------------
OECD38 = [
    "AUS","AUT","BEL","CAN","CHL","COL","CZE","DNK","EST","FIN",
    "FRA","DEU","GRC","HUN","ISL","IRL","ISR","ITA","JPN","KOR",
    "LVA","LTU","LUX","MEX","NLD","NZL","NOR","POL","PRT","SVK",
    "SVN","ESP","SWE","CHE","TUR","GBR","USA","CRI"
]
YEAR_START, YEAR_END = 2000, 2024

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def read_oecd_raw(path: str, new_name: str) -> pd.DataFrame:
    """
    Read a raw OECD CSV and output tidy columns: country | year | <new_name>
    Handles either REF_AREA/TIME_PERIOD/OBS_VALUE or their 'Reference area/Time period/Observation value' variants.
    """
    df = pd.read_csv(path)

    # detect column labels used by OECD exports
    cc = "REF_AREA" if "REF_AREA" in df.columns else ("Reference area" if "Reference area" in df.columns else None)
    yy = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else ("Time period" if "Time period" in df.columns else None)
    vv = "OBS_VALUE"  if "OBS_VALUE"  in df.columns else ("Observation value" if "Observation value" in df.columns else None)
    if not all([cc, yy, vv]):
        raise ValueError(
            f"{os.path.basename(path)} is not a recognized OECD format.\n"
            f"Found columns: {df.columns.tolist()[:12]}"
        )

    out = df[[cc, yy, vv]].copy()
    out.columns = ["country", "year", new_name]

    # enforce numeric types
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out[new_name] = pd.to_numeric(out[new_name], errors="coerce")

    # drop rows with missing keys
    out = out.dropna(subset=["country", "year"])

    # remove within-file duplicate country-year rows (average if multiple)
    out = (out.groupby(["country", "year"], as_index=False)[new_name]
              .mean(numeric_only=True))

    return out


def read_wb_tidy_population(path: str, new_name: str = "population_growth") -> pd.DataFrame:
    """
    Read your tidy World Bank population growth file and standardize to: country | year | population_growth
    Expected columns: 'Country Code' (or 'country'), 'year', 'population_growth'
    """
    df = pd.read_csv(path)

    # rename to standard keys if needed
    rename_map = {}
    if "Country Code" in df.columns: rename_map["Country Code"] = "country"
    if "year" not in df.columns and "Year" in df.columns: rename_map["Year"] = "year"
    if "population_growth" not in df.columns and new_name in df.columns:
        rename_map[new_name] = "population_growth"

    df = df.rename(columns=rename_map)

    # keep only needed columns
    needed = ["country", "year", "population_growth"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Population tidy file missing columns: {missing}. Found: {df.columns.tolist()}")

    df = df[needed].copy()

    # types + drop bad rows
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population_growth"] = pd.to_numeric(df["population_growth"], errors="coerce")
    df = df.dropna(subset=["country", "year"])

    # remove duplicates by averaging (if any)
    df = (df.groupby(["country", "year"], as_index=False)["population_growth"]
            .mean(numeric_only=True))

    return df


def keep_oecd_and_years(df: pd.DataFrame, varname: str) -> pd.DataFrame:
    """Filter to OECD-38, keep years 2000–2024, and sort."""
    df = df[df["country"].isin(OECD38)].copy()
    df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[varname] = pd.to_numeric(df[varname], errors="coerce")
    return df.sort_values(["country", "year"])


# -----------------------------
# 1) LOAD + CLEAN EACH DATASET (and de-dupe within each)
# -----------------------------
gdp = read_oecd_raw(PATH_GDP_RAW, "gdp_growth")
gcf = read_oecd_raw(PATH_GCF_RAW, "investment_rate")        # GFCF as % of GDP
emp = read_oecd_raw(PATH_EMP_RAW, "employment_rate")
edu = read_oecd_raw(PATH_EDU_RAW, "education_tertiary")     # NOTE: if this file has multiple breakdowns, we've already averaged within country-year here
pop = read_wb_tidy_population(PATH_POP_TIDY, "population_growth")

# -----------------------------
# 2) FILTER to OECD-38 and years 2000–2024
# -----------------------------
gdp = keep_oecd_and_years(gdp, "gdp_growth")
gcf = keep_oecd_and_years(gcf, "investment_rate")
emp = keep_oecd_and_years(emp, "employment_rate")
edu = keep_oecd_and_years(edu, "education_tertiary")
pop = keep_oecd_and_years(pop, "population_growth")

# -----------------------------
# 3) MERGE everything to one panel
# -----------------------------
panel = (
    gdp.merge(gcf, on=["country", "year"], how="outer")
       .merge(emp, on=["country", "year"], how="outer")
       .merge(edu, on=["country", "year"], how="outer")
       .merge(pop, on=["country", "year"], how="outer")
)

# -----------------------------
# 4) FINAL DE-DUPE SAFETY NET
#    If any duplicates remain after merging (e.g., from overlapping sources),
#    aggregate to a single row per (country, year) by taking the mean.
# -----------------------------
if panel.duplicated(subset=["country", "year"]).any():
    panel = (panel.groupby(["country", "year"], as_index=False)
                  .mean(numeric_only=True))

# sort nicely
panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

# -----------------------------
# 5) PREVIEW + SAVE
# -----------------------------
print("=== First 10 rows ===")
print(panel.head(10).to_string(index=False))

print("\n=== Last 10 rows ===")
print(panel.tail(10).to_string(index=False))

print("\n=== Panel Info ===")
print("Shape:", panel.shape)
print("Columns:", panel.columns.tolist())

panel.to_csv(OUT_PATH, index=False)
print("\nSaved:", OUT_PATH)


# In[201]:


import seaborn as sns

# Select only numeric variables for correlation
corr_vars = ["gdp_growth", "investment_rate", "employment_rate", 
             "education_tertiary", "population_growth"]

corr = panel[corr_vars].corr()

# Print correlation matrix
print("\nCorrelation matrix:")
print(corr)

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (OECD, 2000–2024)")
plt.show()



# In[198]:


import matplotlib.pyplot as plt

# Scatter plot: GDP growth vs Investment Rate
plt.scatter(panel["investment_rate"], panel["gdp_growth"], alpha=0.6)
plt.xlabel("Investment Rate (% of GDP)")
plt.ylabel("GDP Growth (%)")
plt.title("GDP Growth vs Investment Rate (OECD, 2000–2024)")
plt.show()

# Scatter plot: GDP growth vs Employment Rate
plt.scatter(panel["employment_rate"], panel["gdp_growth"], alpha=0.6, color="green")
plt.xlabel("Employment Rate (%)")
plt.ylabel("GDP Growth (%)")
plt.title("GDP Growth vs Employment Rate (OECD, 2000–2024)")
plt.show()

# Scatter plot: GDP growth vs Education (Tertiary)
plt.scatter(panel["education_tertiary"], panel["gdp_growth"], alpha=0.6, color="orange")
plt.xlabel("Education Tertiary (%)")
plt.ylabel("GDP Growth (%)")
plt.title("GDP Growth vs Education Tertiary (OECD, 2000–2024)")
plt.show()

# Scatter plot: GDP growth vs Population Growth
plt.scatter(panel["population_growth"], panel["gdp_growth"], alpha=0.6, color="red")
plt.xlabel("Population Growth (%)")
plt.ylabel("GDP Growth (%)")
plt.title("GDP Growth vs Population Growth (OECD, 2000–2024)")
plt.show()


# In[197]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your final merged panel
panel = pd.read_csv(r"C:\Users\ardra\OneDrive\Desktop\OECD_panel_final_5vars.csv")

# Pick representative countries
reps = ["USA", "DEU", "JPN", "MEX"]   # (you can add TUR/CHL if you want)
df_reps = panel[panel["country"].isin(reps)]

# -------------------------
# 1) Line plots of GDP vs. key variables
# -------------------------
for c in reps:
    temp = df_reps[df_reps["country"] == c]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{c} – Growth Determinants (2000–2024)", fontsize=14, weight="bold")

    # GDP growth
    ax[0,0].plot(temp["year"], temp["gdp_growth"], marker="o", label="GDP growth")
    ax[0,0].set_title("GDP Growth (%)")
    
    # Investment rate
    ax[0,1].plot(temp["year"], temp["investment_rate"], marker="s", color="darkgreen")
    ax[0,1].set_title("Investment (% GDP)")

    # Employment rate
    ax[1,0].plot(temp["year"], temp["employment_rate"], marker="^", color="purple")
    ax[1,0].set_title("Employment rate (%)")

    # Education
    ax[1,1].plot(temp["year"], temp["education_tertiary"], marker="d", color="orange")
    ax[1,1].set_title("Tertiary Education (%)")

    for row in ax:
        for a in row:
            a.grid(True, alpha=0.3)
            a.set_xlabel("Year")

    plt.tight_layout()
    plt.show()


# In[ ]:




