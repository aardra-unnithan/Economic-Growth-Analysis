# Economic-Growth-Analysis
Analysis of three countries at different growth stages and OECD growth, with a focus on country-specific trajectories.
**# Project structure**
#Main Python script containing all parts of the analysis:
#**Part 1:** Data cleaning for India, Singapore, and Niger (GDP per capita, FDI, population growth, gross capital formation, trade openness).
#**Part 2:** Growth trajectory analysis and cross-country visualizations.
#**Part 3:** Country-specific analysis for India (growth periods, reforms, key determinants).
#**Part 4:** OECD dataset cleaning and analysis (2000–2024).

#data/ → Raw datasets (World Bank WDI CSVs and OECD CSVs).
#outputs/ → Generated plots and summary tables.

**# Data Source**
**#World Bank (WDI)**
GDP per capita (constant 2015 US$) — NY.GDP.PCAP.KD
FDI inflows (% of GDP) — BX.KLT.DINV.WD.GD.ZS
Population growth (annual %) — SP.POP.GROW
Gross capital formation (% of GDP) — NE.GDI.TOTL.ZS
Trade openness (% of GDP) — NE.TRD.GNFS.ZS

**#Preprocessing:** Filtered to India, Singapore, Niger (1980–2024), reshaped to panel format, and computed annual log growth rates of GDP per capita.

**#OECD Database (2000–2024**)
GDP growth rate, investment rate (gross fixed capital formation % GDP), employment rate, tertiary education attainment, and population growth.

**Preprocessing:** Standardized variable names, dropped duplicates, filtered to OECD-38 countries, merged into one panel dataset, and exported as CSV.



