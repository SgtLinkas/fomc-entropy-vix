<div align="center">
<img src="https://img.shields.io/badge/LMU%20Statistics-Research-blueviolet" alt="LMU Research">
<img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">

# fomc-entropy-vix

Empirical research project studying the **causal effect of FOMC communication
uncertainty on market volatility (VIX) 2015-2026**.

# Key Results: 

[![Key Result](charts/fomc_entropy_chart.png)](charts/fomc_entropy_chart.png)

# Key Question:
Does the way the Fed writes its statements predict how much markets move?

# What I Did
I scraped 89 FOMC press statements (2015–2026) from federalreserve.gov, computed Shannon entropy on each, and matched them to VIX changes around each meeting. 
The key control variable is the actual rate change in basis points (from FRED's DFEDTARU series), which I use to separate the "what they decided" effect from the "how they communicated" effect.

*Lukas Hübner | LMU München | BSc Statistics + Wirtschaftspsychologie*

</div>

## Repository layout

```
fomoc-entropy-vix/
├── analysis/
│   ├── entropy_calculation.py        # Shannon entropy & Loughran-McDonald uncertainty index
│   ├── iv_estimation.py              # 2SLS / IV estimation (entropy → VIX)
│   └── event_study.py               # Abnormal VIX change around FOMC meeting dates
├── charts/
│   └── fomc_entropy_chart.png
├── data/
│   ├── fetch_fomc_statements.py      # Scrape FOMC press statements from federalreserve.gov
│   ├── fetch_fed_funds_futures.py    # Download Fed Funds rate / futures data from FRED
│   └── fetch_vix.py 
├── notebooks/
│   └── fomc_entropy_vs_vix.ipynb    # Charts: entropy vs VIX
├── tests/
│   ├── test_entropy_calculation.py
│   ├── test_event_study.py
│   └── test_iv_estimation.py
├── README.md
└── requirements.txt
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data  (FRED API key required for fed-funds and VIX-FRED sources)
python data/fetch_fomc_statements.py --output data/fomc_statements.csv
python data/fetch_vix.py --source yahoo --output data/vix.csv
python data/fetch_fed_funds_futures.py \
    --api-key <YOUR_FRED_KEY> \
    --fomc-dates data/fomc_statements.csv \
    --output data/fed_funds_surprises.csv

# 3. Run analysis
python analysis/entropy_calculation.py
python analysis/event_study.py
python analysis/iv_estimation.py

# 4. Open the notebook
jupyter notebook notebooks/fomc_entropy_vs_vix.ipynb
```

## Running tests

```bash
pytest tests/
```

## Data sources

| Data | Source | Script |
|------|--------|--------|
| FOMC press statements | federalreserve.gov | `data/fetch_fomc_statements.py` |
| CBOE VIX | Yahoo Finance (`^VIX`) / FRED `VIXCLS` | `data/fetch_vix.py` |
| Fed Funds rate | FRED (series `FF`) | `data/fetch_fed_funds_futures.py` |

## Methodology

1. **Entropy** – Shannon entropy of the unigram token distribution in each
   FOMC post-meeting press statement.  Higher entropy → more diverse vocabulary
   → greater communication *complexity / uncertainty*.
2. **IV / 2SLS** – Fed Funds rate surprises (post − pre FOMC implied rate)
   serve as the excluded instrument for entropy to break potential endogeneity
   between communication and market conditions.
3. **Event study** – Abnormal log-changes in VIX in a ±10 trading-day window
   around each meeting, with a 120-day pre-event estimation window.
