# MarketScope

MarketScope is a simple Streamlit application to explore financial instruments.  
It provides three main views:

**Charts**  
   Multi-timeframe candlestick charts (1m, 15m, 4h, 1d) using Yahoo Finance data.

**Returns & Risk Dashboard**  
   Basic performance metrics for a selected ticker, including CAGR, annualized volatility, Sharpe ratio, and maximum drawdown.  
   Includes plots of the price and drawdown history.

**Backtest**  
   A moving-average crossover strategy (fast vs. slow SMA) on any supported timeframe.  
   Compares strategy results against buy-and-hold, with equity curves and performance metrics.
```
MarketScope/
├── app.py
├── requirements.txt
├── README.md
├── core/
│   ├── data.py
│   ├── metrics.py
│   ├── strategies.py
│   └── plotting.py
└── views/
    ├── charts.py
    ├── dashboard.py
    └── backtest.py
```
## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py