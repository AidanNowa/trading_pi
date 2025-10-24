#!/usr/bin/env bash
set -euo pipefail
cd /home/tradingpi/trading_bot

# Redirect everything to cron.log (append mode)
exec >> /home/tradingpi/trading_bot/cron.log 2>&1

echo "===== $(date -Is) start run ====="

PY="/home/tradingpi/trading_bot/.venv/bin/python"

TICKERS="AAPL MSFT GOOGL GOOG AMZN META NVDA TSLA AVGO AMD INTC QCOM TXN ADI MU AMAT LRCX KLAC CRM NOW SNOW ORCL IBM CSCO PANW FTNT CRWD NET ZS OKTA SHOP PYPL SQ RBLX SNAP NFLX JPM BAC WFC C GS MS AXP V MA COF PYPL SCHW BK USB BLK SPGI ICE CME UNH JNJ PFE MRK ABBV LLY BMY AMGN GILD REGN TMO ISRG DXCM MDT SYK BSX WMT COST HD LOW TGT NKE MCD SBUX KO PEP PG UL CL KMB GIS MDLZ YUM DIS CMCSA XOM CVX COP SLB HAL PSX VLO CAT DE BA GE LMT RTX HON MMM EMR NOC GD SPY QQQ DIA IWM VTI XLK XLF XLV XLE XLY XLI XLU XLRE XLK"

$PY base_trading_bot.py $TICKERS --live

echo "===== $(date -Is) end run ====="
