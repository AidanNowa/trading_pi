#!/usr/bin/env bash
set -euo pipefail
cd /home/tradingpi/trading_bot

# Redirect everything to cron.log (append mode)
exec >> /home/tradingpi/trading_bot/cron.log 2>&1

echo "===== $(date -Is) start run ====="

PY="/home/tradingpi/trading_bot/.venv/bin/python"
TICKERS="AAPL MSFT GOOGL AMZN META NVDA TSLA AVGO AMD INTC SPY QQQ XOM JNJ"

$PY base_trading_bot.py $TICKERS --live

echo "===== $(date -Is) end run ====="
