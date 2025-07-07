# Advanced Giru Bot

This repository contains an **advanced, full-featured** version of the original `giru.py` for getting volume profilst. Building on the MVP, this iteration adds robust error handling, a modern OpenAI client integration, and a dual-mode “persona” interface so the bot only speaks when invoked.

---

## 🚀 Key Features

- **Multi-exchange, 1m OHLCV aggregation** from Binance US, Kucoin, Bitrue, and Coinbase (spot).
- **Composite Volume Profile** (50 bins) → POC, VAH, VAL calculation.
- **Technical Indicator Suite**: 20-period SMA, 14-period RSI, 14-period ATR, 14-period ADX, On-Balance Volume.
- **Funding-Rate Fetching** per exchange.
- **Modern OpenAI SDK (v1.x)** with proper async `.create()` calls.
- **Graceful error handling** for rate limits, quota issues, and unexpected API failures.
- **“Giru” persona** for two modes:
  - `!trade` command → swaggering LONG/SHORT calls with entry, stop, target, R/R justified by profile & indicators.
  - Casual replies only when the bot is **@-mentioned**, stripping out the mention and carrying the same cocky tone.
- **Timezone-aware UTC** timestamps and safe resource cleanup.

---

## 📦 Installation

1. **Clone** this repo:  
   ```bash
   git clone https://github.com/your-org/advanced-giru-bot.git
   cd advanced-giru-bot
