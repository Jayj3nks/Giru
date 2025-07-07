import asyncio
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import discord
from discord.ext import commands
import openai
from openai import OpenAI, RateLimitError, OpenAIError

# === Configuration ===
DISCORD_TOKEN  = "MTM5MTYzMTY1MTIxNzM0MjQ4NA.GtSRFE.YpnwHX-HYxWak0j1pamM3rHRX2gvnAQHIST7cg"
OPENAI_API_KEY = "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"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

EXCHANGE_CONFIGS = [
    {'id': 'binanceus', 'params': {'enableRateLimit': True}},
    {'id': 'kucoin',    'params': {'enableRateLimit': True}},
    {'id': 'bitrue',    'params': {'enableRateLimit': True}},
    {'id': 'coinbase',  'params': {'enableRateLimit': True}}
]
SYMBOL_MAP = {
    'binanceus': 'BTC/USDT',
    'kucoin':    'BTC/USDT',
    'bitrue':    'BTC/USDT',
    'coinbase':  'BTC/USD'
}
INTERVAL    = '1m'
LOOKBACK_HR = 24
BINS        = 50
LIMIT       = LOOKBACK_HR * 60

async def fetch_ohlcv(exchange, symbol, since_ms, limit):
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe=INTERVAL, since=since_ms, limit=limit)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        print(f"Error fetching OHLCV from {exchange.id}: {e}")
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])

async def fetch_funding_rate(exchange, symbol):
    try:
        rate = await exchange.fetch_funding_rate(symbol)
        return rate.get('fundingRate')
    except Exception:
        return None


def compute_buy_sell_profiles(df, bins, p_min, p_max):
    edges = np.linspace(p_min, p_max, bins+1)
    buy_vp = np.zeros(bins)
    sell_vp = np.zeros(bins)
    for _, r in df.iterrows():
        vol = r['volume']; buy = r['close'] >= r['open']
        start = np.searchsorted(edges, r['low'], side='right') - 1
        end   = np.searchsorted(edges, r['high'], side='left')
        if end <= start:
            idx = np.clip(start, 0, bins-1)
            (buy_vp if buy else sell_vp)[idx] += vol
        else:
            share = vol / (end - start)
            for i in range(start, end):
                (buy_vp if buy else sell_vp)[i] += share
    return buy_vp, sell_vp


def compute_indicators(df):
    out = pd.DataFrame(index=df.index)
    close = df['close']
    high, low, vol = df['high'], df['low'], df['volume']
    out['sma_20'] = close.rolling(20).mean()
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out['rsi_14'] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    out['atr_14'] = tr.rolling(14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()
    tr_smooth = tr.rolling(14).sum()
    pdi = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
    mdi = 100 * (minus_dm.rolling(14).sum() / tr_smooth)
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi)).replace(np.nan, 0)
    out['adx_14'] = dx.rolling(14).mean()
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    out['obv'] = obv
    return out

async def analyze_market():
    clients = [(eid, getattr(ccxt, eid)(conf)) for eid, conf in [(c['id'], c['params']) for c in EXCHANGE_CONFIGS]]
    since = int((datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HR)).timestamp() * 1000)
    tasks = [fetch_ohlcv(client, SYMBOL_MAP[eid], since, LIMIT) for eid, client in clients]
    dfs = await asyncio.gather(*tasks)
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        return None
    all_df = pd.concat(dfs).sort_values('timestamp')
    p_min, p_max = all_df['low'].min(), all_df['high'].max()
    buys, sells = [], []
    for df in dfs:
        b, s = compute_buy_sell_profiles(df, BINS, p_min, p_max)
        buys.append(b); sells.append(s)
    avg_buy = np.mean(buys, axis=0)
    avg_sell = np.mean(sells, axis=0)
    total_vp = avg_buy + avg_sell
    centers = (np.linspace(p_min, p_max, BINS+1)[:-1] + np.diff(np.linspace(p_min, p_max, BINS+1))/2)
    poc = centers[np.argmax(total_vp)]
    sorted_idx = np.argsort(total_vp)[::-1]
    cum, target, va_idxs = 0, total_vp.sum()*0.7, []
    for i in sorted_idx:
        cum += total_vp[i]; va_idxs.append(i)
        if cum >= target: break
    val, vah = centers[min(va_idxs)], centers[max(va_idxs)]
    ind = compute_indicators(all_df)
    funding = {}
    for eid, client in clients:
        rate = await fetch_funding_rate(client, SYMBOL_MAP[eid])
        funding[eid] = rate
        await client.close()
    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'buy_profile': avg_buy.tolist(),
        'sell_profile': avg_sell.tolist(),
        'indicators': ind.tail(1).to_dict(orient='records')[0],
        'funding_rates': funding
    }

# === Discord + GPT Integration ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

GIRU_PROMPT = """
You are Giru, a cocky, no-nonsense BTC day trader. Based on the market payload below, deliver a Discord-ready trade call with bold formatting, alpha tone, and swagger:
- Give a LONG or SHORT call
- Include entry, stop (1.5x ATR), target (2x ATR), risk/reward
- Justify with volume profile (POC/VAH/VAL), RSI, funding rates, etc
- Format for Discord using bolds, emojis, and punchy language
"""

@bot.command(name='trade')
async def trade(ctx):
    await ctx.send("📊 Giru’s scanning the markets... hold tight.")
    payload = await analyze_market()
    if not payload:
        await ctx.send("⚠️ Market data fetch failed.")
        return
    # OpenAI call with correct exception types
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": GIRU_PROMPT},
                {"role": "user",   "content": f"{payload}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
    except RateLimitError as e:
        msg = str(e)
        if 'insufficient_quota' in msg.lower() or 'quota' in msg.lower():
            await ctx.send("⚠️ Insufficient OpenAI quota. Please add funds to your OpenAI account to continue using this command.")
        else:
            await ctx.send("⚠️ OpenAI rate limit exceeded. Please try again later.")
        return
    except OpenAIError as e:
        await ctx.send(f"❌ OpenAI API error: {e}")
        return
    except Exception as e:
        await ctx.send(f"❌ Unexpected error: {e}")
        return
    giru_response = response.choices[0].message.content.strip()
    await ctx.send(giru_response)

if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)
