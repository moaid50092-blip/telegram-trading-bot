import os
import time
import ccxt
import pandas as pd
import requests
import threading
import numpy as np
from enum import Enum

# --- الإعدادات الثابتة ---
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class Mode(Enum):
    DRY, LIVE = "DRY", "LIVE"

class MarketState(Enum):
    TRENDING, SPECULATIVE, BALANCED, CHOPPY, EXHAUSTED = "TRENDING", "SPECULATIVE", "BALANCED", "CHOPPY", "EXHAUSTED"

class ContextAI:
    @staticmethod
    def calculate_score(df_raw):
        df = df_raw.copy().dropna()
        if len(df) < 30: return 0
        last = df.iloc[-1]
        close = df['close']
        
        # الكفاءة والاستقرار والرفض (نفس المنطق المعتمد)
        net_move = abs(close.iloc[-1] - close.iloc[-10])
        total_path = close.diff().abs().iloc[-10:].sum()
        eff = (net_move / total_path) if (total_path > 0) else 0
        
        short_vol = close.diff().abs().iloc[-5:].std()
        long_vol = close.diff().abs().iloc[-50:].std()
        vol_score = 1 - min(short_vol / long_vol, 1) if long_vol > 0 else 0
        
        sma = close.rolling(20).mean()
        dist_score = max(0, 1 - (abs(last['close'] - sma.iloc[-1]) / (sma.iloc[-1] * 0.1))) if not sma.isna().all() else 0
        
        tr = last['high'] - last['low']
        wick = (last['high'] - max(last['open'], last['close'])) / tr if tr > 0 else 0
        rej_score = 1 - min(wick / 0.4, 1)

        return round((eff * 40 + vol_score * 15 + dist_score * 25 + rej_score * 20), 2)

class InstitutionalBot:
    def __init__(self, balance=1000, risk_pct=0.01, mode=Mode.DRY):
        self.balance = balance
        self.risk_pct = risk_pct
        self.mode = mode
        self.trades = {s: [] for s in SYMBOLS}
        self.block_list = {s: 0 for s in SYMBOLS}
        self.market_logs = {s: {"state": "Scanning", "score": 0, "reason": "Initializing..."} for s in SYMBOLS}
        self.exchange = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True})

    def notify(self, message):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": CHAT_ID, "text": f"🤖 {message}", "parse_mode": "Markdown"})
        except: pass

    def analyze_market(self, symbol, df):
        score = ContextAI.calculate_score(df)
        last = df.iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        
        state = MarketState.BALANCED
        reason = f"صمت: السياق ({score}) غير كافٍ"

        # الطبقة الأساسية (Trending)
        if score > 45 and last['close'] > df['close'].iloc[-15:-1].max():
            state = MarketState.TRENDING
            reason = "إيجابي: تريند نقي مكتمل الأركان"
        
        # طبقة المراقبة الاختيارية (Speculative) - المضافة حديثاً
        elif 35 <= score <= 45 and last['close'] > df['close'].iloc[-5:-1].max():
            state = MarketState.SPECULATIVE
            reason = "تجريبي: بوادر زخم (دخول صغير بنصف المخاطرة)"
        
        elif score < 25:
            state = MarketState.CHOPPY
            reason = "امتناع: ضجيج عالي"

        self.market_logs[symbol] = {"state": state.name, "score": score, "reason": reason}
        return state, score

    def run_cycle(self):
        total_active = sum(len(v) for v in self.trades.values())
        for symbol in SYMBOLS:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=70)
                df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'v'])
                last_price = df['close'].iloc[-1]
                
                state, score = self.analyze_market(symbol, df)
                
                # منطق الدخول
                if (state in [MarketState.TRENDING, MarketState.SPECULATIVE]) and total_active < 2:
                    if time.time() > self.block_list[symbol] and not self.trades[symbol]:
                        
                        # حساب المخاطرة بناءً على نوع الدخول
                        current_risk = self.risk_pct if state == MarketState.TRENDING else (self.risk_pct * 0.5)
                        stop = df['low'].iloc[-5:].min() if state == MarketState.TRENDING else df['low'].iloc[-2:].min()
                        
                        self.trades[symbol].append({"entry": last_price, "stop": stop, "type": state.name, "quality_score": 1.0})
                        self.notify(f"🚀 *دخول {state.name} في {symbol}*\nالسكور: `{score}`\nالمخاطرة: `{current_risk*100}%`")
                
                # إدارة الخروج (نفس منطقك المعتمد)
                self.manage_logic(symbol, last_price, score)
                
            except Exception as e: print(f"Error {symbol}: {e}")

    def manage_logic(self, symbol, current_price, context_score):
        for t in self.trades[symbol][:]:
            if current_price <= t['stop']:
                self.trades[symbol].remove(t)
                self.block_list[symbol] = time.time() + 14400 # حظر 4 ساعات
                self.notify(f"🛑 خروج {symbol} عند الستوب.")

# --- تشغيل البوت وواجهة تليجرام ---
# (استخدم نفس نظام الـ Threading المعتمد في رسائلك السابقة)
