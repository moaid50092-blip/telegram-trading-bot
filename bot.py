import os
import time
import ccxt
import pandas as pd
import requests
import threading
import numpy as np
from enum import Enum

--- الإعدادات (تأكد من وجودها في البيئة) ---

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class Mode(Enum):
DRY, LIVE = "DRY", "LIVE"

class MarketState(Enum):
TRENDING = "TRENDING"        # سكور > 45 + كسر 15 شمعة
SPECULATIVE = "SPECULATIVE"  # سكور 30-45 + كسر 7 شموع (تعديل جديد)
BALANCED = "BALANCED"        # سكور 25-30
CHOPPY = "CHOPPY"            # سكور < 25

class ContextAI:
@staticmethod
def calculate_score(df_raw):
df = df_raw.copy().dropna()
if len(df) < 20: return 15.0 # منع الصفر (حساب تدريجي)

last = df.iloc[-1]  
    close = df['close']  
      
    # 1. الكفاءة (Efficiency) - 40%  
    net_move = abs(close.iloc[-1] - close.iloc[-10])  
    total_path = close.diff().abs().iloc[-10:].sum()  
    eff = (net_move / total_path) if total_path > 0 else 0.1  
      
    # 2. التقلب (Volatility) - 15%  
    short_vol = close.diff().abs().iloc[-5:].std()  
    long_vol = close.diff().abs().iloc[-50:].std()  
    vol_score = (1 - min(short_vol / long_vol, 1)) if long_vol > 0 else 0.5  
      
    # 3. المسافة (Distance) - 25%  
    sma = close.rolling(20).mean()  
    dist_score = 0.5  
    if not sma.isna().all():  
        dist = abs(last['close'] - sma.iloc[-1]) / sma.iloc[-1]  
        dist_score = max(0, 1 - (dist / 0.12))  
          
    # 4. الرفض (Rejection) - 20%  
    tr = last['high'] - last['low']  
    wick = (last['high'] - max(last['open'], last['close'])) / tr if tr > 0 else 0  
    rej_score = 1 - min(wick / 0.5, 1)  

    final_score = (eff * 40 + vol_score * 15 + dist_score * 25 + rej_score * 20)  
    return round(max(final_score, 10.0), 2)

class InstitutionalBot:
def init(self, mode=Mode.DRY):
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
      
    state = MarketState.BALANCED  
    reason = f"انتظار: جودة غير كافية ({score})"  

    # المسار الأول: TRENDING ( Lookback = 15 )  
    if score > 45 and last['close'] > df['close'].iloc[-15:-1].max():  
        state = MarketState.TRENDING  
        reason = "دخول كامل: سياق مؤسسي قوي"  
      
    # المسار الثاني: SPECULATIVE ( Lookback = 7 ) - التعديل المطلوب  
    elif 30 <= score <= 45 and last['close'] > df['close'].iloc[-7:-1].max():  
        state = MarketState.SPECULATIVE  
        reason = "دخول تجريبي: كسر زخم قريب (Lookback 7)"  
          
    elif score < 25:  
        state = MarketState.CHOPPY  
        reason = "امتناع: ضجيج عالي"  

    self.market_logs[symbol] = {"state": state.value, "score": score, "reason": reason}  
    return state, score  

def run_cycle(self):  
    total_active = sum(len(v) for v in self.trades.values())  
    for symbol in SYMBOLS:  
        try:  
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=70)  
            df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'v'])  
            last_price = df['close'].iloc[-1]  
              
            state, score = self.analyze_market(symbol, df)  
              
            # إدارة الدخول  
            if state in [MarketState.TRENDING, MarketState.SPECULATIVE]:  
                if total_active < 3 and not self.trades[symbol] and time.time() > self.block_list[symbol]:  
                      
                    # الحفاظ على مخاطرة TRENDING (100%) و SPECULATIVE (50%)  
                    risk_multiplier = 1.0 if state == MarketState.TRENDING else 0.5  
                    stop_depth = 5 if state == MarketState.TRENDING else 2  
                      
                    stop = df['low'].iloc[-stop_depth:].min()  
                    self.trades[symbol].append({  
                        "entry": last_price,   
                        "stop": stop,   
                        "type": state.value,  
                        "risk": risk_multiplier  
                    })  
                    self.notify(f"🚀 *دخول {state.value} ({symbol})*\nالسكور: `{score}`\nالمخاطرة: `{risk_multiplier*100}%`")  

            self.manage_logic(symbol, last_price, score)  
              
        except Exception as e: print(f"Error {symbol}: {e}")  
        time.sleep(1.2)  

def manage_logic(self, symbol, current_price, score):  
    for t in self.trades[symbol][:]:  
        if current_price <= t['stop']:  
            self.trades[symbol].remove(t)  
            self.block_list[symbol] = time.time() + 14400 # حظر 4 ساعات  
            self.notify(f"🛑 خروج {symbol} (Stop Loss)")  
        elif score < 22: # خروج استباقي مرن  
            self.trades[symbol].remove(t)  
            self.notify(f"⚠️ خروج {symbol} (ضعف سياق)")

def telegram_listener(bot):
offset = None
while True:
try:
url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
r = requests.get(url, params={"timeout": 10, "offset": offset}).json()
for u in r.get("result", []):
offset = u["update_id"] + 1
text = u.get("message", {}).get("text", "")
if text == "/explain":
msg = "🔍 تقرير طبقة التفسير (Momentum Layer):\n"
for s in SYMBOLS:
log = bot.market_logs[s]
msg += f"\n🪙 {s}: {log['state']}\n  • الجودة: {log['score']}/100\n  • السبب: {log['reason']}\n"
bot.notify(msg)
except: pass
time.sleep(1)

if name == "main":
my_bot = InstitutionalBot()
threading.Thread(target=telegram_listener, args=(my_bot,), daemon=True).start()
while True:
my_bot.run_cycle()
time.sleep(30)
