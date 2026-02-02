import os
import time
import ccxt
import pandas as pd
import requests
import threading
import numpy as np
from enum import Enum

# =========================================================
# ① الإعدادات والعملات المعتمدة
# =========================================================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class Mode(Enum):
    DRY, LIVE = "DRY", "LIVE"

class MarketState(Enum):
    TRENDING, BALANCED, CHOPPY, EXHAUSTED = "TRENDING", "BALANCED", "CHOPPY", "EXHAUSTED"

# =========================================================
# ② محرك الذكاء السياقي (Context AI)
# =========================================================
class ContextAI:
    @staticmethod
    def calculate_score(df):
        if len(df) < 50: return 0
        last = df.iloc[-1]
        close = df['close']
        
        net_move = abs(close.iloc[-1] - close.iloc[-10])
        total_path = close.diff().abs().iloc[-10:].sum()
        eff = (net_move / total_path) if total_path > 0 else 0
        
        short_vol = close.diff().abs().iloc[-5:].std()
        long_vol = close.diff().abs().iloc[-50:].std()
        vol_score = 1 - min(short_vol / long_vol, 1) if long_vol > 0 else 0
        
        sma = close.rolling(20).mean().iloc[-1]
        dist = abs(last['close'] - sma) / sma
        dist_score = max(0, 1 - (dist / 0.05))
        
        tr = last['high'] - last['low']
        wick = (last['high'] - max(last['open'], last['close'])) / tr if tr > 0 else 0
        rej_score = 1 - min(wick / 0.4, 1)

        return round((eff * 35 + vol_score * 25 + dist_score * 20 + rej_score * 20), 2)

# =========================================================
# ③ المحرك التنفيذي المحدث (تحديث التفسير في كل دورة)
# =========================================================
class InstitutionalBot:
    def __init__(self, balance=1000, risk_pct=0.01, mode=Mode.DRY):
        self.balance = balance
        self.risk_pct = risk_pct
        self.mode = mode
        self.trades = {s: [] for s in SYMBOLS}
        self.block_list = {s: 0 for s in SYMBOLS}
        
        # مخزن التفسير: يبدأ بوضع الانتظار
        self.market_logs = {s: {"state": "Scanning", "score": 0, "reason": "في انتظار أول دورة تشغيل..."} for s in SYMBOLS}
        
        self.exchange = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

    def notify(self, message):
        print(f"📡 {message}")
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": CHAT_ID, "text": f"🤖 {message}", "parse_mode": "Markdown"})
        except: pass

    def analyze_market(self, symbol, df):
        """دالة التحليل: تقوم بتحديث سجلات التفسير في كل مرة تُستدعى فيها"""
        score = ContextAI.calculate_score(df)
        last = df.iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        
        state = MarketState.BALANCED
        reason = "السوق في حالة توازن - لا توجد فرصة كفؤة حالياً"
        
        # منطق التفسير (Read-Only Observer Logic)
        if (last['close'] - sma_20) / sma_20 > 0.045: 
            state = MarketState.EXHAUSTED
            reason = "امتناع: السعر متمدد جداً (إنهاك شرائي)"
        elif score > 65 and last['close'] > df['close'].iloc[-15:-1].max(): 
            state = MarketState.TRENDING
            reason = "إيجابي: سياق تريند كفؤ ونقي"
        elif score < 40: 
            state = MarketState.CHOPPY
            reason = "امتناع: ضجيج عالي - خطر التذبذب العشوائي"
        elif score < 65:
            reason = f"صمت: جودة السياق ({score}) لم تصل للحد الأدنى (65)"

        # التحديث الحي لسجلات التفسير
        self.market_logs[symbol] = {"state": state.name, "score": score, "reason": reason}
        return state, score

    def run_cycle(self):
        """الدورة الأساسية: تضمن تحديث analyze_market لكل العملات فوراً"""
        total_active = sum(len(v) for v in self.trades.values())
        
        for symbol in SYMBOLS:
            try:
                # 1. جلب البيانات
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=50)
                df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'v'])
                last_price = df['close'].iloc[-1]
                
                # 2. تحديث التفسير (Observer Layer) في كل دورة بشكل إجباري
                state, score = self.analyze_market(symbol, df)
                
                # 3. إدارة الصفقات المفتوحة (إن وجدت)
                self.manage_logic(symbol, last_price, df, score)

                # 4. منطق الدخول (يبقى كما هو بدون تغيير)
                if state == MarketState.TRENDING and total_active < 2 and time.time() > self.block_list[symbol]:
                    can_enter = True
                    if self.trades[symbol]:
                        first = self.trades[symbol][0]
                        if abs(last_price - first['entry'])/first['entry'] < 0.015 or first['quality_score'] < 1.3:
                            can_enter = False
                    
                    if can_enter:
                        stop = df['low'].iloc[-5:].min()
                        self.trades[symbol].append({"entry": last_price, "stop": stop, "quality_score": 1.0, "initial_score": score})
                        self.notify(f"🚀 *دخول {symbol}*\nالدرجة: {score}/100\nالحالة: {'نادرة' if score > 85 else 'نقية'}")
            
            except Exception as e: print(f"Error {symbol}: {e}")
            time.sleep(1) # لضمان عدم الضغط على API

    def manage_logic(self, symbol, current_price, df, context_score):
        # نفس منطق الإدارة السابق (تأمين، خروج سلوكي، ستوب) بدون تغيير
        for t in self.trades[symbol][:]:
            t['quality_score'] = min(t['quality_score'] + 0.1, 2.5) if context_score > 65 else max(t['quality_score'] - 0.2, 0.5)
            if current_price > t['entry'] * 1.012 and t['stop'] < t['entry']:
                t['stop'] = t['entry']
                self.notify(f"🛡️ {symbol}: تأمين.")
            if context_score < 40 / t['quality_score']:
                self.notify(f"⚠️ {symbol}: خروج سلوكي (انهيار السياق).")
                self.trades[symbol].remove(t)
                continue
            if current_price <= t['stop']:
                if current_price < t['entry']:
                    self.block_list[symbol] = time.time() + (4 * 3600)
                self.trades[symbol].remove(t)

# =========================================================
# ④ واجهة تليجرام المحدثة
# =========================================================
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
                    msg = "🔍 **تقرير الشفافية (Live Market Context):**\n"
                    for s in SYMBOLS:
                        log = bot.market_logs[s]
                        block = "🚫 محظور" if time.time() < bot.block_list[s] else "✅ متاح"
                        msg += f"\n🪙 **{s}**\n  • الحالة: `{log['state']}`\n  • الجودة: `{log['score']}/100`\n  • التفسير: {log['reason']}\n  • التنفيذ: {block}\n"
                    bot.notify(msg)
                
                elif text == "/status":
                    active_count = sum(len(v) for v in bot.trades.values())
                    msg = f"📊 **ملخص الحساب:**\nالوضع: {bot.mode.value}\nالصفقات: {active_count}/2\n"
                    for s in SYMBOLS:
                        if bot.trades[s]:
                            t = bot.trades[s][0]
                            msg += f"🔹 {s}: سكور {round(t['quality_score'], 1)}\n"
                    bot.notify(msg)

                elif text == "/dry": bot.mode = Mode.DRY; bot.notify("🧪 وضع DRY مفعل")
                elif text == "/live": bot.mode = Mode.LIVE; bot.notify("⚠️ وضع LIVE مفعل")
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    my_bot = InstitutionalBot()
    threading.Thread(target=telegram_listener, args=(my_bot,), daemon=True).start()
    while True:
        my_bot.run_cycle()
        time.sleep(30) # دورة تحديث كل 30 ثانية
