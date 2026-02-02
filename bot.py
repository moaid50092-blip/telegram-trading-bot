import os
import time
import ccxt
import pandas as pd
import requests
import threading
from enum import Enum

# =========================================================
# ① الإعدادات والربط
# =========================================================
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class Mode(Enum):
    DRY, LIVE = "DRY", "LIVE"

class MarketState(Enum):
    TRENDING, BALANCED, CHOPPY, EXHAUSTED = "TRENDING", "BALANCED", "CHOPPY", "EXHAUSTED"

class TradeState(Enum):
    IDLE, IN_TRADE, BLOCKED = "IDLE", "IN_TRADE", "BLOCKED"

# =========================================================
# ② العقل الاستراتيجي (Cumulative Quality Scoring)
# =========================================================
class BehavioralTradingBot:
    def __init__(self, symbol="BTC/USDT", balance=1000, risk_pct=0.01, mode=Mode.DRY):
        self.symbol = symbol
        self.balance = balance
        self.risk_pct = risk_pct
        self.mode = mode
        self.trade_state = TradeState.IDLE
        self.trades = []
        self.block_until = 0 

        self.exchange = ccxt.binance({
            'apiKey': API_KEY, 'secret': API_SECRET,
            'enableRateLimit': True, 'options': {'defaultType': 'spot'}
        })

    def notify(self, message):
        print(f"📡 {message}")
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": CHAT_ID, "text": f"🤖 {message}"})
        except: pass

    def analyze_market(self, data):
        """تحليل السياق المستقل وكفاءة الحركة"""
        if len(data) < 30: return MarketState.CHOPPY, "نقص بيانات"
        last = data.iloc[-1]
        
        net_change = abs(data['close'].iloc[-1] - data['close'].iloc[-10])
        total_movement = data['close'].diff().abs().iloc[-10:].sum()
        efficiency_ratio = net_change / total_movement if total_movement > 0 else 0

        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        dist = (last['close'] - sma_20) / sma_20
        if dist > 0.045: return MarketState.EXHAUSTED, "إنهاك"

        total_range = last['high'] - last['low']
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        has_rejection = (upper_wick > total_range * 0.4) or (lower_wick > total_range * 0.4)

        if efficiency_ratio < 0.5 or has_rejection:
            return MarketState.CHOPPY, "ضوضاء/رفض"

        if last['close'] > data['close'].iloc[-15:-1].max() and efficiency_ratio > 0.65:
            return MarketState.TRENDING, "سياق كفؤ"

        return MarketState.BALANCED, "توازن"

    def execute_order(self, side, price, stop, reason):
        if time.time() < self.block_until: return
        risk_usd = self.balance * self.risk_pct
        dist = abs(price - stop)
        size = risk_usd / dist if dist > 0 else 0
        size_prec = self.exchange.amount_to_precision(self.symbol, size)
        
        self.notify(f"🚀 {reason}\nالسعر: {price} | الستوب: {stop}")
        if self.mode == Mode.LIVE:
            try:
                order = self.exchange.create_order(self.symbol, 'market', side, size_prec)
            except Exception as e:
                self.notify(f"❌ خطأ: {e}")
                return

        # يبدأ السكور عند 1.0 ويزيد مع كل شمعة ناجحة
        self.trades.append({"entry": price, "stop": stop, "size": size_prec, "quality_score": 1.0})
        self.trade_state = TradeState.IN_TRADE

    def manage_logic(self, current_price, data):
        """تطوير السكور التراكمي وإدارة الخروج"""
        state, _ = self.analyze_market(data)
        
        for t in self.trades[:]:
            # 1. بناء السكور التراكمي (Cumulative Growth)
            if state == MarketState.TRENDING:
                t['quality_score'] = min(t['quality_score'] + 0.1, 2.5) # ينمو السكور بحد أقصى 2.5
            else:
                t['quality_score'] = max(t['quality_score'] - 0.2, 0.5) # ينخفض بسرعة عند التذبذب

            # 2. التأمين (Break-even)
            if current_price > t['entry'] * 1.012 and t['stop'] < t['entry']:
                t['stop'] = t['entry']
                self.notify("🛡️ تأمين: الوقف عند الدخول.")

            # 3. الخروج السلوكي المرتبط بالسكور (Dynamic Threshold)
            net_c = abs(data['close'].iloc[-1] - data['close'].iloc[-5])
            vol = data['close'].diff().abs().iloc[-5:].sum()
            curr_eff = net_c / vol if vol > 0 else 1
            
            # كلما زاد السكور، أصبح البوت أكثر صبراً (عتبة خروج أقل)
            # سكور 1.0 -> عتبة 0.4 | سكور 2.0 -> عتبة 0.2
            exit_threshold = 0.4 / t['quality_score'] 
            
            if curr_eff < exit_threshold:
                self.notify(f"⚠️ خروج سلوكي: ضعف الجودة (Score: {round(t['quality_score'], 1)})")
                self.close_trade(t)
                continue

            if current_price <= t['stop']:
                if current_price < t['entry']:
                    self.block_until = time.time() + (4 * 3600)
                    self.notify("🛑 حظر 4 ساعات.")
                self.close_trade(t)

    def close_trade(self, trade):
        if trade in self.trades: self.trades.remove(trade)
        if not self.trades: self.trade_state = TradeState.IDLE

# =========================================================
# ③ محرك التشغيل المستقل
# =========================================================
def run_trading_engine(bot):
    bot.notify("🧠 نظام 'رصيد الثقة التراكمي' مفعل وجاهز.")
    while True:
        try:
            ohlcv = bot.exchange.fetch_ohlcv(bot.symbol, timeframe='15m', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'vol'])
            last_price = df['close'].iloc[-1]
            
            state, reason = bot.analyze_market(df)
            bot.manage_logic(last_price, df)

            if state == MarketState.TRENDING and len(bot.trades) < 2:
                if time.time() > bot.block_until:
                    can_enter = True
                    # شرط الصفقة الثانية: سياق جديد + فارق سعري + سكور عالي للأولى
                    if len(bot.trades) > 0:
                        first_trade = bot.trades[0]
                        price_diff = abs(last_price - first_trade['entry']) / first_trade['entry']
                        # لا يسمح بالثانية إلا إذا كان سكور الأولى ارتفع (أثبتت جودتها)
                        if price_diff < 0.015 or first_trade['quality_score'] < 1.3:
                            can_enter = False 
                    
                    if can_enter:
                        bot.execute_order("buy", last_price, df['low'].iloc[-5:].min(), "دخول سياقي معزز بجودة عالية")
            
        except Exception as e: print(f"⚠️ خطأ: {e}")
        time.sleep(60)

# (واجهة تليجرام المعتادة)
def telegram_listener(bot_instance):
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            r = requests.get(url, params={"timeout": 10, "offset": offset}).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                text = u.get("message", {}).get("text", "")
                if text == "/start": bot_instance.notify("🤖 نظام التوازن الأقصى مفعل.")
                elif text == "/status":
                    status = "⏳ مُعلق" if time.time() < bot_instance.block_until else "✅ جاهز"
                    score_msg = ""
                    if bot_instance.trades:
                        score_msg = f"\n🎯 سكور الجودة: {round(bot_instance.trades[0]['quality_score'], 1)}"
                    bot_instance.notify(f"📊 الوضع: {bot_instance.mode}\n🛡️ الحالة: {status}\n📦 الصفقات: {len(bot_instance.trades)}{score_msg}")
                elif text == "/dry": bot_instance.mode = Mode.DRY; bot_instance.notify("🧪 وضع DRY")
                elif text == "/live": bot_instance.mode = Mode.LIVE; bot_instance.notify("⚠️ وضع LIVE")
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    my_bot = BehavioralTradingBot()
    threading.Thread(target=telegram_listener, args=(my_bot,), daemon=True).start()
    run_trading_engine(my_bot)
