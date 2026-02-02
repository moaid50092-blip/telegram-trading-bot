import os
import time
import ccxt
import pandas as pd
import requests
import threading # المكتبة الضرورية للتشغيل المتوازي
from enum import Enum

# =========================================================
# ① الإعدادات والربط (استخدم Secrets في Replit)
# =========================================================
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class Mode(Enum):
    DRY, LIVE = "DRY", "LIVE"

class MarketState(Enum):
    TRENDING, BALANCED, CHOPPY = "TRENDING", "BALANCED", "CHOPPY"

class TradeState(Enum):
    IDLE, IN_TRADE = "IDLE", "IN_TRADE"

# =========================================================
# ② العقل الاحترافي المتكامل
# =========================================================
class BehavioralTradingBot:
    def __init__(self, symbol="BTC/USDT", balance=1000, risk_pct=0.01, mode=Mode.DRY):
        self.symbol = symbol
        self.balance = balance
        self.risk_pct = risk_pct
        self.mode = mode
        self.trade_state = TradeState.IDLE
        self.trades = []  # دعم التعزيز الذكي

        self.exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def notify(self, message):
        print(f"📡 {message}")
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": CHAT_ID, "text": f"🤖 {message}"})
        except: pass

    def analyze_market(self, data):
        if len(data) < 20: return MarketState.CHOPPY, False
        last = data.iloc[-1]
        prev = data.iloc[-5]
        
        # كشف التريند و الرفض
        is_trending = last['high'] > prev['high'] and last['low'] > prev['low']
        lower_wick = min(last['open'], last['close']) - last['low']
        total_range = last['high'] - last['low']
        rejection = lower_wick > (total_range * 0.5) if total_range > 0 else False
        
        volatility = data['close'].pct_change().std()
        if volatility > 0.015: return MarketState.CHOPPY, rejection
        
        state = MarketState.TRENDING if is_trending else MarketState.BALANCED
        return state, rejection

    def execute_order(self, side, price, stop, reason):
        risk_usd = self.balance * self.risk_pct
        dist = abs(price - stop)
        if dist == 0: return
        
        size = risk_usd / dist
        size_prec = self.exchange.amount_to_precision(self.symbol, size)
        
        msg = f"🔔 {reason} | السعر {price} | الستوب {stop}"
        self.notify(msg)

        if self.mode == Mode.LIVE:
            try:
                order = self.exchange.create_order(self.symbol, 'market', side, size_prec)
                self.notify(f"✅ تم التنفيذ الحي! ID: {order['id']}")
            except Exception as e:
                self.notify(f"❌ خطأ تنفيذ: {e}")
                return

        self.trades.append({"entry": price, "stop": stop, "size": size_prec, "trailing": price})
        self.trade_state = TradeState.IN_TRADE

    def manage_logic(self, current_price):
        """إدارة الوقف المتحرك والتأمين"""
        for t in self.trades[:]:
            # تأمين الصفقة عند ربح 1.5%
            if current_price > t['entry'] * 1.015:
                if t['stop'] < t['entry']:
                    t['stop'] = t['entry']
                    self.notify("🛡️ تم تحريك الستوب لنقطة الدخول (تأمين).")
            
            # الخروج
            if current_price <= t['stop']:
                self.notify(f"🛑 خروج بربح/خسارة عند {current_price}")
                self.trades.remove(t)
        
        if not self.trades: self.trade_state = TradeState.IDLE

# =========================================================
# ③ نظام الاستماع للأوامر (تليجرام)
# =========================================================
def telegram_listener(bot_instance):
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            response = requests.get(url, params={"timeout": 10, "offset": offset}).json()

            for update in response.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message", {})
                text = message.get("text", "")

                if text == "/start":
                    bot_instance.notify("👋 أهلاً بك! النظام يعمل الآن.\nاستخدم القائمة للتحكم.")
                elif text == "/dry":
                    bot_instance.mode = Mode.DRY
                    bot_instance.notify("🧪 تم التحويل لوضع DRY (تجريبي).")
                elif text == "/live":
                    bot_instance.mode = Mode.LIVE
                    bot_instance.notify("⚠️ تنبيه: تم تفعيل التداول الحقيقي LIVE!")
                elif text == "/status":
                    msg = f"📊 الحالة: {bot_instance.mode}\n💰 الرصيد: {bot_instance.balance}\n📦 الصفقات المفتوحة: {len(bot_instance.trades)}"
                    bot_instance.notify(msg)

        except Exception as e:
            print(f"خطأ في مستمع تليجرام: {e}")
        time.sleep(1)

# =========================================================
# ④ حلقة التشغيل الأساسية (محرك التداول)
# =========================================================
def run_trading_engine(bot):
    bot.notify("🚀 تم تشغيل النظام بنجاح.. بانتظار الفرصة الأولى.")
    
    while True:
        try:
            ohlcv = bot.exchange.fetch_ohlcv(bot.symbol, timeframe='15m', limit=50)
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
            last_price = df['close'].iloc[-1]
            state, has_rejection = bot.analyze_market(df)

            # 1. إدارة الصفقات المفتوحة
            bot.manage_logic(last_price)

            # 2. منطق الدخول والتعزيز
            if state != MarketState.CHOPPY and has_rejection:
                if bot.trade_state == TradeState.IDLE:
                    bot.execute_order("buy", last_price, df['low'].iloc[-1], "دخول رئيسي")
                elif len(bot.trades) < 2 and last_price > bot.trades[0]['entry'] * 1.02:
                    bot.execute_order("buy", last_price, bot.trades[0]['entry'], "تعزيز ذكي")
            
        except Exception as e:
            print(f"⚠️ خطأ في محرك التداول: {e}")
        
        time.sleep(60)

# =========================================================
# ⑤ نقطة الانطلاق الرسمية (Execution)
# =========================================================
if __name__ == "__main__":
    # 1. تهيئة نسخة البوت
    my_bot = BehavioralTradingBot() 

    # 2. تشغيل "مستمع تليجرام" في مسار مستقل (Background Thread)
    listener_thread = threading.Thread(target=telegram_listener, args=(my_bot,))
    listener_thread.daemon = True # لضمان إغلاق المسار عند توقف البرنامج
    listener_thread.start()

    # 3. تشغيل محرك التداول الأساسي في المسار الرئيسي
    run_trading_engine(my_bot)
