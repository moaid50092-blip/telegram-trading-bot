import os
import time
import ccxt
import pandas as pd
import requests
import threading
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from enum import Enum
import warnings
from dotenv import load_dotenv

# تحميل المتغيرات البيئية
load_dotenv()
warnings.filterwarnings('ignore')

# ==================== 1️⃣ TRADING MODE CONFIGURATION ====================
TRADING_MODE = "DRY"  # غير القيمة إلى "LIVE" للتداول الحقيقي

# ==================== CONFIGURATION ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
INITIAL_CAPITAL = 1000
MAX_CAPITAL_PER_TRADE = 0.1
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.04
MAX_DAILY_LOSS = 0.05

BREAKEVEN_TRIGGER = 0.012
TRAILING_ACTIVATION = 0.03
TRAILING_DISTANCE = 0.01

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BINANCE_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET_KEY')

# ==================== LOGGING SYSTEM ====================
class UTCFormatter(logging.Formatter):
    converter = lambda *args: datetime.now(timezone.utc).timetuple()

def setup_logging():
    log_formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
    main_handler = RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=5)
    main_handler.setFormatter(log_formatter)
    error_handler = RotatingFileHandler('errors.log', maxBytes=2*1024*1024, backupCount=3)
    error_handler.setFormatter(log_formatter)
    logger = logging.getLogger('StableBot')
    logger.setLevel(logging.INFO)
    logger.addHandler(main_handler)
    logger.addHandler(error_handler)
    return logger

logger = setup_logging()

# ==================== ENUMS ====================
class TradeStatus(Enum):
    ACTIVE = "ACTIVE"
    WIN = "WIN"
    LOSS = "LOSS"

class TradePhase(Enum):
    PHASE_1_ENTRY = "PHASE_1_ENTRY"
    PHASE_2_BREAKEVEN = "PHASE_2_BREAKEVEN"
    PHASE_3_TRAILING = "PHASE_3_TRAILING"

# ==================== CAPITAL MANAGER ====================
class CapitalManager:
    def __init__(self, initial_capital):
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.daily_loss_limit = initial_capital * MAX_DAILY_LOSS
        self.daily_loss = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        
    def reset_daily_stats(self):
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset_date:
            self.daily_loss = 0
            self.last_reset_date = today
    
    def can_open_trade(self, symbol, planned_investment):
        self.reset_daily_stats()
        if self.daily_loss >= self.daily_loss_limit: return False, "تجاوز الحد اليومي"
        if planned_investment > self.available_capital: return False, "رأس مال غير كافي"
        return True, "موافق"

    def update_after_trade(self, trade_result, investment, profit_loss):
        if trade_result == TradeStatus.WIN:
            self.current_capital += profit_loss
            self.available_capital += investment + profit_loss
        elif trade_result == TradeStatus.LOSS:
            self.current_capital -= profit_loss
            self.available_capital += (investment - profit_loss)
            self.daily_loss += profit_loss
        logger.info(f"Capital Update: ${self.current_capital:.2f}")

    def notify(self, message):
        mode_tag = f"🔴 [LIVE]" if TRADING_MODE == "LIVE" else "🧪 [DRY]"
        full_msg = f"{mode_tag}\n{message}"
        if TELEGRAM_TOKEN and CHAT_ID:
            try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                               json={"chat_id": CHAT_ID, "text": full_msg, "parse_mode": "Markdown"}, timeout=10)
            except: pass

# ==================== THREE-PHASE TRADE MANAGER ====================
class ThreePhaseTradeManager:
    def __init__(self, capital_manager, exchange):
        self.capital_manager = capital_manager
        self.exchange = exchange
        self.active_trades = {}
        self.trade_counter = 0

    def create_trade(self, symbol, entry_price, quantity, context_score):
        # تنفيذ شراء حقيقي في وضع LIVE
        if TRADING_MODE == "LIVE":
            try:
                if not BINANCE_KEY or not BINANCE_SECRET:
                    raise ValueError("مفاتيح API مفقودة!")
                order = self.exchange.create_market_buy_order(symbol, quantity)
                entry_price = order.get('price', entry_price)
                logger.info(f"LIVE BUY EXECUTED: {symbol} at {entry_price}")
            except Exception as e:
                logger.error(f"LIVE BUY FAILED: {e}")
                return None

        trade_id = f"TRADE_{self.trade_counter:04d}"
        self.trade_counter += 1
        trade = {
            'id': trade_id, 'symbol': symbol, 'entry_price': entry_price,
            'current_stop_loss': entry_price * (1 - STOP_LOSS_PERCENT),
            'take_profit': entry_price * (1 + TAKE_PROFIT_PERCENT),
            'quantity': quantity, 'investment': quantity * entry_price,
            'phase': TradePhase.PHASE_1_ENTRY, 'highest_price': entry_price,
            'breakeven_price': entry_price * (1 + BREAKEVEN_TRIGGER),
            'trailing_activation_price': entry_price * (1 + TRAILING_ACTIVATION),
            'trailing_active': False
        }
        self.active_trades[trade_id] = trade
        self.capital_manager.notify(f"🚀 دخول صفقة: {symbol}\nالسعر: {entry_price:.4f}\nقوة الفرصة: {context_score}")
        return trade_id

    def manage_trade_phase(self, trade_id, current_price):
        trade = self.active_trades.get(trade_id)
        if not trade: return
        if current_price > trade['highest_price']: trade['highest_price'] = current_price
        if trade['phase'] == TradePhase.PHASE_1_ENTRY and current_price >= trade['breakeven_price']:
            trade['current_stop_loss'] = trade['entry_price']
            trade['phase'] = TradePhase.PHASE_2_BREAKEVEN
        if trade['phase'] == TradePhase.PHASE_2_BREAKEVEN and current_price >= trade['trailing_activation_price']:
            trade['trailing_active'] = True
            trade['phase'] = TradePhase.PHASE_3_TRAILING
        if trade['trailing_active']:
            new_stop = trade['highest_price'] * (1 - TRAILING_DISTANCE)
            if new_stop > trade['current_stop_loss']: trade['current_stop_loss'] = new_stop

    def check_exit_conditions(self, trade_id, current_price):
        trade = self.active_trades.get(trade_id)
        if not trade: return
        if current_price <= trade['current_stop_loss']:
            self.execute_exit(trade_id, current_price, TradeStatus.LOSS if current_price < trade['entry_price'] else TradeStatus.WIN, "خروج آمن/تتبع")
        elif not trade['trailing_active'] and current_price >= trade['take_profit']:
            self.execute_exit(trade_id, current_price, TradeStatus.WIN, "هدف الربح")

    def execute_exit(self, trade_id, exit_price, exit_status, reason):
        trade = self.active_trades.pop(trade_id)
        # تنفيذ بيع حقيقي في وضع LIVE
        if TRADING_MODE == "LIVE":
            try:
                self.exchange.create_market_sell_order(trade['symbol'], trade['quantity'])
                logger.info(f"LIVE SELL EXECUTED: {trade['symbol']}")
            except Exception as e:
                logger.error(f"LIVE SELL FAILED: {e}")

        pnl = (exit_price - trade['entry_price']) * trade['quantity']
        self.capital_manager.update_after_trade(exit_status, trade['investment'], abs(pnl))
        self.capital_manager.notify(f"🏁 خروج: {trade['symbol']}\nالربح/الخسارة: ${pnl:.2f}\nالسبب: {reason}")

# ==================== STABLE TRADING SYSTEM ====================
class StableTradingSystem:
    def __init__(self):
        self.exchange = self._init_exchange()
        self.capital_manager = CapitalManager(INITIAL_CAPITAL)
        self.trade_manager = ThreePhaseTradeManager(self.capital_manager, self.exchange)
        self.markets_loaded = False
        self.load_markets_async()

    def _init_exchange(self):
        params = {'enableRateLimit': True}
        if TRADING_MODE == "LIVE" and BINANCE_KEY and BINANCE_SECRET:
            params.update({'apiKey': BINANCE_KEY, 'secret': BINANCE_SECRET})
        return ccxt.binance(params)

    def load_markets_async(self):
        def _load():
            try: self.exchange.load_markets(); self.markets_loaded = True
            except: pass
        threading.Thread(target=_load, daemon=True).start()

    def calculate_context_score(self, df):
        try:
            if len(df) < 30: return 0
            net_move = abs(df['close'].iloc[-1] - df['close'].iloc[-10])
            path = df['close'].diff().abs().iloc[-10:].sum()
            efficiency = (net_move / path) * 40 if path > 0 else 0
            atr_s = (df['high'] - df['low']).rolling(5).mean().iloc[-1]
            atr_l = (df['high'] - df['low']).rolling(20).mean().iloc[-1]
            vol_ratio = min((atr_s / atr_l), 1.5) * 20 if atr_l > 0 else 0
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            dist_score = 20 if df['close'].iloc[-1] > sma20 else 5
            last = df.iloc[-1]
            candle_range = last['high'] - last['low']
            upper_wick = last['high'] - max(last['open'], last['close'])
            rejection_penalty = (upper_wick / candle_range) * 30 if candle_range > 0 else 0
            return round(max(min(efficiency + vol_ratio + dist_score - rejection_penalty, 100), 0), 2)
        except: return 0

    def run_trading_cycle(self):
        try:
            # 2️⃣ إلغاء فلتر التوقيت: النظام يعمل الآن 24/7 بناءً على التحليل فقط
            current_prices, ohlcv_data = {}, {}
            for symbol in SYMBOLS:
                try:
                    bars = self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=50)
                    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                    current_prices[symbol] = df['close'].iloc[-1]
                    ohlcv_data[symbol] = df
                except: continue
            
            for t_id in list(self.trade_manager.active_trades.keys()):
                price = current_prices.get(self.trade_manager.active_trades[t_id]['symbol'])
                if price:
                    self.trade_manager.manage_trade_phase(t_id, price)
                    self.trade_manager.check_exit_conditions(t_id, price)
            
            self.entry_scanning(current_prices, ohlcv_data)
        except Exception as e: logger.error(f"Cycle Error: {e}")

    def entry_scanning(self, current_prices, ohlcv_data):
        if len(self.trade_manager.active_trades) < 3:
            for symbol, price in current_prices.items():
                if not any(t['symbol'] == symbol for t in self.trade_manager.active_trades.values()):
                    score = self.calculate_context_score(ohlcv_data[symbol])
                    if score < 40: continue
                    pos_size = self.capital_manager.current_capital * MAX_CAPITAL_PER_TRADE
                    can, _ = self.capital_manager.can_open_trade(symbol, pos_size)
                    if can:
                        self.trade_manager.create_trade(symbol, price, pos_size/price, score)
                        break

# ==================== EXECUTION ====================
if __name__ == "__main__":
    bot_system = StableTradingSystem()

    class TelegramInterface:
        def __init__(self, system):
            self.system, self.offset = system, 0
        def start_polling(self):
            while True:
                try:
                    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
                    resp = requests.get(url, params={"offset": self.offset, "timeout": 20}, timeout=25).json()
                    for update in resp.get("result", []):
                        self.offset = update["update_id"] + 1
                        if "message" in update: self.handle_command(update["message"]["chat"]["id"], update["message"].get("text", ""))
                except: time.sleep(10)
        def handle_command(self, chat_id, text):
            if not text: return
            cmd = text.split()[0]
            if cmd == "/status":
                msg = f"💰 المحفظة: ${self.system.capital_manager.current_capital:.2f}\n📊 الصفقات: {len(self.system.trade_manager.active_trades)}/3\n⚙️ الوضع: {TRADING_MODE}"
                self.send(chat_id, msg)
            elif cmd == "/ping": self.send(chat_id, "🏓 Pong! النظام يعمل 24/7")

        def send(self, chat_id, text):
            try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": chat_id, "text": text})
            except: pass

    if TELEGRAM_TOKEN and CHAT_ID:
        threading.Thread(target=TelegramInterface(bot_system).start_polling, daemon=True).start()

    logger.info(f"Bot started in {TRADING_MODE} mode (No Time Filter).")
    while True:
        try:
            bot_system.run_trading_cycle()
            time.sleep(180) # دورة كل 3 دقائق
        except KeyboardInterrupt: break
        except Exception as e: 
            logger.error(f"Loop Error: {e}")
            time.sleep(60)
