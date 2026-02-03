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

# ==================== CONFIGURATION (UTC BASED) ====================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
INITIAL_CAPITAL = 1000
MAX_CAPITAL_PER_TRADE = 0.1
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.04
MAX_DAILY_LOSS = 0.05
MAX_TOTAL_LOSS = 0.2

OPTIMAL_HOURS = list(range(8, 22))
AVOID_HOURS = [0, 1, 2, 3, 4, 5]

BREAKEVEN_TRIGGER = 0.012
TRAILING_ACTIVATION = 0.03
TRAILING_DISTANCE = 0.01

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ==================== LOGGING SYSTEM (UTC) ====================
class UTCFormatter(logging.Formatter):
    converter = lambda *args: datetime.now(timezone.utc).timetuple()

def setup_logging():
    log_formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
    main_handler = RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=5)
    main_handler.setFormatter(log_formatter)
    main_handler.setLevel(logging.INFO)
    error_handler = RotatingFileHandler('errors.log', maxBytes=2*1024*1024, backupCount=3)
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)
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

class MarketState(Enum):
    TRENDING = "TRENDING"
    SPECULATIVE = "SPECULATIVE"
    BALANCED = "BALANCED"

# ==================== LIQUIDITY TIMER ====================
class LiquidityTimer:
    @staticmethod
    def is_optimal_time():
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        current_weekday = now_utc.weekday()
        if current_weekday >= 5:
            return False, "نهاية الأسبوع - سيولة منخفضة"
        if current_hour in AVOID_HOURS:
            return False, f"ساعة {current_hour} UTC - سيولة منخفضة"
        return True, f"ساعة {current_hour} UTC"

    @staticmethod
    def get_sleep_duration():
        current_hour = datetime.now(timezone.utc).hour
        return 180 if current_hour in OPTIMAL_HOURS else 300

# ==================== CAPITAL MANAGER ====================
class CapitalManager:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.daily_loss_limit = initial_capital * MAX_DAILY_LOSS
        self.total_loss_limit = initial_capital * MAX_TOTAL_LOSS
        self.daily_loss = 0
        self.total_loss = 0
        self.daily_profit = 0
        self.total_profit = 0
        self.trade_history = []
        self.last_reset_date = datetime.now(timezone.utc).date()
        
    def reset_daily_stats(self):
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset_date:
            self.daily_loss = 0
            self.daily_profit = 0
            self.last_reset_date = today
    
    def can_open_trade(self, symbol, planned_investment):
        self.reset_daily_stats()
        if self.daily_loss >= self.daily_loss_limit: return False, "تجاوز الحد اليومي"
        if self.total_loss >= self.total_loss_limit: return False, "تجاوز الحد الكلي"
        if planned_investment > self.available_capital: return False, "رأس مال غير كافي"
        return True, "موافق"

    def update_after_trade(self, trade_result, investment, profit_loss):
        if trade_result == TradeStatus.WIN:
            self.current_capital += profit_loss
            self.available_capital += investment + profit_loss
            self.daily_profit += profit_loss
        elif trade_result == TradeStatus.LOSS:
            self.current_capital -= profit_loss
            self.available_capital += (investment - profit_loss)
            self.daily_loss += profit_loss
        logger.info(f"Capital Update: {self.current_capital}")

    def notify(self, message):
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
            except: pass

# ==================== THREE-PHASE TRADE MANAGER ====================
class ThreePhaseTradeManager:
    def __init__(self, capital_manager):
        self.capital_manager = capital_manager
        self.active_trades = {}
        self.trade_counter = 0

    def create_trade(self, symbol, entry_price, quantity, market_score, market_state):
        trade_id = f"TRADE_{self.trade_counter:04d}"
        self.trade_counter += 1
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
        trade = {
            'id': trade_id, 'symbol': symbol, 'entry_price': entry_price,
            'current_stop_loss': stop_loss, 'take_profit': entry_price * (1 + TAKE_PROFIT_PERCENT),
            'quantity': quantity, 'investment': quantity * entry_price,
            'status': TradeStatus.ACTIVE, 'phase': TradePhase.PHASE_1_ENTRY,
            'entry_time': datetime.now(timezone.utc), 'highest_price': entry_price,
            'breakeven_price': entry_price * (1 + BREAKEVEN_TRIGGER),
            'trailing_activation_price': entry_price * (1 + TRAILING_ACTIVATION),
            'trailing_active': False
        }
        self.active_trades[trade_id] = trade
        self.capital_manager.notify(f"🚀 صفقـة جـديدة: {symbol} @ {entry_price}")
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
            self.execute_exit(trade_id, current_price, TradeStatus.LOSS if current_price < trade['entry_price'] else TradeStatus.WIN, "SL/Trailing")
        elif not trade['trailing_active'] and current_price >= trade['take_profit']:
            self.execute_exit(trade_id, current_price, TradeStatus.WIN, "TP")

    def execute_exit(self, trade_id, exit_price, exit_status, reason):
        trade = self.active_trades.pop(trade_id)
        pnl = (exit_price - trade['entry_price']) * trade['quantity']
        self.capital_manager.update_after_trade(exit_status, trade['investment'], abs(pnl))
        self.capital_manager.notify(f"🏁 خروج: {trade['symbol']} | ربح: {pnl:.2f} | سبب: {reason}")

# ==================== STABLE TRADING SYSTEM ====================
class StableTradingSystem:
    def __init__(self):
        self.capital_manager = CapitalManager(INITIAL_CAPITAL)
        self.trade_manager = ThreePhaseTradeManager(self.capital_manager)
        self.liquidity_timer = LiquidityTimer()
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.markets_loaded = False
        self.load_markets_async()

    def load_markets_async(self):
        def _load():
            try:
                self.exchange.load_markets()
                self.markets_loaded = True
            except Exception as e: logger.error(f"Market Load Error: {e}")
        threading.Thread(target=_load, daemon=True).start()

    def run_trading_cycle(self):
        try:
            can_trade, _ = self.liquidity_timer.is_optimal_time()
            if not can_trade: return
            current_prices = {}
            for symbol in SYMBOLS:
                try: ticker = self.exchange.fetch_ticker(symbol); current_prices[symbol] = ticker['last']
                except: pass
            for t_id in list(self.trade_manager.active_trades.keys()):
                price = current_prices.get(self.trade_manager.active_trades[t_id]['symbol'])
                if price:
                    self.trade_manager.manage_trade_phase(t_id, price)
                    self.trade_manager.check_exit_conditions(t_id, price)
            self.entry_scanning(current_prices)
        except Exception as e: logger.error(f"Cycle Error: {e}")

    def entry_scanning(self, current_prices):
        if len(self.trade_manager.active_trades) < 3:
            for symbol, price in current_prices.items():
                if not any(t['symbol'] == symbol for t in self.trade_manager.active_trades.values()):
                    pos_size = self.capital_manager.current_capital * MAX_CAPITAL_PER_TRADE
                    can, _ = self.capital_manager.can_open_trade(symbol, pos_size)
                    if can:
                        self.trade_manager.create_trade(symbol, price, pos_size/price, 50, MarketState.TRENDING)
                        break

# ==================== EXECUTION AND TELEGRAM INTERFACE ====================
if __name__ == "__main__":
    bot_system = StableTradingSystem()

    class TelegramInterface:
        def __init__(self, system):
            self.system = system
            self.offset = 0

        def start_polling(self):
            logger.info("Telegram Interface Started (Daemon Mode)")
            while True:
                try:
                    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
                    resp = requests.get(url, params={"offset": self.offset, "timeout": 20}, timeout=25).json()
                    for update in resp.get("result", []):
                        self.offset = update["update_id"] + 1
                        if "message" in update:
                            self.handle_command(update["message"]["chat"]["id"], update["message"].get("text", ""))
                except: time.sleep(10)

        def handle_command(self, chat_id, text):
            cmd = text.split()[0] if text else ""
            if cmd == "/status":
                msg = f"💰 المحفظة: ${self.system.capital_manager.current_capital:.2f}\n"
                msg += f"📊 الصفقات: {len(self.system.trade_manager.active_trades)}/3\n"
                msg += f"🌐 الأسواق: {'متصل' if self.system.markets_loaded else 'جاري التحميل'}"
                self.send(chat_id, msg)
            elif cmd == "/explain":
                msg = "📖 نظام المراحل الثلاث:\n1- دخول (SL 2%)\n2- تأمين (BE عند 1.2%)\n3- تتبع (Trailing عند 3%)"
                self.send(chat_id, msg)
            elif cmd == "/dry":
                self.send(chat_id, "🧪 وضع التداول: التجريبي (Dry Run)\nالأموال حقيقية؟: لا")
            elif cmd == "/live":
                self.send(chat_id, "⚠️ الوضع الحقيقي (Live): غير مفعل.\nيتطلب مفاتيح API وتغيير كود التنفيذ.")

        def send(self, chat_id, text):
            try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=5)
            except: pass

    # تشغيل واجهة التلغرام في خيط منفصل
    if TELEGRAM_TOKEN and CHAT_ID:
        tg_ui = TelegramInterface(bot_system)
        threading.Thread(target=tg_ui.start_polling, daemon=True).start()

    # الحلقة الرئيسية لنظام التداول
    while True:
        try:
            bot_system.run_trading_cycle()
            time.sleep(bot_system.liquidity_timer.get_sleep_duration())
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Main Loop Failure: {e}")
            time.sleep(60)
