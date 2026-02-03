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
# إعداد نظام التسجيل بملفات منفصلة وتوقيت UTC
class UTCFormatter(logging.Formatter):
    converter = lambda *args: datetime.now(timezone.utc).timetuple()

def setup_logging():
    log_formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # ملف السجلات العام
    main_handler = RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=5)
    main_handler.setFormatter(log_formatter)
    main_handler.setLevel(logging.INFO)
    
    # ملف الأخطاء فقط
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

# ==================== LIQUIDITY TIMER (UTC) ====================
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
        if current_hour in OPTIMAL_HOURS:
            return True, f"ساعة {current_hour} UTC - سيولة ممتازة"
        return True, f"ساعة {current_hour} UTC - سيولة مقبولة"

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
            logger.info(f"إعادة تعيين الإحصائيات اليومية ليوم جديد: {today}")
            self.daily_loss = 0
            self.daily_profit = 0
            self.last_reset_date = today
    
    def can_open_trade(self, symbol, planned_investment):
        self.reset_daily_stats()
        if self.daily_loss >= self.daily_loss_limit:
            return False, "تم تجاوز الحد اليومي للخسارة"
        if self.total_loss >= self.total_loss_limit:
            return False, "تم تجاوز الحد الكلي للخسارة"
        if planned_investment > self.available_capital:
            return False, "رأس مال غير كافي"
        return True, "يمكن فتح الصفقة"

    def update_after_trade(self, trade_result, investment, profit_loss):
        # التحديثات الحسابية (تتم داخلياً لغايات الـ Paper Trading)
        if trade_result == TradeStatus.WIN:
            self.current_capital += profit_loss
            self.available_capital += investment + profit_loss
            self.daily_profit += profit_loss
        elif trade_result == TradeStatus.LOSS:
            self.current_capital -= profit_loss
            self.available_capital += (investment - profit_loss)
            self.daily_loss += profit_loss
        
        logger.info(f"تحديث المحفظة: النتيجة {trade_result.value}, PnL: {profit_loss}")

    def notify(self, message):
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
                requests.post(url, json=payload, timeout=10)
            except Exception as e:
                logger.error(f"فشل إرسال إشعار تلغرام: {e}")

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
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT)
        
        trade = {
            'id': trade_id, 'symbol': symbol, 'entry_price': entry_price,
            'current_stop_loss': stop_loss, 'take_profit': take_profit,
            'quantity': quantity, 'investment': quantity * entry_price,
            'status': TradeStatus.ACTIVE, 'phase': TradePhase.PHASE_1_ENTRY,
            'entry_time': datetime.now(timezone.utc), 'highest_price': entry_price,
            'breakeven_price': entry_price * (1 + BREAKEVEN_TRIGGER),
            'trailing_activation_price': entry_price * (1 + TRAILING_ACTIVATION),
            'trailing_active': False, 'breakeven_applied': False
        }
        
        self.active_trades[trade_id] = trade
        logger.info(f"تم إنشاء صفقة وهمية: {trade_id} لـ {symbol}")
        self.capital_manager.notify(f"🎯 صفقة جديدة #{trade_id}\nالعملة: {symbol}\nالسعر: {entry_price}")
        return trade_id

    def manage_trade_phase(self, trade_id, current_price):
        trade = self.active_trades.get(trade_id)
        if not trade: return
        
        if current_price > trade['highest_price']:
            trade['highest_price'] = current_price

        # المرحلة 1 -> 2
        if trade['phase'] == TradePhase.PHASE_1_ENTRY and current_price >= trade['breakeven_price']:
            trade['current_stop_loss'] = trade['entry_price']
            trade['phase'] = TradePhase.PHASE_2_BREAKEVEN
            logger.info(f"تأمين الصفقة {trade_id} عند سعر الدخول")
            
        # المرحلة 2 -> 3
        if trade['phase'] == TradePhase.PHASE_2_BREAKEVEN and current_price >= trade['trailing_activation_price']:
            trade['trailing_active'] = True
            trade['phase'] = TradePhase.PHASE_3_TRAILING
            logger.info(f"تفعيل التتبع (Trailing) للصفقة {trade_id}")

        if trade['trailing_active']:
            new_stop = trade['highest_price'] * (1 - TRAILING_DISTANCE)
            if new_stop > trade['current_stop_loss']:
                trade['current_stop_loss'] = new_stop

    def check_exit_conditions(self, trade_id, current_price):
        trade = self.active_trades.get(trade_id)
        if not trade: return
        
        reason = None
        if current_price <= trade['current_stop_loss']:
            reason = "Stop Loss / Trailing"
            status = TradeStatus.LOSS if current_price < trade['entry_price'] else TradeStatus.WIN
        elif not trade['trailing_active'] and current_price >= trade['take_profit']:
            reason = "Take Profit"
            status = TradeStatus.WIN
            
        if reason:
            self.execute_exit(trade_id, current_price, status, reason)

    def execute_exit(self, trade_id, exit_price, exit_status, reason):
        trade = self.active_trades.pop(trade_id)
        pnl = (exit_price - trade['entry_price']) * trade['quantity']
        self.capital_manager.update_after_trade(exit_status, trade['investment'], abs(pnl))
        
        msg = f"✅ خروج #{trade_id} ({reason})\nالعملة: {trade['symbol']}\nPnL: ${pnl:.2f}"
        self.capital_manager.notify(msg)
        logger.info(f"إغلاق صفقة {trade_id}: {reason}, PnL: {pnl}")

# ==================== STABLE TRADING SYSTEM (CORE) ====================
class StableTradingSystem:
    def __init__(self):
        self.capital_manager = CapitalManager(INITIAL_CAPITAL)
        self.trade_manager = ThreePhaseTradeManager(self.capital_manager)
        self.liquidity_timer = LiquidityTimer()
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.markets_loaded = False
        logger.info("تم بدء تشغيل النظام المستقر...")
        self.load_markets_async()

    def load_markets_async(self):
        def _load():
            try:
                self.exchange.load_markets()
                self.markets_loaded = True
                logger.info("تم تحميل الأسواق من Binance بنجاح.")
            except Exception as e:
                logger.error(f"فشل تحميل الأسواق: {e}")
        threading.Thread(target=_load, daemon=True).start()

    def run_trading_cycle(self):
        try:
            can_trade, reason = self.liquidity_timer.is_optimal_time()
            if not can_trade:
                logger.info(f"توقف مؤقت: {reason}")
                return

            current_prices = {}
            for symbol in SYMBOLS:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_prices[symbol] = ticker['last']
                except Exception as e:
                    logger.warning(f"خطأ في جلب سعر {symbol}: {e}")
            
            # إدارة الصفقات المفتوحة
            for t_id in list(self.trade_manager.active_trades.keys()):
                price = current_prices.get(self.trade_manager.active_trades[t_id]['symbol'])
                if price:
                    self.trade_manager.manage_trade_phase(t_id, price)
                    self.trade_manager.check_exit_conditions(t_id, price)

            # منطق الدخول (المبسط للمحاكاة)
            self.entry_scanning(current_prices)

        except Exception as e:
            logger.error(f"خطأ غير متوقع في دورة التداول: {e}", exc_info=True)

    def entry_scanning(self, current_prices):
        # هنا يتم وضع منطق الـ Analyzer الخاص بك (تم الإبقاء على الهيكل العام)
        if len(self.trade_manager.active_trades) < 3:
            for symbol, price in current_prices.items():
                # محاكاة شرط دخول بسيط لغرض العمل المستقر
                active_for_symbol = any(t['symbol'] == symbol for t in self.trade_manager.active_trades.values())
                if not active_for_symbol:
                    # فحص الأمان المالي
                    pos_size = self.capital_manager.current_capital * MAX_CAPITAL_PER_TRADE
                    can, msg = self.capital_manager.can_open_trade(symbol, pos_size)
                    if can:
                        # في كودك الأصلي هنا يتم استدعاء Analyzer، هنا سنفترض تحقق الشروط تقنياً
                        quantity = pos_size / price
                        self.trade_manager.create_trade(symbol, price, quantity, 50.0, MarketState.TRENDING)
                        break

# ==================== MAIN EXECUTION ====================
def main():
    logger.info("=== بدء تشغيل البوت (وضع المحاكاة المستقر) ===")
    system = StableTradingSystem()
    
    # تشغيل بوت تلغرام في خيط منفصل (Thread)
    # (ملاحظة: تحتاج لإضافة كلاس TelegramBot الأصلي هنا إذا أردت الرد على الأوامر)
    
    while True:
        try:
            system.run_trading_cycle()
            sleep_time = system.liquidity_timer.get_sleep_duration()
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("تم إيقاف البوت يدوياً.")
            break
        except Exception as e:
            logger.error(f"خطأ في الحلقة الرئيسية: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
