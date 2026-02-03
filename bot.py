import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
import threading
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import traceback

# ==================== تهيئة البيئة ====================
load_dotenv()

# ==================== إعدادات التداول ====================
class TradingConfig:
    # الأصول المطلوبة
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOT/USDT"]
    
    # إدارة رأس المال
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1000))
    MAX_CAPITAL_PER_TRADE = 0.10  # 10% من رأس المال للصفقة
    MAX_OPEN_TRADES = 3
    MIN_CAPITAL_FOR_TRADE = 10  # أقل مبلغ للصفقة
    
    # وقف الخسارة وجني الأرباح
    STOP_LOSS_PERCENT = 0.02  # 2%
    TAKE_PROFIT_PERCENT = 0.04  # 4%
    
    # نظام المراحل
    BREAKEVEN_TRIGGER = 0.012  # 1.2%
    TRAILING_ACTIVATION = 0.03  # 3%
    TRAILING_DISTANCE = 0.01  # 1%
    
    # توقيت السوق
    OPTIMAL_HOURS = list(range(8, 22))  # 8 صباحاً - 10 مساءً UTC
    AVOID_HOURS = [0, 1, 2, 3, 4, 5]  # 12 صباحاً - 5 صباحاً UTC
    
    # إعدادات الأداء
    MIN_SCORE = 45
    SCAN_INTERVAL = 180  # 3 دقائق
    API_RATE_LIMIT_DELAY = 0.5  # تأخير بين طلبات API
    MAX_RETRIES = 3  # أقصى محاولات للاتصال
    
    # التواصل
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

@dataclass
class TradeRecord:
    """سجل الصفقة"""
    trade_id: str
    symbol: str
    entry_price: float
    entry_time: str
    quantity: float
    stop_loss: float
    take_profit: float
    phase: str = "ENTRY"
    status: str = "ACTIVE"
    highest_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    score: Optional[float] = None

@dataclass
class CapitalManager:
    """مدير رأس المال"""
    initial_capital: float
    current_capital: float = field(init=False)
    available_capital: float = field(init=False)
    invested_capital: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    
    def __post_init__(self):
        self.current_capital = self.initial_capital
        self.available_capital = self.initial_capital
    
    def update_capital(self, pnl: float):
        """تحديث رأس المال بعد صفقة"""
        self.current_capital += pnl
        self.available_capital += pnl
        self.daily_pnl += pnl
        self.daily_trades += 1
    
    def can_open_trade(self, required_amount: float) -> bool:
        """التحقق من إمكانية فتح صفقة جديدة"""
        return (self.available_capital >= required_amount and 
                required_amount >= TradingConfig.MIN_CAPITAL_FOR_TRADE)
    
    def get_stats(self) -> Dict:
        """الحصول على إحصائيات رأس المال"""
        return {
            "current_capital": self.current_capital,
            "available_capital": self.available_capital,
            "invested_capital": self.invested_capital,
            "total_return": ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades
        }

# ==================== نظام السجلات ====================
class Logger:
    """مدير السجلات المبسط"""
    
    @staticmethod
    def setup(name: str = "StableBot"):
        # إنشاء مجلد السجلات
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # إعداد السجل
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # ملف السجلات
        file_handler = RotatingFileHandler(
            'logs/trading.log',
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        )
        
        # شكل السجلات
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        file_handler.setFormatter(formatter)
        
        # سجل وحدة التحكم
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # إضافة المعالجات
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

# ==================== نظام التواصل ====================
class TelegramNotifier:
    """مدير إشعارات التلغرام"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_notification = {}
    
    def send_message(self, message: str, msg_type: str = "info"):
        """إرسال رسالة عبر تلغرام"""
        if not self.token or not self.chat_id:
            return False
        
        try:
            # منع التكرار
            now = time.time()
            if msg_type in self.last_notification:
                if now - self.last_notification[msg_type] < 30:  # 30 ثانية
                    return False
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.last_notification[msg_type] = now
                return True
            else:
                print(f"خطأ تلغرام: {response.text}")
                return False
                
        except Exception as e:
            print(f"خطأ في إرسال تلغرام: {e}")
            return False
    
    def format_trade_entry(self, symbol: str, price: float, quantity: float, score: float) -> str:
        """تنسيق رسالة دخول صفقة"""
        return (
            f"🚀 *دخول صفقة جديدة*\n"
            f"• العملة: `{symbol}`\n"
            f"• السعر: `${price:.4f}`\n"
            f"• الكمية: `{quantity:.4f}`\n"
            f"• القيمة: `${price * quantity:.2f}`\n"
            f"• السكور: `{score:.1f}`\n"
            f"• الوقت: `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
        )
    
    def format_trade_exit(self, symbol: str, entry: float, exit: float, quantity: float, 
                          pnl: float, reason: str) -> str:
        """تنسيق رسالة خروج صفقة"""
        pnl_percent = ((exit / entry) - 1) * 100
        status = "✅" if pnl > 0 else "❌"
        
        return (
            f"{status} *إغلاق صفقة*\n"
            f"• العملة: `{symbol}`\n"
            f"• الدخول: `${entry:.4f}`\n"
            f"• الخروج: `${exit:.4f}`\n"
            f"• الكمية: `{quantity:.4f}`\n"
            f"• P&L: `${pnl:.2f}` ({pnl_percent:.2f}%)\n"
            f"• السبب: `{reason}`\n"
            f"• الوقت: `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
        )

# ==================== محرك التداول ====================
class TradingEngine:
    """محرك التداول الأساسي"""
    
    def __init__(self, exchange):
        self.exchange = exchange
    
    def calculate_score(self, df: pd.DataFrame) -> float:
        """حساب درجة التداول"""
        try:
            if len(df) < 30:
                return 0
            
            close = df['close']
            
            # 1. كفاءة الحركة (40 نقطة)
            net_move = abs(close.iloc[-1] - close.iloc[-10])
            total_path = close.diff().abs().iloc[-10:].sum()
            efficiency = (net_move / total_path * 40) if total_path > 0 else 0
            
            # 2. اتجاه السوق (20 نقطة)
            sma_20 = close.rolling(20).mean().iloc[-1]
            trend = 20 if close.iloc[-1] > sma_20 else 5
            
            # 3. مؤشر الرفض (خصم 20 نقطة)
            last_candle = df.iloc[-1]
            candle_range = last_candle['high'] - last_candle['low']
            
            if candle_range > 0:
                upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                upper_wick_ratio = upper_wick / candle_range
                rejection_penalty = upper_wick_ratio * 30
            else:
                rejection_penalty = 0
            
            # 4. حجم التداول (10 نقاط)
            volume_avg = df['vol'].iloc[-10:].mean()
            volume_current = df['vol'].iloc[-1]
            volume_score = 10 if volume_current > volume_avg else 5
            
            # النتيجة النهائية
            score = efficiency + trend + volume_score - rejection_penalty
            return max(0, min(100, score))
            
        except Exception as e:
            print(f"خطأ في حساب السكور: {e}")
            return 0
    
    def fetch_market_data(self, symbol: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق مع إعادة المحاولة"""
        for attempt in range(retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe='15m',
                    limit=35
                )
                
                if not ohlcv or len(ohlcv) < 20:
                    time.sleep(1)
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                return df
                
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # تأخير متزايد
                    print(f"محاولة {attempt + 1} فشلت لـ {symbol}: {e}. الانتظار {wait_time} ثانية...")
                    time.sleep(wait_time)
                else:
                    print(f"فشل جلب البيانات لـ {symbol} بعد {retries} محاولات")
                    return None
        
        return None

# ==================== البوت الرئيسي ====================
class StableBotPro:
    """البوت الرئيسي للتداول"""
    
    def __init__(self):
        # التهيئة
        self.config = TradingConfig
        self.logger = Logger.setup("StableBotPro")
        
        # الأنظمة الفرعية
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.capital_manager = CapitalManager(self.config.INITIAL_CAPITAL)
        self.trading_engine = TradingEngine(self.exchange)
        self.notifier = TelegramNotifier(self.config.TELEGRAM_TOKEN, self.config.TELEGRAM_CHAT_ID)
        
        # حالة النظام
        self.active_trades: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        self.system_start_time = datetime.now(timezone.utc)
        
        # إنشاء المجلدات
        self._setup_directories()
        
        # تحميل الصفقات المحفوظة
        self._load_active_trades()
        
        self.logger.info(f"بدء StableBotPro برأس مال: ${self.config.INITIAL_CAPITAL}")
        self.notifier.send_message(
            f"🚀 *StableBotPro بدأ التشغيل*\n"
            f"• الرأس المال: `${self.config.INITIAL_CAPITAL:.2f}`\n"
            f"• الأصول: {len(self.config.SYMBOLS)}\n"
            f"• الوقت: `{self.system_start_time.strftime('%H:%M UTC')}`"
        )
    
    def _setup_directories(self):
        """إنشاء المجلدات الضرورية"""
        directories = [
            'logs',
            'data/active_trades',
            'data/closed_trades',
            'data/backups'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"تم إنشاء مجلد: {directory}")
    
    def _load_active_trades(self):
        """تحميل الصفقات النشطة من الملفات"""
        try:
            trades_dir = 'data/active_trades'
            if not os.path.exists(trades_dir):
                return
            
            loaded_count = 0
            for filename in os.listdir(trades_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(trades_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # التحقق من صحة البيانات
                        if all(key in data for key in ['trade_id', 'symbol', 'entry_price', 'quantity']):
                            trade = TradeRecord(**data)
                            self.active_trades[trade.trade_id] = trade
                            loaded_count += 1
                        else:
                            self.logger.warning(f"بيانات تالفة في: {filename}")
                            
                    except Exception as e:
                        self.logger.error(f"خطأ في تحميل {filename}: {e}")
            
            self.logger.info(f"تم تحميل {loaded_count} صفقة نشطة")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل الصفقات: {e}")
    
    def _save_trade(self, trade: TradeRecord):
        """حفظ الصفقة إلى ملف"""
        try:
            filename = f"data/active_trades/{trade.trade_id}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(asdict(trade), f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الصفقة {trade.trade_id}: {e}")
    
    def _move_to_closed(self, trade: TradeRecord):
        """نقل الصفقة إلى الأرشيف"""
        try:
            # حذف من الملفات النشطة
            active_file = f"data/active_trades/{trade.trade_id}.json"
            if os.path.exists(active_file):
                os.remove(active_file)
            
            # حفظ في الأرشيف
            closed_file = f"data/closed_trades/{trade.trade_id}.json"
            with open(closed_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(trade), f, indent=2, default=str, ensure_ascii=False)
            
            # إضافة إلى القائمة
            self.closed_trades.append(trade)
            
        except Exception as e:
            self.logger.error(f"خطأ في نقل الصفقة {trade.trade_id}: {e}")
    
    def _is_market_open(self) -> bool:
        """التحقق من كون السوق مفتوح للتداول"""
        current_hour = datetime.now(timezone.utc).hour
        
        if current_hour in self.config.AVOID_HOURS:
            return False
        
        # يمكن إضافة فحص العطلات أو أحداث السوق هنا
        return True
    
    def evaluate_trade_opportunity(self, symbol: str, df: pd.DataFrame) -> Optional[TradeRecord]:
        """تقييم فرصة التداول"""
        try:
            # حساب النتيجة
            score = self.trading_engine.calculate_score(df)
            
            if score < self.config.MIN_SCORE:
                return None
            
            # التحقق من الشروط
            current_price = df['close'].iloc[-1]
            trade_amount = self.capital_manager.current_capital * self.config.MAX_CAPITAL_PER_TRADE
            
            if not self.capital_manager.can_open_trade(trade_amount):
                self.logger.debug(f"رأس المال غير كافي لـ {symbol}")
                return None
            
            # التحقق من عدم وجود صفقة مفتوحة على نفس الأصل
            if any(trade.symbol == symbol for trade in self.active_trades.values()):
                self.logger.debug(f"صفقة مفتوحة بالفعل على {symbol}")
                return None
            
            # التحقق من عدد الصفقات المفتوحة
            if len(self.active_trades) >= self.config.MAX_OPEN_TRADES:
                self.logger.debug("وصلت إلى الحد الأقصى للصفقات المفتوحة")
                return None
            
            # حساب الكمية
            quantity = trade_amount / current_price
            
            # إنشاء سجل الصفقة
            trade = TradeRecord(
                trade_id=f"T-{int(time.time())}-{symbol.replace('/', '-')}",
                symbol=symbol,
                entry_price=current_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                quantity=quantity,
                stop_loss=current_price * (1 - self.config.STOP_LOSS_PERCENT),
                take_profit=current_price * (1 + self.config.TAKE_PROFIT_PERCENT),
                highest_price=current_price,
                score=score
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"خطأ في تقييم فرصة {symbol}: {e}")
            return None
    
    def manage_active_trades(self, market_prices: Dict[str, float]):
        """إدارة الصفقات النشطة"""
        trades_to_close = []
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol not in market_prices:
                continue
            
            current_price = market_prices[trade.symbol]
            
            # تحديث أعلى سعر
            if current_price > trade.highest_price:
                trade.highest_price = current_price
            
            # إدارة المراحل
            self._update_trade_phase(trade, current_price)
            
            # التحقق من شروط الخروج
            exit_reason = self._check_exit_conditions(trade, current_price)
            
            if exit_reason:
                # إعداد بيانات الخروج
                trade.exit_price = current_price
                trade.exit_time = datetime.now(timezone.utc).isoformat()
                trade.exit_reason = exit_reason
                trade.pnl = (current_price - trade.entry_price) * trade.quantity
                trade.status = "CLOSED"
                
                trades_to_close.append(trade)
            
            else:
                # تحديث الصفقة على القرص
                self._save_trade(trade)
        
        # إغلاق الصفقات المؤهلة
        for trade in trades_to_close:
            self.close_trade(trade)
    
    def _update_trade_phase(self, trade: TradeRecord, current_price: float):
        """تحديث مرحلة الصفقة"""
        # من ENTRY إلى BREAKEVEN
        if (trade.phase == "ENTRY" and 
            current_price >= trade.entry_price * (1 + self.config.BREAKEVEN_TRIGGER)):
            trade.phase = "BREAKEVEN"
            trade.stop_loss = trade.entry_price
            self.logger.info(f"{trade.trade_id} انتقل إلى نقطة التعادل")
        
        # من BREAKEVEN إلى TRAILING
        elif (trade.phase == "BREAKEVEN" and 
              current_price >= trade.entry_price * (1 + self.config.TRAILING_ACTIVATION)):
            trade.phase = "TRAILING"
            self.logger.info(f"{trade.trade_id} تفعيل التوقف المتحرك")
        
        # تحديث التوقف المتحرك
        if trade.phase == "TRAILING":
            new_stop = trade.highest_price * (1 - self.config.TRAILING_DISTANCE)
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
    
    def _check_exit_conditions(self, trade: TradeRecord, current_price: float) -> Optional[str]:
        """التحقق من شروط الخروج"""
        # وقف الخسارة
        if current_price <= trade.stop_loss:
            return "STOP_LOSS"
        
        # جني الأرباح (في المرحلة الأولى والثانية فقط)
        if trade.phase in ["ENTRY", "BREAKEVEN"] and current_price >= trade.take_profit:
            return "TAKE_PROFIT"
        
        # كسر التوقف المتحرك
        if trade.phase == "TRAILING" and current_price <= trade.stop_loss:
            return "TRAILING_STOP"
        
        return None
    
    def open_trade(self, trade: TradeRecord):
        """فتح صفقة جديدة"""
        try:
            # تحديث رأس المال
            trade_value = trade.entry_price * trade.quantity
            self.capital_manager.available_capital -= trade_value
            self.capital_manager.invested_capital += trade_value
            
            # حفظ الصفقة
            self.active_trades[trade.trade_id] = trade
            self._save_trade(trade)
            
            # إرسال إشعار
            message = self.notifier.format_trade_entry(
                trade.symbol, trade.entry_price, trade.quantity, trade.score
            )
            self.notifier.send_message(message, "trade_entry")
            
            self.logger.info(f"✅ فتحت صفقة {trade.trade_id} على {trade.symbol}")
            
        except Exception as e:
            self.logger.error(f"خطأ في فتح الصفقة: {e}")
    
    def close_trade(self, trade: TradeRecord):
        """إغلاق صفقة"""
        try:
            # تحديث رأس المال
            self.capital_manager.update_capital(trade.pnl)
            self.capital_manager.invested_capital -= (trade.entry_price * trade.quantity)
            
            # إرسال إشعار
            message = self.notifier.format_trade_exit(
                trade.symbol, trade.entry_price, trade.exit_price,
                trade.quantity, trade.pnl, trade.exit_reason
            )
            self.notifier.send_message(message, "trade_exit")
            
            # تسجيل النتيجة
            status_emoji = "✅" if trade.pnl > 0 else "❌"
            self.logger.info(f"{status_emoji} أغلقت صفقة {trade.trade_id}: ${trade.pnl:.2f}")
            
            # نقل إلى الأرشيف
            self._move_to_closed(trade)
            
            # إزالة من الصفقات النشطة
            if trade.trade_id in self.active_trades:
                del self.active_trades[trade.trade_id]
            
        except Exception as e:
            self.logger.error(f"خطأ في إغلاق الصفقة {trade.trade_id}: {e}")
    
    def run_trading_cycle(self):
        """تشغيل دورة التداول"""
        try:
            # التحقق من وقت السوق
            if not self._is_market_open():
                if not self.active_trades:
                    self.logger.info("⏸️ وقت سوق غير مناسب - انتظار...")
                    return
            
            # جلب بيانات السوق
            market_prices = {}
            for symbol in self.config.SYMBOLS:
                try:
                    df = self.trading_engine.fetch_market_data(symbol)
                    if df is not None and not df.empty:
                        current_price = df['close'].iloc[-1]
                        market_prices[symbol] = current_price
                        
                        # البحث عن فرص جديدة
                        if len(self.active_trades) < self.config.MAX_OPEN_TRADES:
                            trade_opportunity = self.evaluate_trade_opportunity(symbol, df)
                            if trade_opportunity:
                                self.open_trade(trade_opportunity)
                        
                        # احترام حدود API
                        time.sleep(self.config.API_RATE_LIMIT_DELAY)
                        
                except Exception as e:
                    self.logger.error(f"خطأ في معالجة {symbol}: {e}")
                    continue
            
            # إدارة الصفقات النشطة
            if self.active_trades and market_prices:
                self.manage_active_trades(market_prices)
            
            # تسجيل حالة النظام (كل 10 دورات)
            if hasattr(self, '_cycle_count'):
                self._cycle_count += 1
                if self._cycle_count % 10 == 0:
                    self._log_system_status()
            else:
                self._cycle_count = 1
            
        except Exception as e:
            self.logger.error(f"خطأ في دورة التداول: {e}")
            self.notifier.send_message(f"🚨 *خطأ في دورة التداول*\n{str(e)[:200]}", "error")
    
    def _log_system_status(self):
        """تسجيل حالة النظام"""
        stats = self.capital_manager.get_stats()
        status = (
            f"📊 حالة النظام:\n"
            f"• الرأس المال: ${stats['current_capital']:.2f}\n"
            f"• الصفقات النشطة: {len(self.active_trades)}\n"
            f"• P&L اليوم: ${stats['daily_pnl']:.2f}\n"
            f"• إجمالي العائد: {stats['total_return']:.2f}%"
        )
        self.logger.info(status)

# ==================== واجهة التحكم بالتلغرام ====================
class TelegramControl:
    """واجهة التحكم بالتلغرام"""
    
    def __init__(self, bot: StableBotPro):
        self.bot = bot
        self.token = TradingConfig.TELEGRAM_TOKEN
        self.chat_id = TradingConfig.TELEGRAM_CHAT_ID
        self.commands = {
            "/status": "عرض حالة النظام",
            "/trades": "عرض الصفقات النشطة",
            "/capital": "عرض رأس المال",
            "/pause": "إيقاف البوت مؤقتاً",
            "/resume": "استئناف البوت",
            "/help": "عرض الأوامر المتاحة"
        }
        self.is_paused = False
    
    def start_listening(self):
        """بدء الاستماع للأوامر"""
        if not self.token or not self.chat_id:
            print("⚠️ إعدادات التلغرام غير مكتملة - تعطيل واجهة التحكم")
            return
        
        print("🚀 بدء واجهة تحكم التلغرام...")
        
        offset = 0
        while True:
            try:
                # التحقق من الإيقاف المؤقت
                if self.is_paused:
                    time.sleep(5)
                    continue
                
                # جلب التحديثات
                url = f"https://api.telegram.org/bot{self.token}/getUpdates"
                params = {"offset": offset, "timeout": 20}
                
                response = requests.get(url, params=params, timeout=25)
                if response.status_code == 200:
                    updates = response.json().get("result", [])
                    
                    for update in updates:
                        offset = update["update_id"] + 1
                        
                        if "message" in update and "text" in update["message"]:
                            self.handle_command(
                                update["message"]["chat"]["id"],
                                update["message"]["text"]
                            )
                
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"📡 خطأ في اتصال التلغرام: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"⚠️ خطأ غير متوقع في التلغرام: {e}")
                time.sleep(5)
    
    def handle_command(self, chat_id: int, command: str):
        """معالجة الأوامر"""
        command = command.strip().lower()
        
        if command == "/status":
            self.send_status(chat_id)
        
        elif command == "/trades":
            self.send_active_trades(chat_id)
        
        elif command == "/capital":
            self.send_capital_info(chat_id)
        
        elif command == "/pause":
            self.is_paused = True
            self.send_message(chat_id, "⏸️ تم إيقاف البوت مؤقتاً")
        
        elif command == "/resume":
            self.is_paused = False
            self.send_message(chat_id, "▶️ تم استئناف البوت")
        
        elif command == "/help":
            self.send_help(chat_id)
        
        elif command.startswith("/"):
            self.send_message(chat_id, "⚠️ أمر غير معروف. استخدم /help لعرض الأوامر المتاحة.")
    
    def send_status(self, chat_id: int):
        """إرسال حالة النظام"""
        stats = self.bot.capital_manager.get_stats()
        uptime = datetime.now(timezone.utc) - self.bot.system_start_time
        
        message = (
            f"📊 *حالة النظام*\n"
            f"• وقت التشغيل: `{uptime.total_seconds() / 3600:.1f} ساعة`\n"
            f"• الرأس المال: `${stats['current_capital']:.2f}`\n"
            f"• المتاح: `${stats['available_capital']:.2f}`\n"
            f"• الصفقات النشطة: `{len(self.bot.active_trades)}`\n"
            f"• P&L اليوم: `${stats['daily_pnl']:.2f}`\n"
            f"• إجمالي العائد: `{stats['total_return']:.2f}%`"
        )
        
        self.send_message(chat_id, message)
    
    def send_active_trades(self, chat_id: int):
        """إرسال الصفقات النشطة"""
        if not self.bot.active_trades:
            self.send_message(chat_id, "📭 لا توجد صفقات نشطة حالياً.")
            return
        
        message = "📋 *الصفقات النشطة:*\n\n"
        for trade in self.bot.active_trades.values():
            # محاولة الحصول على السعر الحالي
            try:
                ticker = self.bot.exchange.fetch_ticker(trade.symbol)
                current_price = ticker['last']
                pnl = (current_price - trade.entry_price) * trade.quantity
                pnl_percent = ((current_price / trade.entry_price) - 1) * 100
            except:
                current_price = trade.entry_price
                pnl = 0
                pnl_percent = 0
            
            message += (
                f"• `{trade.symbol}`\n"
                f"  الدخول: `${trade.entry_price:.4f}`\n"
                f"  الحالي: `${current_price:.4f}`\n"
                f"  P&L: `${pnl:.2f}` ({pnl_percent:.2f}%)\n"
                f"  المرحلة: `{trade.phase}`\n"
                f"  SL: `${trade.stop_loss:.4f}`\n\n"
            )
        
        self.send_message(chat_id, message)
    
    def send_capital_info(self, chat_id: int):
        """إرسال معلومات رأس المال"""
        stats = self.bot.capital_manager.get_stats()
        
        message = (
            f"💰 *معلومات رأس المال*\n"
            f"• الابتدائي: `${self.bot.capital_manager.initial_capital:.2f}`\n"
            f"• الحالي: `${stats['current_capital']:.2f}`\n"
            f"• المتاح: `${stats['available_capital']:.2f}`\n"
            f"• المستثمر: `${stats['invested_capital']:.2f}`\n"
            f"• إجمالي العائد: `{stats['total_return']:.2f}%`\n"
            f"• الصفقات اليوم: `{stats['daily_trades']}`"
        )
        
        self.send_message(chat_id, message)
    
    def send_help(self, chat_id: int):
        """إرسال قائمة الأوامر"""
        message = "📋 *قائمة الأوامر:*\n\n"
        for cmd, desc in self.commands.items():
            message += f"• `{cmd}` - {desc}\n"
        
        self.send_message(chat_id, message)
    
    def send_message(self, chat_id: int, text: str):
        """إرسال رسالة"""
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"فشل إرسال رسالة تلغرام: {e}")

# ==================== التشغيل الرئيسي ====================
def main():
    """الدالة الرئيسية"""
    try:
        # إنشاء البوت
        bot = StableBotPro()
        
        # بدء واجهة التحكم بالتلغرام في خيط منفصل
        if TradingConfig.TELEGRAM_TOKEN and TradingConfig.TELEGRAM_CHAT_ID:
            telegram_control = TelegramControl(bot)
            telegram_thread = threading.Thread(
                target=telegram_control.start_listening,
                daemon=True,
                name="TelegramControl"
            )
            telegram_thread.start()
            print("✅ واجهة تحكم التلغرام بدأت بنجاح")
        
        # حلقة التداول الرئيسية
        print("🚀 بدء حلقة التداول...")
        
        while True:
            try:
                # تشغيل دورة التداول
                bot.run_trading_cycle()
                
                # انتظار الفاصل الزمني
                time.sleep(TradingConfig.SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n🛑 إيقاف البوت بواسطة المستخدم...")
                
                # إرسال تقرير نهائي
                stats = bot.capital_manager.get_stats()
                final_msg = (
                    f"🛑 *تم إيقاف البوت*\n"
                    f"• الرأس المال النهائي: `${stats['current_capital']:.2f}`\n"
                    f"• إجمالي العائد: `{stats['total_return']:.2f}%`\n"
                    f"• الصفقات النشطة: `{len(bot.active_trades)}`\n"
                    f"• وقت التشغيل: `{((datetime.now(timezone.utc) - bot.system_start_time).total_seconds() / 3600):.1f} ساعة`"
                )
                bot.notifier.send_message(final_msg, "system_stop")
                break
                
            except Exception as e:
                print(f"🚨 خطأ حرج في الحلقة الرئيسية: {e}")
                bot.logger.critical(f"خطأ حرج: {e}")
                time.sleep(60)  # انتظار دقيقة قبل إعادة المحاولة
    
    except Exception as e:
        print(f"💥 فشل تشغيل النظام: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # قائمة المتطلبات
    requirements = [
        "ccxt>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("=" * 50)
    print("🚀 StableBot Pro - نظام التداول الآلي المبسط")
    print("=" * 50)
    
    # التحقق من وجود ملف .env
    if not os.path.exists('.env'):
        print("⚠️  ملف .env غير موجود!")
        print("📝 قم بإنشاء ملف .env وأضف:")
        print("INITIAL_CAPITAL=1000")
        print("TELEGRAM_TOKEN=your_token_here")
        print("TELEGRAM_CHAT_ID=your_chat_id_here")
    
    # بدء النظام
    main()
