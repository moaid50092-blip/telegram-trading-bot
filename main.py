#!/usr/bin/env python3
"""
StableBotPro v4.0 - نظام التداول الآلي المتكامل
مزود بطبقة تنفيذ كاملة ونظام مراقبة يومي
"""

import os
import sys
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
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
from enum import Enum
import random
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import statistics

# ==================== تهيئة النظام ====================
load_dotenv()

# ==================== الأنواع الأساسية ====================
class TradingMode(Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"

# ==================== إعدادات التداول الثابتة ====================
class TradingConfig:
    # === القائمة الموسعة للأصول (حتى 50 عملة) ===
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
        "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT",
        "XLM/USDT", "BCH/USDT", "ALGO/USDT", "VET/USDT", "FIL/USDT",
        "TRX/USDT", "XTZ/USDT", "THETA/USDT", "EOS/USDT", "AAVE/USDT",
        "SNX/USDT", "MKR/USDT", "COMP/USDT", "YFI/USDT", "SUSHI/USDT",
        "CRV/USDT", "1INCH/USDT", "REN/USDT", "BAT/USDT", "ZRX/USDT",
        "OMG/USDT", "ENJ/USDT", "STORJ/USDT", "SAND/USDT", "MANA/USDT",
        "GALA/USDT", "AXS/USDT", "CHZ/USDT", "FTM/USDT", "NEAR/USDT",
        "GRT/USDT", "ANKR/USDT", "ICP/USDT", "FLOW/USDT", "RUNE/USDT"
    ]
    
    # === إدارة رأس المال ===
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1000))
    MAX_CAPITAL_PER_TRADE = 0.10  # 10% من رأس المال للصفقة
    MAX_OPEN_TRADES = 3  # أقصى عدد للصفقات المفتوحة
    
    # === وقف الخسارة وجني الأرباح ===
    STOP_LOSS_PERCENT = 0.02  # 2%
    TAKE_PROFIT_PERCENT = 0.04  # 4%
    MAX_DAILY_LOSS = 0.05  # 5% من رأس المال اليومي
    MAX_TOTAL_DRAWDOWN = 0.20  # 20% أقصى خسارة كلية من رأس المال الأولي
    
    # === نظام المراحل ===
    BREAKEVEN_TRIGGER = 0.012  # 1.2%
    TRAILING_ACTIVATION = 0.03  # 3%
    TRAILING_DISTANCE = 0.01  # 1%
    
    # === توقيت السوق ===
    OPTIMAL_HOURS = list(range(8, 22))  # 8 صباحاً - 10 مساءً UTC
    AVOID_HOURS = [0, 1, 2, 3, 4, 5]  # 12 صباحاً - 5 صباحاً UTC
    
    # === التواصل ===
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # === إعدادات الأداء ===
    MIN_SCORE = 45  # الحد الأدنى للدخول
    SCAN_INTERVAL = 180  # ثواني بين الدورات
    API_DELAY = 0.3  # تأخير بين طلبات API
    
    # === إعدادات الفلاتر ===
    ENABLE_MARKET_FILTER = True
    MARKET_ATR_THRESHOLD = 0.008  # أقل من 0.8% يعتبر سايدوايز
    MARKET_EMA_SLOPE_THRESHOLD = 0.0005  # ميل EMA 50
    MIN_ATR_PERCENT = 0.005  # الحد الأدنى للتقلب
    
    # === فلتر الارتباط ===
    ENABLE_CORRELATION_FILTER = True
    CORRELATION_GROUPS = {
        "MAJOR": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "LARGE_CAP": ["SOL/USDT", "ADA/USDT", "XRP/USDT", "DOT/USDT"],
        "MID_CAP": ["DOGE/USDT", "AVAX/USDT", "MATIC/USDT", "LTC/USDT"],
        "SMALL_CAP": ["LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT"]
    }
    
    # === إدارة المخاطرة الديناميكية ===
    ENABLE_DYNAMIC_RISK = True
    RISK_LEVELS = {
        "HIGH": {"min_score": 70, "sl_multiplier": 0.8, "trailing_distance": 0.008},
        "MEDIUM": {"min_score": 55, "sl_multiplier": 1.0, "trailing_distance": 0.01},
        "LOW": {"min_score": 45, "sl_multiplier": 1.2, "trailing_distance": 0.012}
    }
    
    # === إدارة الوضع ===
    DEFAULT_MODE = TradingMode.PAPER
    ALLOW_MODE_SWITCH = True
    MODE_SWITCH_PASSWORD = os.getenv('MODE_SWITCH_PASSWORD', '')
    
    # === تحقق API ===
    @classmethod
    def validate_api_keys(cls, mode: TradingMode) -> Tuple[bool, str]:
        if mode == TradingMode.PAPER:
            return True, "OK"
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            return False, "مفاتيح API غير موجودة"
        
        if len(api_key) < 20 or len(api_secret) < 20:
            return False, "مفاتيح API غير صالحة"
        
        return True, "OK"

# ==================== هياكل البيانات ====================
@dataclass
class TradeRecord:
    """سجل الصفقة"""
    trade_id: str
    symbol: str
    entry_price: float
    entry_time: str
    quantity: float
    stop_loss: float
    take_profit: Optional[float] = None
    phase: str = "ENTRY"
    status: str = "ACTIVE"
    highest_price: float = 0.0
    score: Optional[float] = None
    risk_level: Optional[str] = None
    original_take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    execution_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    stop_loss_modified: bool = False
    take_profit_modified: bool = False

@dataclass
class MarketAnalysis:
    """تحليل السوق"""
    symbol: str
    score: float = 0.0
    price: float = 0.0
    atr_percent: float = 0.0
    ema_slope: float = 0.0
    is_sideways: bool = False
    correlation_group: Optional[str] = None
    ranking: int = 0
    last_ohlcv: Optional[pd.DataFrame] = None

@dataclass
class OrderResult:
    """نتيجة تنفيذ الأمر"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str
    amount: float
    price: float
    filled: float
    remaining: float
    status: OrderStatus
    average_price: Optional[float] = None
    cost: Optional[float] = None
    fee: Optional[float] = None
    fee_currency: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    retry_count: int = 0
    is_simulated: bool = False

@dataclass
class TradeExecution:
    """تنفيذ الصفقة"""
    trade_id: str
    symbol: str
    entry_order: OrderResult
    exit_order: Optional[OrderResult] = None
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    net_price_entry: float = 0.0
    net_price_exit: Optional[float] = None
    total_fees: float = 0.0
    net_pnl: float = 0.0

# ==================== نظام السجلات ====================
class Logger:
    @staticmethod
    def setup(name: str = "StableBot"):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        file_handler = RotatingFileHandler(
            'logs/trading.log',
            maxBytes=10*1024*1024,
            backupCount=10,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

# ==================== نظام إدارة الوضع ====================
class ModeManager:
    def __init__(self, bot):
        self.bot = bot
        self.current_mode = TradingConfig.DEFAULT_MODE
        self.mode_lock = threading.Lock()
        self.mode_change_time = None
        self.require_password = bool(TradingConfig.MODE_SWITCH_PASSWORD)
        
    def switch_mode(self, new_mode: TradingMode, password: str = None) -> Tuple[bool, str]:
        with self.mode_lock:
            # التحقق من الصلاحية
            if new_mode == self.current_mode:
                return False, f"البوت بالفعل في وضع {new_mode.value}"
            
            # التحقق من كلمة المرور إذا مطلوبة
            if self.require_password and new_mode == TradingMode.LIVE:
                if not password or password != TradingConfig.MODE_SWITCH_PASSWORD:
                    return False, "كلمة المرور غير صحيحة"
            
            # التحقق من API keys للوضع LIVE
            if new_mode == TradingMode.LIVE:
                is_valid, msg = TradingConfig.validate_api_keys(new_mode)
                if not is_valid:
                    return False, f"فشل التحقق: {msg}"
            
            # تنفيذ التبديل
            old_mode = self.current_mode
            self.current_mode = new_mode
            self.mode_change_time = datetime.now(timezone.utc)
            self.bot.trading_mode = new_mode
            self.bot.exchange_interface.mode = new_mode
            
            # في حالة التحويل إلى LIVE، نقوم بمزامنة الرصيد
            if new_mode == TradingMode.LIVE:
                success, msg = self.bot._sync_live_balance()
                if not success:
                    return False, f"فشل التبديل: {msg}"
            
            # تسجيل الحدث
            self.bot.logger.warning(f"🔀 تبديل الوضع: {old_mode.value} → {new_mode.value}")
            
            return True, f"تم التبديل إلى وضع {new_mode.value}"
    
    def get_mode_info(self) -> Dict:
        return {
            "current_mode": self.current_mode.value,
            "mode_change_time": self.mode_change_time.isoformat() if self.mode_change_time else None,
            "is_live": self.current_mode == TradingMode.LIVE,
            "require_password": self.require_password
        }

# ==================== معالج أوامر Telegram ====================
class TelegramCommandHandler:
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger('TelegramHandler')
        self.commands = {
            '/start': self.handle_start,
            '/help': self.handle_help,
            '/status': self.handle_status,
            '/mode': self.handle_mode,
            '/stats': self.handle_stats,
            '/trades': self.handle_trades,
            '/stop': self.handle_stop,
            '/restart': self.handle_restart,
            '/balance': self.handle_balance,
            '/drawdown': self.handle_drawdown
        }
    
    def handle_command(self, command: str, args: List[str] = None) -> str:
        if not command in self.commands:
            return "❌ أمر غير معروف. استخدم /help للقائمة."
        
        try:
            return self.commands[command](args)
        except Exception as e:
            self.logger.error(f"خطأ في معالجة الأمر {command}: {e}")
            return f"❌ خطأ في معالجة الأمر: {str(e)}"
    
    def handle_start(self, args=None) -> str:
        return """🤖 *StableBot Pro v4.0*
        
الأوامر المتاحة:
• `/status` - حالة البوت
• `/mode paper` - وضع المحاكاة
• `/mode live [password]` - وضع التداول الحي
• `/stats` - إحصائيات اليوم
• `/trades` - الصفقات النشطة
• `/balance` - الرصيد الحقيقي
• `/drawdown` - نسبة الخسارة الكلية
• `/stop` - إيقاف البوت (بعد تأكيد)
• `/restart` - إعادة تشغيل البوت
• `/help` - هذه القائمة

⚙️ الوضع الحالي: """ + self.bot.mode_manager.get_mode_info()['current_mode']
    
    def handle_help(self, args=None) -> str:
        return self.handle_start()
    
    def handle_status(self, args=None) -> str:
        info = self.bot.mode_manager.get_mode_info()
        active_trades = len(self.bot.active_trades)
        
        return f"""📊 *حالة البوت*
• الوضع: `{info['current_mode']}`
• الصفقات النشطة: `{active_trades}`
• رأس المال: `${self.bot.current_capital:.2f}`
• الرصيد المتاح: `${self.bot.available_capital:.2f}`
• P&L اليوم: `${self.bot.daily_pnl:.2f}`
• الخسارة الكلية: `${self.bot.total_drawdown_percent:.1f}%`
• آخر تحديث: `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"""
    
    def handle_mode(self, args) -> str:
        if not args or len(args) < 1:
            return "❌ استخدم: `/mode paper` أو `/mode live [password]`"
        
        target_mode = args[0].upper()
        password = args[1] if len(args) > 1 else None
        
        if target_mode not in ['PAPER', 'LIVE']:
            return "❌ الوضع يجب أن يكون: paper أو live"
        
        try:
            new_mode = TradingMode(target_mode)
            
            if new_mode == TradingMode.LIVE:
                if not self.bot.mode_manager.require_password and not password:
                    return "⚠️ تحويل إلى LIVE يتطلب تأكيدًا. أرسل: `/mode live CONFIRM`"
            
            success, message = self.bot.mode_manager.switch_mode(new_mode, password)
            
            if success:
                # إرسال تحذير إذا كان LIVE
                if new_mode == TradingMode.LIVE:
                    warning_msg = "🚨 *تحذير مهم:*\n"
                    warning_msg += "• البوت الآن في وضع التداول الحي\n"
                    warning_msg += "• سيقوم بتنفيذ صفقات حقيقية\n"
                    warning_msg += "• الأوامر ستخصم من رصيدك الحقيقي\n"
                    warning_msg += "• تم مزامنة الرصيد من المنصة\n"
                    warning_msg += "• تأكد من متابعة البوت باستمرار\n"
                    
                    self.bot._send_notification(warning_msg)
                    return f"✅ {message}\n\n{warning_msg}"
                else:
                    return f"✅ {message}"
            else:
                return f"❌ {message}"
                
        except Exception as e:
            return f"❌ خطأ في تغيير الوضع: {str(e)}"
    
    def handle_stats(self, args=None) -> str:
        return self.bot.monitor.generate_daily_report()
    
    def handle_trades(self, args=None) -> str:
        if not self.bot.active_trades:
            return "📭 لا توجد صفقات نشطة حالياً"
        
        response = "📈 *الصفقات النشطة:*\n"
        for trade_id, trade in self.bot.active_trades.items():
            response += f"\n• `{trade.symbol}`\n"
            response += f"  الدخول: `${trade.entry_price:.4f}`\n"
            response += f"  الكمية: `{trade.quantity:.6f}`\n"
            response += f"  Stop: `${trade.stop_loss:.4f}`"
            if trade.stop_loss_order_id:
                response += f" (منصة)"
            response += f"\n"
            if trade.take_profit:
                response += f"  Take Profit: `${trade.take_profit:.4f}`"
                if trade.take_profit_order_id:
                    response += f" (منصة)"
                response += f"\n"
            response += f"  المرحلة: `{trade.phase}`\n"
        
        return response
    
    def handle_balance(self, args=None) -> str:
        if self.bot.trading_mode == TradingMode.PAPER:
            return f"📊 *رصيد المحاكاة*\n• المتاح: `${self.bot.available_capital:.2f}`\n• الإجمالي: `${self.bot.current_capital:.2f}`"
        else:
            try:
                balance = self.bot.exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {})
                free = usdt_balance.get('free', 0)
                total = usdt_balance.get('total', 0)
                
                return f"🏦 *رصيد المنصة الحقيقي*\n• المتاح: `${free:.2f}`\n• الإجمالي: `${total:.2f}`\n• محجوز: `${total - free:.2f}`"
            except Exception as e:
                return f"❌ خطأ في جلب الرصيد: {str(e)}"
    
    def handle_drawdown(self, args=None) -> str:
        max_drawdown = TradingConfig.MAX_TOTAL_DRAWDOWN * 100
        current_drawdown = self.bot.total_drawdown_percent
        status = "🟢" if current_drawdown < max_drawdown * 0.8 else "🟡" if current_drawdown < max_drawdown else "🔴"
        
        return f"""📉 *الخسارة الكلية*
• الحالية: `{current_drawdown:.1f}%`
• الحد الأقصى: `{max_drawdown:.1f}%`
• الحالة: {status}"""
    
    def handle_stop(self, args=None) -> str:
        return "⏸️ لإيقاف البوت، استخدم Ctrl+C في واجهة التشغيل"
    
    def handle_restart(self, args=None) -> str:
        return "🔄 لإعادة التشغيل، أوقف البوت وأعده تشغيل يدوياً"

# ==================== أنظمة الفلاتر الأساسية ====================
class MarketFilter:
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def analyze_market_regime(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        if len(df) < 100:
            return True, {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            atr = self._calculate_atr(high, low, close, 14)
            current_atr = atr[-1] if atr[-1] > 0 else 0
            atr_percent = current_atr / close[-1] if close[-1] > 0 else 0
            
            ema_50 = self._calculate_ema(close, 50)
            ema_slope = self._calculate_slope(ema_50[-20:])
            
            is_sideways = False
            if self.config.ENABLE_MARKET_FILTER:
                is_sideways = (atr_percent < self.config.MARKET_ATR_THRESHOLD and 
                              abs(ema_slope) < self.config.MARKET_EMA_SLOPE_THRESHOLD)
            
            analysis = {
                "atr_percent": atr_percent,
                "ema_slope": ema_slope,
                "is_sideways": is_sideways,
                "is_tradable": not is_sideways
            }
            
            return not is_sideways, analysis
            
        except Exception as e:
            return True, {}
    
    def _calculate_atr(self, high, low, close, period):
        try:
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.values
        except:
            return np.zeros(len(high))
    
    def _calculate_ema(self, data, period):
        try:
            return pd.Series(data).ewm(span=period, adjust=False).mean().values
        except:
            return np.zeros(len(data))
    
    def _calculate_slope(self, data):
        if len(data) < 2:
            return 0
        try:
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data, 1)
            return slope
        except:
            return 0

class CorrelationFilter:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.group_mapping = self._create_group_mapping()
    
    def _create_group_mapping(self) -> Dict[str, str]:
        mapping = {}
        for group_name, symbols in self.config.CORRELATION_GROUPS.items():
            for symbol in symbols:
                mapping[symbol] = group_name
        return mapping
    
    def get_symbol_group(self, symbol: str) -> Optional[str]:
        return self.group_mapping.get(symbol)
    
    def can_trade_symbol(self, symbol: str, active_trades: Dict[str, TradeRecord]) -> Tuple[bool, str]:
        if not self.config.ENABLE_CORRELATION_FILTER:
            return True, "Correlation filter disabled"
        
        symbol_group = self.get_symbol_group(symbol)
        if not symbol_group:
            return True, "Symbol not in any correlation group"
        
        for trade in active_trades.values():
            trade_group = self.get_symbol_group(trade.symbol)
            if trade_group == symbol_group:
                return False, f"Already trading in {symbol_group} group"
        
        return True, "OK"

class DynamicRiskManager:
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def get_risk_parameters(self, score: float) -> Dict:
        if not self.config.ENABLE_DYNAMIC_RISK:
            return {
                "level": "STANDARD",
                "sl_multiplier": 1.0,
                "trailing_distance": self.config.TRAILING_DISTANCE,
                "can_trade": score >= self.config.MIN_SCORE
            }
        
        if score >= self.config.RISK_LEVELS["HIGH"]["min_score"]:
            level = "HIGH"
        elif score >= self.config.RISK_LEVELS["MEDIUM"]["min_score"]:
            level = "MEDIUM"
        elif score >= self.config.RISK_LEVELS["LOW"]["min_score"]:
            level = "LOW"
        else:
            return {
                "level": "REJECTED",
                "sl_multiplier": 1.0,
                "trailing_distance": self.config.TRAILING_DISTANCE,
                "can_trade": False
            }
        
        params = self.config.RISK_LEVELS[level]
        return {
            "level": level,
            "sl_multiplier": params["sl_multiplier"],
            "trailing_distance": params["trailing_distance"],
            "can_trade": True
        }
    
    def calculate_stop_loss(self, entry_price: float, risk_params: Dict) -> float:
        sl_multiplier = risk_params.get("sl_multiplier", 1.0)
        sl_distance = self.config.STOP_LOSS_PERCENT * sl_multiplier
        return entry_price * (1 - sl_distance)
    
    def calculate_trailing_distance(self, risk_params: Dict) -> float:
        return risk_params.get("trailing_distance", self.config.TRAILING_DISTANCE)

# ==================== فلتر Bear Market Safety ====================
class BearMarketFilter:
    """
    فلتر أمان فقط - لا يولد إشارات ولا يغير استراتيجية
    فقط يمنع أو يقلل التداول في ظروف السوق الهابطة
    """
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = True
        self.logger = logging.getLogger('BearMarketFilter')
        
        # إعدادات الفلتر
        self.BTC_SYMBOL = "BTC/USDT"
        self.MIN_BTC_TREND = -0.02
        self.MAX_DRAWDOWN = -0.15
        
        # حالات الفلتر
        self.last_btc_price = None
        self.market_condition = "NORMAL"
    
    def analyze_market_condition(self, exchange) -> Dict:
        """
        تحليل حالة السوق بدون التأثير على استراتيجية التداول
        """
        if not self.enabled:
            return {"can_trade": True, "condition": "NORMAL", "reason": "Filter disabled"}
        
        try:
            # 1. تحليل BTC (مؤشر رئيسي)
            btc_ticker = exchange.fetch_ticker(self.BTC_SYMBOL)
            current_btc = btc_ticker['last']
            
            btc_ohlcv = exchange.fetch_ohlcv(self.BTC_SYMBOL, '1d', limit=30)
            if len(btc_ohlcv) >= 7:
                btc_df = pd.DataFrame(btc_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                
                # حساب تغيير 7 أيام
                btc_7d_change = (current_btc - btc_df['close'].iloc[-7]) / btc_df['close'].iloc[-7]
                
                # حساب المتوسط المتحرك 20 يوم
                ma_20 = btc_df['close'].rolling(20).mean().iloc[-1]
                below_ma_20 = current_btc < ma_20
                
                # 2. تحليل السوق العام
                bearish_count = 0
                total_check = min(10, len(self.config.SYMBOLS))
                
                for symbol in self.config.SYMBOLS[:total_check]:
                    try:
                        symbol_ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=20)
                        if len(symbol_ohlcv) >= 20:
                            symbol_df = pd.DataFrame(symbol_ohlcv, 
                                                   columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                            
                            symbol_ma20 = symbol_df['close'].rolling(20).mean().iloc[-1]
                            if symbol_df['close'].iloc[-1] < symbol_ma20:
                                bearish_count += 1
                        
                        time.sleep(0.1)
                    except:
                        continue
                
                bearish_ratio = bearish_count / total_check if total_check > 0 else 0
                
                # 3. تحديد حالة السوق
                condition = "NORMAL"
                reasons = []
                
                if btc_7d_change < self.MIN_BTC_TREND:
                    condition = "CAUTION"
                    reasons.append(f"BTC هبط {btc_7d_change*100:.1f}% في 7 أيام")
                
                if below_ma_20:
                    condition = "CAUTION" if condition == "NORMAL" else "BEARISH"
                    reasons.append("BTC تحت المتوسط المتحرك 20 يوم")
                
                if bearish_ratio > 0.7:
                    condition = "BEARISH"
                    reasons.append(f"{bearish_ratio*100:.0f}% من العملات في ترند هبوطي")
                
                # 4. قرار التداول
                can_trade = True
                if condition == "BEARISH":
                    can_trade = False
                    reasons.append("توقفت جميع الصفقات الجديدة")
                elif condition == "CAUTION":
                    can_trade = True
                    reasons.append("يسمح بصفقة واحدة فقط في نفس الوقت")
                
                return {
                    "can_trade": can_trade,
                    "condition": condition,
                    "reasons": reasons,
                    "btc_7d_change": btc_7d_change,
                    "below_ma_20": below_ma_20,
                    "bearish_ratio": bearish_ratio,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            else:
                return {"can_trade": True, "condition": "NORMAL", "reason": "لا بيانات كافية"}
                
        except Exception as e:
            self.logger.error(f"خطأ في BearMarketFilter: {e}")
            return {"can_trade": True, "condition": "NORMAL", "reason": f"خطأ: {str(e)}"}
    
    def apply_filter(self, ranked_symbols: List[Tuple[str, MarketAnalysis]], 
                    market_condition: Dict) -> List[Tuple[str, MarketAnalysis]]:
        """
        تطبيق الفلتر على القائمة المرتبة
        """
        if not self.enabled or market_condition.get("condition") == "NORMAL":
            return ranked_symbols
        
        condition = market_condition.get("condition", "NORMAL")
        
        if condition == "BEARISH":
            self.logger.warning("⛔ Bear Market Filter: منع جميع الصفقات الجديدة")
            return []
        
        elif condition == "CAUTION":
            if ranked_symbols:
                self.logger.warning("⚠️ Bear Market Filter: السماح بصفقة واحدة فقط")
                return ranked_symbols[:1]
        
        return ranked_symbols

# ==================== فلتر العملات الشرعية ====================
class EthicalFilter:
    """
    فلتر العملات الشرعية - استبعاد العملات ذات الشبهات الواضحة
    بدون تغيير آلية الاختيار أو الاستراتيجية
    """
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = True
        self.logger = logging.getLogger('EthicalFilter')
        
        # القائمة السوداء
        self.BLACKLIST = [
            "DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "PEPE/USDT", "BONK/USDT",
            "FUN/USDT", "CHP/USDT", "BET/USDT", "TRX/USDT", "WIN/USDT",
            "LUNC/USDT", "USTC/USDT", "XMR/USDT", "ZEC/USDT", "DASH/USDT"
        ]
        
        # القائمة البيضاء
        self.WHITELIST_MODE = False
        self.WHITELIST = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
            "XRP/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT", "AVAX/USDT",
            "LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT", "XLM/USDT",
            "BCH/USDT", "ALGO/USDT", "VET/USDT", "FIL/USDT", "XTZ/USDT",
            "EOS/USDT", "AAVE/USDT", "SNX/USDT", "MKR/USDT", "COMP/USDT",
            "YFI/USDT", "SUSHI/USDT", "CRV/USDT", "1INCH/USDT", "REN/USDT"
        ]
    
    def is_symbol_allowed(self, symbol: str) -> Tuple[bool, str]:
        """
        التحقق إذا كانت العملة مسموحة
        """
        if not self.enabled:
            return True, "Filter disabled"
        
        # التحقق من Blacklist
        if symbol in self.BLACKLIST:
            return False, f"العملة في القائمة السوداء"
        
        # إذا كان Whitelist mode مفعلاً
        if self.WHITELIST_MODE and symbol not in self.WHITELIST:
            return False, f"العملة غير موجودة في القائمة البيضاء"
        
        # كلمات مفتاحية
        symbol_lower = symbol.lower()
        gambling_keywords = ['bet', 'casino', 'poker', 'gamble', 'lottery', 'dice']
        if any(keyword in symbol_lower for keyword in gambling_keywords):
            return False, "تحتوي على كلمات مقامرة"
        
        meme_keywords = ['dog', 'shib', 'floki', 'pepe', 'bonk', 'elon', 'moon']
        if any(keyword in symbol_lower for keyword in meme_keywords):
            return False, "مشروع ميم كوين عالي المخاطرة"
        
        return True, "OK"
    
    def filter_symbols(self, ranked_symbols: List[Tuple[str, MarketAnalysis]]) -> List[Tuple[str, MarketAnalysis]]:
        """
        تطبيق الفلتر على القائمة المرتبة
        """
        if not self.enabled:
            return ranked_symbols
        
        filtered = []
        removed_count = 0
        
        for symbol, analysis in ranked_symbols:
            is_allowed, reason = self.is_symbol_allowed(symbol)
            
            if is_allowed:
                filtered.append((symbol, analysis))
            else:
                removed_count += 1
                self.logger.info(f"⛔ EthicalFilter: استبعاد {symbol} - {reason}")
        
        if removed_count > 0:
            self.logger.info(f"✅ EthicalFilter: بقي {len(filtered)} عملة من أصل {len(ranked_symbols)}")
        
        return filtered
    
    def get_filter_stats(self) -> Dict:
        """
        إحصائيات الفلتر
        """
        return {
            "enabled": self.enabled,
            "blacklist_count": len(self.BLACKLIST),
            "whitelist_count": len(self.WHITELIST) if self.WHITELIST_MODE else 0,
            "whitelist_mode": self.WHITELIST_MODE
        }

# ==================== طبقة التنفيذ المحسنة ====================
class ExecutionConfig:
    MAKER_FEE = 0.001
    TAKER_FEE = 0.001
    SLIPPAGE_PERCENT = 0.001
    MAX_SLIPPAGE = 0.005
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    DEFAULT_ORDER_TYPE = OrderType.MARKET
    USE_POST_ONLY = False
    TIMEOUT_SECONDS = 30
    ALLOW_PARTIAL_FILLS = True
    MIN_FILL_PERCENT = 0.8
    ENABLE_EXCHANGE_ORDERS = True  # تفعيل
