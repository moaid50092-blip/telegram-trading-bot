#!/usr/bin/env python3
"""
StableBotPro v4.1 - نظام التداول الآلي المتكامل مع Hardening
نسخة معدلة مع إصلاحات الأمان والاستقرار
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
from collections import defaultdict, deque

load_dotenv()

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

class TradingConfig:
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT",
        "AVAX/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT",
        "XLM/USDT", "BCH/USDT", "ALGO/USDT", "VET/USDT", "FIL/USDT",
        # "TRX/USDT" تم حذفه لحل التعارض مع BLACKLIST في EthicalFilter (إصلاح حرج #5)
        "XTZ/USDT", "THETA/USDT", "EOS/USDT", "AAVE/USDT",
        "SNX/USDT", "MKR/USDT", "COMP/USDT", "YFI/USDT", "SUSHI/USDT",
        "CRV/USDT", "1INCH/USDT", "REN/USDT", "BAT/USDT", "ZRX/USDT",
        "OMG/USDT", "ENJ/USDT", "STORJ/USDT", "SAND/USDT", "MANA/USDT",
        "GALA/USDT", "AXS/USDT", "CHZ/USDT", "FTM/USDT", "NEAR/USDT",
        "GRT/USDT", "ANKR/USDT", "ICP/USDT", "FLOW/USDT", "RUNE/USDT"
    ]

    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1000))  
    MAX_CAPITAL_PER_TRADE = 0.10  
    MAX_OPEN_TRADES = 3  
    
    STOP_LOSS_PERCENT = 0.02  
    TAKE_PROFIT_PERCENT = 0.04  
    MAX_DAILY_LOSS = 0.05  
    MAX_TOTAL_DRAWDOWN = 0.20  
    
    BREAKEVEN_TRIGGER = 0.012  
    TRAILING_ACTIVATION = 0.03  
    TRAILING_DISTANCE = 0.01  
    
    OPTIMAL_HOURS = list(range(8, 22))  
    AVOID_HOURS = [0, 1, 2, 3, 4, 5]  
    
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')  
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')  
    
    MIN_SCORE = 45  
    SCAN_INTERVAL = 180  
    API_DELAY = 0.3  
    
    ENABLE_MARKET_FILTER = True  
    MARKET_ATR_THRESHOLD = 0.008  
    MARKET_EMA_SLOPE_THRESHOLD = 0.0005  
    MIN_ATR_PERCENT = 0.005  
    
    ENABLE_CORRELATION_FILTER = True  
    CORRELATION_GROUPS = {  
        "MAJOR": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],  
        "LARGE_CAP": ["SOL/USDT", "ADA/USDT", "XRP/USDT", "DOT/USDT"],  
        "MID_CAP": ["DOGE/USDT", "AVAX/USDT", "MATIC/USDT", "LTC/USDT"],  
        "SMALL_CAP": ["LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT"]  
    }  
    
    ENABLE_DYNAMIC_RISK = True  
    RISK_LEVELS = {  
        "HIGH": {"min_score": 70, "sl_multiplier": 0.8, "trailing_distance": 0.008},  
        "MEDIUM": {"min_score": 55, "sl_multiplier": 1.0, "trailing_distance": 0.01},  
        "LOW": {"min_score": 45, "sl_multiplier": 1.2, "trailing_distance": 0.012}  
    }  
    
    DEFAULT_MODE = TradingMode.PAPER  
    ALLOW_MODE_SWITCH = True  
    MODE_SWITCH_PASSWORD = os.getenv('MODE_SWITCH_PASSWORD', '')  
    
    @classmethod  
    def validate_api_keys(cls, mode: TradingMode) -> Tuple[bool, str]:  
        if mode == TradingMode.PAPER:  
            return True, "OK"  
      
        api_key = os.getenv('BINANCE_API_KEY')  
        api_secret = os.getenv('BINANCE_API_SECRET')  
      
        if not api_key or not api_secret:  
            return False, "مفاتيح API غير موجودة"  
      
        if len(api_key) < 64 or len(api_secret) < 64:  
            return False, "مفاتيح API غير صالحة"  
      
        return True, "OK"  

    API_RETRY_MAX = 5  
    API_RETRY_BACKOFF = 1  
    CIRCUIT_BREAKER_MAX_FAILS = 3  
    MIN_LIQUIDITY_USDT = 300000  
    MAX_SPREAD_PERCENT = 0.002  
    PROGRESSIVE_ENTRY_INITIAL_PERCENT = 0.4  
    PROGRESSIVE_ENTRY_BOOST_THRESHOLD = 10  
    COOLING_OFF_LOSS_STREAK = 3  
    COOLING_OFF_DURATION = 86400  
    MAX_TRADE_AGE = 259200  
    AI_CONFIDENCE_THRESHOLD = 0.3

@dataclass
class TradeRecord:
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
    entry_amount_percent: float = TradingConfig.PROGRESSIVE_ENTRY_INITIAL_PERCENT
    open_time_timestamp: float = field(default_factory=time.time)

@dataclass
class MarketAnalysis:
    symbol: str
    score: float = 0.0
    price: float = 0.0
    atr_percent: float = 0.0
    ema_slope: float = 0.0
    is_sideways: bool = False
    correlation_group: Optional[str] = None
    ranking: int = 0
    last_ohlcv: Optional[pd.DataFrame] = None
    avg_volume: float = 0.0
    spread: float = 0.0

@dataclass
class OrderResult:
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
    trade_id: str
    symbol: str
    entry_order: Optional[OrderResult] = None
    exit_order: Optional[OrderResult] = None
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    net_price_entry: float = 0.0
    net_price_exit: Optional[float] = None
    total_fees: float = 0.0
    net_pnl: float = 0.0

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

class ModeManager:
    def __init__(self, bot):
        self.bot = bot
        self.current_mode = TradingConfig.DEFAULT_MODE
        self.mode_lock = threading.Lock()
        self.mode_change_time = None
        self.require_password = bool(TradingConfig.MODE_SWITCH_PASSWORD)

    def switch_mode(self, new_mode: TradingMode, password: str = None) -> Tuple[bool, str]:  
        with self.mode_lock:  
            if new_mode == self.current_mode:  
                return False, f"البوت بالفعل في وضع {new_mode.value}"  
              
            if self.require_password and new_mode == TradingMode.LIVE:  
                if not password or password != TradingConfig.MODE_SWITCH_PASSWORD:  
                    return False, "كلمة المرور غير صحيحة"  
              
            if new_mode == TradingMode.LIVE:  
                is_valid, msg = TradingConfig.validate_api_keys(new_mode)  
                if not is_valid:  
                    return False, f"فشل التحقق: {msg}"  
              
            old_mode = self.current_mode  
            self.current_mode = new_mode  
            self.mode_change_time = datetime.now(timezone.utc)  
            self.bot.trading_mode = new_mode  
            self.bot.exchange_interface.mode = new_mode  
              
            if new_mode == TradingMode.LIVE:  
                success, msg = self.bot._sync_live_balance()  
                if not success:  
                    return False, f"فشل التبديل: {msg}"  
              
            self.bot.logger.warning(f"🔀 تبديل الوضع: {old_mode.value} → {new_mode.value}")  
              
            return True, f"تم التبديل إلى وضع {new_mode.value}"  
  
    def get_mode_info(self) -> Dict:  
        return {  
            "current_mode": self.current_mode.value,  
            "mode_change_time": self.mode_change_time.isoformat() if self.mode_change_time else None,  
            "is_live": self.current_mode == TradingMode.LIVE,  
            "require_password": self.require_password  
        }

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
        if command not in self.commands:  
            return "❌ أمر غير معروف. استخدم /help للقائمة."  
          
        try:  
            return self.commands[command](args)  
        except Exception as e:  
            self.logger.error(f"خطأ في معالجة الأمر {command}: {e}")  
            return f"❌ خطأ في معالجة الأمر: {str(e)}"  
  
    def handle_start(self, args=None) -> str:  
        return """🤖 *StableBot Pro v4.0*

الأوامر المتاحة:
• /status - حالة البوت
• /mode paper - وضع المحاكاة
• /mode live [password] - وضع التداول الحي
• /stats - إحصائيات اليوم
• /trades - الصفقات النشطة
• /balance - الرصيد الحقيقي
• /drawdown - نسبة الخسارة الكلية
• /stop - إيقاف البوت (بعد تأكيد)
• /restart - إعادة تشغيل البوت
• /help - هذه القائمة

⚙️ الوضع الحالي: """ + self.bot.mode_manager.get_mode_info()['current_mode']

    def handle_help(self, args=None) -> str:  
        return self.handle_start()  
  
    def handle_status(self, args=None) -> str:  
        info = self.bot.mode_manager.get_mode_info()  
        active_trades = len(self.bot.active_trades)  
          
        return f"""📊 *حالة البوت*

• الوضع: {info['current_mode']}
• الصفقات النشطة: {active_trades}
• رأس المال: ${self.bot.current_capital:.2f}
• الرصيد المتاح: ${self.bot.available_capital:.2f}
• P&L اليوم: ${self.bot.daily_pnl:.2f}
• الخسارة الكلية: ${self.bot.total_drawdown_percent:.1f}%
• آخر تحديث: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"""

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
                if new_mode == TradingMode.LIVE:  
                    warning_msg = "🚨 *تحذير مهم:*\\n"  
                    warning_msg += "• البوت الآن في وضع التداول الحي\\n"  
                    warning_msg += "• سيقوم بتنفيذ صفقات حقيقية\\n"  
                    warning_msg += "• الأوامر ستخصم من رصيدك الحقيقي\\n"  
                    warning_msg += "• تم مزامنة الرصيد من المنصة\\n"  
                    warning_msg += "• تأكد من متابعة البوت باستمرار\\n"  
                      
                    self.bot._send_notification(warning_msg)  
                    return f"✅ {message}\\n\\n{warning_msg}"  
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
          
        response = "📈 *الصفقات النشطة:*\\n"  
        for trade_id, trade in self.bot.active_trades.items():  
            response += f"\\n• `{trade.symbol}`\\n"  
            response += f"  الدخول: `${trade.entry_price:.4f}`\\n"  
            response += f"  الكمية: `{trade.quantity:.6f}`\\n"  
            response += f"  Stop: `${trade.stop_loss:.4f}`"  
            if trade.stop_loss_order_id:  
                response += f" (منصة)"  
            response += f"\\n"  
            if trade.take_profit:  
                response += f"  Take Profit: `${trade.take_profit:.4f}`"  
                if trade.take_profit_order_id:  
                    response += f" (منصة)"  
                response += f"\\n"  
            response += f"  المرحلة: `{trade.phase}`\\n"  
          
        return response  
  
    def handle_balance(self, args=None) -> str:  
        if self.bot.trading_mode == TradingMode.PAPER:  
            return f"📊 *رصيد المحاكاة*\\n• المتاح: `\( {self.bot.available_capital:.2f}`\\n• الإجمالي: ` \){self.bot.current_capital:.2f}`"  
        else:  
            try:  
                balance = self.bot.exchange.fetch_balance()  
                usdt_balance = balance.get('USDT', {})  
                free = usdt_balance.get('free', 0)  
                total = usdt_balance.get('total', 0)  
                  
                return f"🏦 *رصيد المنصة الحقيقي*\\n• المتاح: `\( {free:.2f}`\\n• الإجمالي: ` \){total:.2f}`\\n• محجوز: `${total - free:.2f}`"  
            except Exception as e:  
                return f"❌ خطأ في جلب الرصيد: {str(e)}"  
  
    def handle_drawdown(self, args=None) -> str:  
        max_drawdown = TradingConfig.MAX_TOTAL_DRAWDOWN * 100  
        current_drawdown = self.bot.total_drawdown_percent  
        status = "🟢" if current_drawdown < max_drawdown * 0.8 else "🟡" if current_drawdown < max_drawdown else "🔴"  
          
        return f"""📉 *الخسارة الكلية*

• الحالية: {current_drawdown:.1f}%
• الحد الأقصى: {max_drawdown:.1f}%
• الحالة: {status}"""

    def handle_stop(self, args=None) -> str:  
        return "⏸️ لإيقاف البوت، استخدم Ctrl+C في واجهة التشغيل"  
  
    def handle_restart(self, args=None) -> str:  
        return "🔄 لإعادة التشغيل، أوقف البوت وأعده تشغيل يدوياً"

class MarketFilter:
    def __init__(self, config: TradingConfig):
        self.config = config

    def analyze_market_regime(self, df: pd.DataFrame) -> Tuple[bool, Dict]:  
        if len(df) < 100:  
            return False, {"reason": "بيانات غير كافية < 100 شمعة"}  
          
        try:  
            close = df['close'].values  
            high = df['high'].values  
            low = df['low'].values  
              
            atr = self._calculate_atr(high, low, close, 14)  
            current_atr = atr[-1] if len(atr) > 0 and atr[-1] > 0 else 0  
            atr_percent = current_atr / close[-1] if close[-1] > 0 else 0  
              
            ema_50 = self._calculate_ema(close, 50)  
            ema_slope = self._calculate_slope(ema_50[-20:]) if len(ema_50) >= 20 else 0  
              
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
            logging.getLogger('MarketFilter').error(f"خطأ في تحليل النظام السوقي: {e}")  
            return False, {"error": str(e), "reason": "فشل الحساب → ممنوع التداول"}  
  
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

class BearMarketFilter:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = True
        self.logger = logging.getLogger('BearMarketFilter')

        self.BTC_SYMBOL = "BTC/USDT"  
        self.MIN_BTC_TREND = -0.02  
        self.MAX_DRAWDOWN = -0.15  
          
        self.last_btc_price = None  
        self.market_condition = "NORMAL"  
  
    def analyze_market_condition(self, exchange) -> Dict:  
        if not self.enabled:  
            return {"can_trade": True, "condition": "NORMAL", "reason": "Filter disabled"}  
          
        try:  
            btc_ticker = exchange.fetch_ticker(self.BTC_SYMBOL)  
            current_btc = btc_ticker['last']  
              
            btc_ohlcv = exchange.fetch_ohlcv(self.BTC_SYMBOL, '1d', limit=30)  
            if len(btc_ohlcv) >= 7:  
                btc_df = pd.DataFrame(btc_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])  
                  
                btc_7d_change = (current_btc - btc_df['close'].iloc[-7]) / btc_df['close'].iloc[-7]  
                  
                ma_20 = btc_df['close'].rolling(20).mean().iloc[-1]  
                below_ma_20 = current_btc < ma_20  
                  
                bearish_count = 0  
                total_check = min(10, len(self.config.SYMBOLS))  
                  
                for symbol in self.config.SYMBOLS[:total_check]:  
                    try:  
                        symbol_ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=20)  
                        if len(symbol_ohlcv) >= 20:  
                            symbol_df = pd.DataFrame(symbol_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])  
                              
                            symbol_ma20 = symbol_df['close'].rolling(20).mean().iloc[-1]  
                            if symbol_df['close'].iloc[-1] < symbol_ma20:  
                                bearish_count += 1  
                          
                        time.sleep(0.1)  
                    except:  
                        continue  
                  
                bearish_ratio = bearish_count / total_check if total_check > 0 else 0  
                  
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

class EthicalFilter:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.enabled = True
        self.logger = logging.getLogger('EthicalFilter')

        self.BLACKLIST = [  
            "DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "PEPE/USDT", "BONK/USDT",  
            "FUN/USDT", "CHP/USDT", "BET/USDT", "TRX/USDT", "WIN/USDT",  
            "LUNC/USDT", "USTC/USDT", "XMR/USDT", "ZEC/USDT", "DASH/USDT"  
        ]  
          
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
        if not self.enabled:  
            return True, "Filter disabled"  
          
        if symbol in self.BLACKLIST:  
            return False, f"العملة في القائمة السوداء"  
          
        if self.WHITELIST_MODE and symbol not in self.WHITELIST:  
            return False, f"العملة غير موجودة في القائمة البيضاء"  
          
        symbol_lower = symbol.lower()  
        gambling_keywords = ['bet', 'casino', 'poker', 'gamble', 'lottery', 'dice']  
        if any(keyword in symbol_lower for keyword in gambling_keywords):  
            return False, "تحتوي على كلمات مقامرة"  
          
        meme_keywords = ['dog', 'shib', 'floki', 'pepe', 'bonk', 'elon', 'moon']  
        if any(keyword in symbol_lower for keyword in meme_keywords):  
            return False, "مشروع ميم كوين عالي المخاطرة"  
          
        return True, "OK"  
  
    def filter_symbols(self, ranked_symbols: List[Tuple[str, MarketAnalysis]]) -> List[Tuple[str, MarketAnalysis]]:  
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
        return {  
            "enabled": self.enabled,  
            "blacklist_count": len(self.BLACKLIST),  
            "whitelist_count": len(self.WHITELIST) if self.WHITELIST_MODE else 0,  
            "whitelist_mode": self.WHITELIST_MODE  
        }

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
    ENABLE_EXCHANGE_ORDERS = True

class EnhancedExchangeInterface:
    def __init__(self, exchange, config: ExecutionConfig, mode: TradingMode = TradingMode.PAPER):
        self.exchange = exchange
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger('ExchangeInterface')
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_retries': 0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'partial_fills': 0,
            'platform_orders_created': 0,
            'platform_orders_canceled': 0
        }
        self.api_failures = defaultdict(int)

    def calculate_slippage(self, price: float, side: str) -> float:  
        slippage = price * self.config.SLIPPAGE_PERCENT  
        random_factor = random.uniform(0.8, 1.2)  
        final_slippage = slippage * random_factor  
        max_allowed = price * self.config.MAX_SLIPPAGE  
        return min(final_slippage, max_allowed)  
  
    def calculate_fee(self, amount: float, price: float, is_maker: bool = False) -> float:  
        fee_rate = self.config.MAKER_FEE if is_maker else self.config.TAKER_FEE  
        value = amount * price  
        return value * fee_rate  
  
    def format_amount(self, symbol: str, amount: float) -> float:  
        try:  
            market = self.exchange.market(symbol)  
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)  
            precision = market.get('precision', {}).get('amount', 8)  
              
            if amount < min_amount:  
                return 0.0  
              
            formatted = Decimal(str(amount)).quantize(  
                Decimal('1.' + '0' * precision),  
                rounding=ROUND_HALF_UP  
            )  
            return float(formatted)  
        except:  
            return amount  
  
    def format_price(self, symbol: str, price: float) -> float:
        try:
            market = self.exchange.market(symbol)
            # إصلاح حرج #4 – استخدام tickSize الحقيقي من PRICE_FILTER (Binance)
            for f in market.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    tick_size = float(f['tickSize'])
                    return tick_size * round(price / tick_size)
            # fallback إذا لم يوجد الفلتر (نادر)
            precision = market['precision'].get('price', 8)
            return round(price, precision)
        except Exception:
            return price  
  
    def create_stop_loss_order(self, symbol: str, amount: float, stop_price: float) -> Optional[str]:  
        if self.mode != TradingMode.LIVE or not self.config.ENABLE_EXCHANGE_ORDERS:  
            return None  
      
        try:  
            formatted_amount = self.format_amount(symbol, amount)  
            formatted_stop_price = self.format_price(symbol, stop_price)  
              
            params = {  
                'stopPrice': formatted_stop_price,  
                'type': 'STOP_LOSS_MARKET'  # إصلاح حرج: استخدام STOP_LOSS_MARKET لتوافق أفضل
            }  
              
            order = self.exchange.create_order(  
                symbol=symbol,  
                type='stop_loss_market',  
                side='sell',  
                amount=formatted_amount,  
                params=params  
            )  
              
            order_id = order.get('id')  
            if order_id:  
                self.execution_stats['platform_orders_created'] += 1  
                self.logger.info(f"✅ تم إنشاء أمر Stop-Loss MARKET على المنصة: {order_id}")  
                return order_id  
              
        except Exception as e:  
            self.logger.error(f"❌ فشل إنشاء أمر Stop-Loss MARKET على المنصة: {e}")  
      
        return None  
  
    def create_take_profit_order(self, symbol: str, amount: float, limit_price: float) -> Optional[str]:  
        if self.mode != TradingMode.LIVE or not self.config.ENABLE_EXCHANGE_ORDERS:  
            return None  
      
        try:  
            formatted_amount = self.format_amount(symbol, amount)  
            formatted_limit_price = self.format_price(symbol, limit_price)  
              
            order = self.exchange.create_limit_sell_order(  
                symbol=symbol,  
                amount=formatted_amount,  
                price=formatted_limit_price  
            )  
              
            order_id = order.get('id')  
            if order_id:  
                self.execution_stats['platform_orders_created'] += 1  
                self.logger.info(f"✅ تم إنشاء أمر Take-Profit على المنصة: {order_id}")  
                return order_id  
              
        except Exception as e:  
            self.logger.error(f"❌ فشل إنشاء أمر Take-Profit على المنصة: {e}")  
      
        return None  
  
    def cancel_order(self, symbol: str, order_id: str) -> bool:  
        if self.mode != TradingMode.LIVE:  
            return False  
      
        try:  
            result = self.exchange.cancel_order(order_id, symbol)  
            if result:  
                self.execution_stats['platform_orders_canceled'] += 1  
                self.logger.info(f"✅ تم إلغاء الأمر على المنصة: {order_id}")  
                return True  
        except Exception as e:  
            self.logger.error(f"❌ فشل إلغاء الأمر {order_id}: {e}")  
      
        return False  
  
    def check_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:  
        if self.mode != TradingMode.LIVE:  
            return None  
      
        try:  
            order = self.exchange.fetch_order(order_id, symbol)  
            return {  
                'status': order.get('status'),  
                'filled': order.get('filled', 0),  
                'remaining': order.get('remaining', 0)  
            }  
        except Exception as e:  
            self.logger.error(f"❌ فشل فحص حالة الأمر {order_id}: {e}")  
      
        return None  
  
    def execute_market_order(self, symbol: str, side: str, amount: float, max_retries: int = None) -> OrderResult:  
        if max_retries is None:  
            max_retries = self.config.MAX_RETRIES  
      
        formatted_amount = self.format_amount(symbol, amount)  
        if formatted_amount <= 0:  
            return OrderResult(  
                order_id="", symbol=symbol, order_type=OrderType.MARKET,  
                side=side, amount=amount, price=0.0, filled=0.0,  
                remaining=amount, status=OrderStatus.REJECTED,  
                error="الكمية غير صالحة"  
            )  
      
        if self.mode == TradingMode.PAPER:  
            return self._simulate_order(symbol, side, formatted_amount, OrderType.MARKET)  
      
        for attempt in range(max_retries + 1):  
            try:  
                order = self.exchange.create_market_order(symbol, side, formatted_amount)  
                time.sleep(1)  
                  
                order_id = order.get('id', '')  
                if order_id:  
                    order_details = self.exchange.fetch_order(order_id, symbol)  
                else:  
                    order_details = order  
                  
                result = self._parse_order_result(order_details, OrderType.MARKET)  
                result.retry_count = attempt  
                  
                if result.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:  
                    self.execution_stats['successful_orders'] += 1  
                    return result  
                      
            except ccxt.NetworkError as e:  
                if attempt < max_retries:  
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))  
                continue  
            except Exception as e:  
                break  
      
        self.execution_stats['failed_orders'] += 1  
        return OrderResult(  
            order_id="", symbol=symbol, order_type=OrderType.MARKET,  
            side=side, amount=formatted_amount, price=0.0,  
            filled=0.0, remaining=formatted_amount,  
            status=OrderStatus.REJECTED,  
            error=f"فشل بعد {max_retries + 1} محاولات",  
            retry_count=max_retries  
        )  
  
    def execute_limit_order(self, symbol: str, side: str, amount: float, price: float, max_retries: int = None) -> OrderResult:  
        if max_retries is None:  
            max_retries = self.config.MAX_RETRIES  
      
        formatted_amount = self.format_amount(symbol, amount)  
        formatted_price = self.format_price(symbol, price)  
      
        if formatted_amount <= 0:  
            return OrderResult(  
                order_id="", symbol=symbol, order_type=OrderType.LIMIT,  
                side=side, amount=amount, price=0.0, filled=0.0,  
                remaining=amount, status=OrderStatus.REJECTED,  
                error="الكمية غير صالحة"  
            )  
      
        if self.mode == TradingMode.PAPER:  
            return self._simulate_order(symbol, side, formatted_amount, OrderType.LIMIT)  
      
        for attempt in range(max_retries + 1):  
            try:  
                params = {'postOnly': True} if self.config.USE_POST_ONLY else {}  
                order = self.exchange.create_limit_order(  
                    symbol, side, formatted_amount, formatted_price, params  
                )  
                  
                order_id = order.get('id', '')  
                start_time = time.time()  
              
                while time.time() - start_time < self.config.TIMEOUT_SECONDS:  
                    order_details = self.exchange.fetch_order(order_id, symbol)  
                    status = order_details.get('status', '')  
                      
                    if status in ['closed', 'filled', 'canceled', 'expired']:  
                        break  
                      
                    filled = order_details.get('filled', 0)  
                    if filled > 0 and self.config.ALLOW_PARTIAL_FILLS:  
                        fill_percent = filled / formatted_amount  
                        if fill_percent >= self.config.MIN_FILL_PERCENT:  
                            break  
                      
                    time.sleep(2)  
              
                result = self._parse_order_result(order_details, OrderType.LIMIT)  
                result.retry_count = attempt  
              
                if result.status == OrderStatus.PARTIAL:  
                    self.execution_stats['partial_fills'] += 1  
              
                if result.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:  
                    self.execution_stats['successful_orders'] += 1  
                    return result  
                else:  
                    try:  
                        self.exchange.cancel_order(order_id, symbol)  
                    except:  
                        pass  
                    return result  
                      
            except ccxt.NetworkError as e:  
                if attempt < max_retries:  
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))  
                continue  
            except Exception as e:  
                break  
      
        self.execution_stats['failed_orders'] += 1  
        return OrderResult(  
            order_id="", symbol=symbol, order_type=OrderType.LIMIT,  
            side=side, amount=formatted_amount, price=formatted_price,  
            filled=0.0, remaining=formatted_amount,  
            status=OrderStatus.REJECTED,  
            error=f"فشل بعد {max_retries + 1} محاولات",  
            retry_count=max_retries  
        )  
  
    def _simulate_order(self, symbol: str, side: str, amount: float, order_type: OrderType) -> OrderResult:  
        try:  
            ticker = self.exchange.fetch_ticker(symbol)  
            current_price = ticker['last']  
              
            slippage = self.calculate_slippage(current_price, side)  
              
            if side == 'buy':  
                execution_price = current_price + slippage  
            else:  
                execution_price = current_price - slippage  
              
            fee = self.calculate_fee(amount, execution_price, is_maker=(order_type == OrderType.LIMIT))  
              
            return OrderResult(  
                order_id=f"SIM-{int(time.time())}-{random.randint(1000, 9999)}",  
                symbol=symbol,  
                order_type=order_type,  
                side=side,  
                amount=amount,  
                price=execution_price,  
                filled=amount,  
                remaining=0.0,  
                status=OrderStatus.FILLED,  
                average_price=execution_price,  
                cost=amount * execution_price,  
                fee=fee,  
                fee_currency='USDT',  
                is_simulated=True  
            )  
        except Exception as e:  
            return OrderResult(  
                order_id="", symbol=symbol, order_type=order_type,  
                side=side, amount=amount, price=0.0, filled=0.0,  
                remaining=amount, status=OrderStatus.REJECTED,  
                error=str(e), is_simulated=True  
            )  
  
    def _parse_order_result(self, order_data: Dict, order_type: OrderType) -> OrderResult:  
        try:  
            ccxt_status = order_data.get('status', '')  
            filled = float(order_data.get('filled', 0))  
            amount = float(order_data.get('amount', 0))  
              
            # إصلاح حرج: توحيد حالات الأوامر
            status_map = {  
                'open': OrderStatus.OPEN,  
                'closed': OrderStatus.CLOSED,  
                'canceled': OrderStatus.CANCELED,  
                'expired': OrderStatus.EXPIRED,  
                'rejected': OrderStatus.REJECTED,  
                'filled': OrderStatus.FILLED,  
                'partially_filled': OrderStatus.PARTIAL,  # إضافة partial من ccxt
            }  
            status = status_map.get(ccxt_status.lower(), OrderStatus.PENDING)  
              
            if filled >= amount:  
                status = OrderStatus.FILLED  
            elif filled > 0:  
                status = OrderStatus.PARTIAL  
              
            average_price = 0.0  
            if filled > 0:  
                cost = float(order_data.get('cost', 0))  
                average_price = cost / filled if cost > 0 else float(order_data.get('price', 0))  
              
            fee_data = order_data.get('fee', {})  
            if isinstance(fee_data, dict):  
                fee = fee_data.get('cost', 0)  
                fee_currency = fee_data.get('currency', '')  
            else:  
                fee = fee_data if fee_data else 0  
                fee_currency = ''  
              
            self.execution_stats['total_fees'] += fee  
              
            return OrderResult(  
                order_id=str(order_data.get('id', '')),  
                symbol=order_data.get('symbol', ''),  
                order_type=order_type,  
                side=order_data.get('side', ''),  
                amount=amount,  
                price=float(order_data.get('price', 0)),  
                filled=filled,  
                remaining=amount - filled,  
                status=status,  
                average_price=average_price,  
                cost=float(order_data.get('cost', 0)),  
                fee=fee,  
                fee_currency=fee_currency,  
                timestamp=order_data.get('timestamp', datetime.now(timezone.utc).isoformat())  
            )  
        except Exception as e:  
            self.logger.error(f"خطأ في parse_order_result: {e}")  
            return OrderResult(  
                order_id="", symbol="", order_type=order_type,  
                side="", amount=0, price=0, filled=0, remaining=0,  
                status=OrderStatus.REJECTED, error=str(e)  
            )  
  
    def get_execution_stats(self) -> Dict:  
        return self.execution_stats.copy()

class ExecutionManager:
    def __init__(self, exchange_interface: EnhancedExchangeInterface):
        self.exchange = exchange_interface
        self.config = exchange_interface.config
        self.logger = logging.getLogger('ExecutionManager')
        self.trade_executions: Dict[str, TradeExecution] = {}

    def create_exchange_orders(self, trade: TradeRecord) -> Tuple[bool, str]:  
        if self.exchange.mode != TradingMode.LIVE:  
            return True, "وضع المحاكاة - لا حاجة لأوامر المنصة"  
      
        try:  
            if trade.stop_loss and not trade.stop_loss_order_id:  
                stop_order_id = self.exchange.create_stop_loss_order(  
                    trade.symbol, trade.quantity, trade.stop_loss  
                )  
                if stop_order_id:  
                    trade.stop_loss_order_id = stop_order_id  
                    self.logger.info(f"✅ تم إنشاء أمر Stop-Loss على المنصة للصفقة {trade.trade_id}")  
                else:  
                    self.logger.warning(f"⚠️ فشل إنشاء أمر Stop-Loss على المنصة للصفقة {trade.trade_id}")  
              
            if trade.take_profit and not trade.take_profit_order_id:  
                tp_order_id = self.exchange.create_take_profit_order(  
                    trade.symbol, trade.quantity, trade.take_profit  
                )  
                if tp_order_id:  
                    trade.take_profit_order_id = tp_order_id  
                    self.logger.info(f"✅ تم إنشاء أمر Take-Profit على المنصة للصفقة {trade.trade_id}")  
                else:  
                    self.logger.warning(f"⚠️ فشل إنشاء أمر Take-Profit على المنصة للصفقة {trade.trade_id}")  
              
            return True, "تم إنشاء أوامر المنصة"  
              
        except Exception as e:  
            return False, str(e)  
  
    def update_exchange_orders(self, trade: TradeRecord) -> Tuple[bool, str]:  
        if self.exchange.mode != TradingMode.LIVE:  
            return True, "وضع المحاكاة - لا حاجة لتحديث أوامر المنصة"  
      
        try:  
            updated = False  
              
            if trade.stop_loss_modified and trade.stop_loss_order_id:  
                self.exchange.cancel_order(trade.symbol, trade.stop_loss_order_id)  
                  
                new_stop_order_id = self.exchange.create_stop_loss_order(  
                    trade.symbol, trade.quantity, trade.stop_loss  
                )  
                  
                if new_stop_order_id:  
                    trade.stop_loss_order_id = new_stop_order_id  
                    trade.stop_loss_modified = False  
                    updated = True  
                    self.logger.info(f"🔄 تم تحديث أمر Stop-Loss على المنصة للصفقة {trade.trade_id}")  
              
            if trade.take_profit is None and trade.take_profit_order_id:  
                self.exchange.cancel_order(trade.symbol, trade.take_profit_order_id)  
                trade.take_profit_order_id = None  
                trade.take_profit_modified = False  
                updated = True  
                self.logger.info(f"🔄 تم إلغاء أمر Take-Profit على المنصة للصفقة {trade.trade_id}")  
              
            return True, "تم تحديث أوامر المنصة" if updated else "لا حاجة للتحديث"  
              
        except Exception as e:  
            return False, str(e)  
  
    def check_exchange_order_status(self, trade: TradeRecord) -> Optional[str]:  
        if self.exchange.mode != TradingMode.LIVE:  
            return None  
      
        try:  
            if trade.stop_loss_order_id:  
                status = self.exchange.check_order_status(trade.symbol, trade.stop_loss_order_id)  
                if status and status.get('status') in ['filled', 'closed']:  
                    return "STOP_LOSS (منصة)"  
              
            if trade.take_profit_order_id:  
                status = self.exchange.check_order_status(trade.symbol, trade.take_profit_order_id)  
                if status and status.get('status') in ['filled', 'closed']:  
                    return "TAKE_PROFIT (منصة)"  
      
        except Exception as e:  
            self.logger.error(f"خطأ في فحص حالة أوامر المنصة: {e}")  
      
        return None  
  
    def cleanup_exchange_orders(self, trade: TradeRecord) -> bool:  
        if self.exchange.mode != TradingMode.LIVE:  
            return True  
      
        try:  
            success = True  
              
            if trade.stop_loss_order_id:  
                if not self.exchange.cancel_order(trade.symbol, trade.stop_loss_order_id):  
                    success = False  
              
            if trade.take_profit_order_id:  
                if not self.exchange.cancel_order(trade.symbol, trade.take_profit_order_id):  
                    success = False  
              
            return success  
              
        except Exception as e:  
            self.logger.error(f"خطأ في تنظيف أوامر المنصة: {e}")  
            return False  
  
    def execute_entry(self, symbol: str, amount: float, price: float = None, order_type: OrderType = None) -> Tuple[Optional[TradeExecution], Optional[str]]:  
        try:  
            if self.exchange.mode == TradingMode.LIVE:  
                market = self.exchange.exchange.market(symbol)  
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)  
                if amount < min_amount:  
                    return None, "AMOUNT_TOO_SMALL"  
                  
                balance = self.exchange.exchange.fetch_balance()  
                usdt_free = balance.get('USDT', {}).get('free', 0)  
                cost = amount * (price if price else self.exchange.exchange.fetch_ticker(symbol)['last'])  
                if cost > usdt_free:  
                    return None, "INSUFFICIENT_BALANCE"  
              
            # إصلاح حرج #1 – استخدام المولّد الموحد
            trade_id = f"TRD-{time.time_ns()}-{random.randint(100000, 999999)}"
              
            if order_type is None:  
                order_type = self.config.DEFAULT_ORDER_TYPE  
              
            if order_type == OrderType.MARKET:  
                order_result = self.exchange.execute_market_order(symbol, 'buy', amount)  
            else:  
                if price is None:  
                    ticker = self.exchange.exchange.fetch_ticker(symbol)  
                    price = ticker['last']  
                order_result = self.exchange.execute_limit_order(symbol, 'buy', amount, price)  
              
            if order_result.status in [OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED]:  
                return None, order_result.error or f"فشل تنفيذ الأمر: {order_result.status.value}"  
              
            slippage = 0.0  
            if order_result.average_price and order_result.price:  
                slippage = abs(order_result.average_price - order_result.price)  
                self.exchange.execution_stats['total_slippage'] += slippage  
              
            net_price = order_result.average_price or order_result.price  
            if net_price > 0 and order_result.filled > 0:  
                fee_per_unit = order_result.fee / order_result.filled if order_result.fee else 0  
                net_price += fee_per_unit  
              
            execution = TradeExecution(  
                trade_id=trade_id,  
                symbol=symbol,  
                entry_order=order_result,  
                entry_fee=order_result.fee or 0,  
                slippage_entry=slippage,  
                net_price_entry=net_price,  
                total_fees=order_result.fee or 0  
            )  
              
            self.trade_executions[trade_id] = execution  
            return execution, None  
              
        except Exception as e:  
            return None, str(e)  
  
    def execute_exit(self, symbol: str, amount: float, trade_id: str = None, price: float = None, order_type: OrderType = None) -> Tuple[Optional[TradeExecution], Optional[str]]:
        # إصلاح حرج #2 – المعامل trade_id هو المفتاح الصحيح (كان execution_id سابقاً)
        try:  
            execution = None
            if trade_id and trade_id in self.trade_executions:  
                execution = self.trade_executions[trade_id]  
                symbol = execution.symbol  
                if execution.entry_order:
                    amount = execution.entry_order.filled  
              
            if order_type is None:  
                order_type = self.config.DEFAULT_ORDER_TYPE  
              
            if order_type == OrderType.MARKET:  
                order_result = self.exchange.execute_market_order(symbol, 'sell', amount)  
            else:  
                if price is None:  
                    ticker = self.exchange.exchange.fetch_ticker(symbol)  
                    price = ticker['last']  
                order_result = self.exchange.execute_limit_order(symbol, 'sell', amount, price)  
              
            if order_result.status in [OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED]:  
                if order_type == OrderType.LIMIT:  
                    order_result = self.exchange.execute_market_order(symbol, 'sell', amount)  
                    if order_result.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:  
                        return None, "فشل الخروج"  
              
            slippage = 0.0  
            if order_result.average_price and order_result.price:  
                slippage = abs(order_result.average_price - order_result.price)  
                self.exchange.execution_stats['total_slippage'] += slippage  
              
            net_price = order_result.average_price or order_result.price  
            if net_price > 0 and order_result.filled > 0:  
                fee_per_unit = order_result.fee / order_result.filled if order_result.fee else 0  
                net_price -= fee_per_unit  
              
            if execution:  
                execution.exit_order = order_result  
                execution.exit_fee = order_result.fee or 0  
                execution.slippage_exit = slippage  
                execution.net_price_exit = net_price  
                execution.total_fees += (order_result.fee or 0)  
                  
                if execution.net_price_entry > 0 and net_price > 0:  
                    execution.net_pnl = (net_price - execution.net_price_entry) * order_result.filled  
                  
                return execution, None  
            else:  
                execution = TradeExecution(  
                    trade_id=trade_id or f"EXIT-{int(time.time())}",  
                    symbol=symbol,  
                    entry_order=None,  
                    exit_order=order_result,  
                    exit_fee=order_result.fee or 0,  
                    slippage_exit=slippage,  
                    net_price_exit=net_price,  
                    total_fees=order_result.fee or 0  
                )  
                  
                self.trade_executions[execution.trade_id] = execution  
                return execution, None  
              
        except Exception as e:  
            return None, str(e)

class Monitor:
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger('Monitor')
        self.daily_stats = {
            'date': datetime.now(timezone.utc).date(),
            'trades_opened': 0,
            'trades_closed': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'execution_success': 0,
            'execution_failed': 0,
            'platform_orders_created': 0,
            'platform_orders_canceled': 0
        }

    def update_stats(self, trade: TradeRecord, execution: TradeExecution = None):  
        if trade.status == "ACTIVE":  
            self.daily_stats['trades_opened'] += 1  
        elif trade.status == "CLOSED":  
            self.daily_stats['trades_closed'] += 1  
            if trade.pnl:  
                self.daily_stats['total_pnl'] += trade.pnl  
      
        if execution:  
            self.daily_stats['total_fees'] += execution.total_fees  
            self.daily_stats['total_slippage'] += (execution.slippage_entry + (execution.slippage_exit or 0))  
  
    def update_platform_stats(self, stats: Dict):  
        self.daily_stats['platform_orders_created'] = stats.get('platform_orders_created', 0)  
        self.daily_stats['platform_orders_canceled'] = stats.get('platform_orders_canceled', 0)  
  
    def generate_daily_report(self) -> str:  
        today = datetime.now(timezone.utc).date()  
          
        if today > self.daily_stats['date']:  
            self._save_report()  
            self.daily_stats = {  
                'date': today,  
                'trades_opened': 0,  
                'trades_closed': 0,  
                'total_pnl': 0.0,  
                'total_fees': 0.0,  
                'total_slippage': 0.0,  
                'execution_success': 0,  
                'execution_failed': 0,  
                'platform_orders_created': 0,  
                'platform_orders_canceled': 0  
            }  
          
        total_executions = self.daily_stats['execution_success'] + self.daily_stats['execution_failed']  
        success_rate = (self.daily_stats['execution_success'] / total_executions * 100) if total_executions > 0 else 0  
          
        avg_slippage = 0  
        if self.daily_stats['trades_closed'] > 0:  
            avg_slippage = self.daily_stats['total_slippage'] / (self.daily_stats['trades_closed'] * 2)  
          
        avg_fees = 0  
        if self.daily_stats['trades_closed'] > 0:  
            avg_fees = self.daily_stats['total_fees'] / self.daily_stats['trades_closed']  
          
        report = f"""

📅 تقرير يومي - {today}

📊 الصفقات:
• المفتوحة: {self.daily_stats['trades_opened']}
• المغلقة: {self.daily_stats['trades_closed']}
• P&L اليوم: ${self.daily_stats['total_pnl']:.2f}

🎯 التنفيذ:
• معدل النجاح: {success_rate:.1f}%
• متوسط الانزلاق: {avg_slippage:.4f}%
• متوسط الرسوم: ${avg_fees:.4f}
• أوامر المنصة: {self.daily_stats['platform_orders_created']}

💰 رأس المال:
• الحالي: ${self.bot.current_capital:.2f}
• المتاح: ${self.bot.available_capital:.2f}
• الخسارة الكلية: {self.bot.total_drawdown_percent:.1f}%

"""  
          
        return report  
  
    def _save_report(self):  
        if not os.path.exists('reports'):  
            os.makedirs('reports')  
          
        filename = f"reports/report_{self.daily_stats['date']}.json"  
        with open(filename, 'w') as f:  
            json.dump(self.daily_stats, f, indent=2)

class StableBotPro:
    def __init__(self, trading_mode: TradingMode = TradingMode.PAPER):
        self.config = TradingConfig
        self.trading_mode = trading_mode

        # ─── إصلاحات أمان عامة ───
        self.is_halted = False                      # Global Kill Switch
        self.api_failure_count = 0
        self.max_api_failures_before_halt = 12      # قيمة معقولة

        # فصل الحسابات بوضوح
        self.initial_capital_snapshot = self.config.INITIAL_CAPITAL
        self.live_total_balance = 0.0
        self.live_available_balance = 0.0
        self.paper_total = self.config.INITIAL_CAPITAL
        self.paper_available = self.config.INITIAL_CAPITAL
        self.daily_pnl = 0.0
        self.total_drawdown_percent = 0.0

        # استخدام RLock لتقليل مخاطر race condition
        self.state_lock = threading.RLock()

        exchange_params = {  
            'enableRateLimit': True,  
            'options': {'defaultType': 'spot'}  
        }  
          
        if trading_mode == TradingMode.LIVE:  
            exchange_params.update({  
                'apiKey': os.getenv('BINANCE_API_KEY'),  
                'secret': os.getenv('BINANCE_API_SECRET')  
            })  
          
        self.exchange = ccxt.binance(exchange_params)  
          
        self.execution_config = ExecutionConfig()  
        self.exchange_interface = EnhancedExchangeInterface(  
            self.exchange, self.execution_config, trading_mode  
        )  
        self.execution_manager = ExecutionManager(self.exchange_interface)  
        self.monitor = Monitor(self)  
          
        self.active_trades: Dict[str, TradeRecord] = {}  
        self.loss_streak = 0  
        self.cooling_off_end = 0  
          
        self.market_filter = MarketFilter(self.config)
        self.correlation_filter = CorrelationFilter(self.config)
        self.risk_manager = DynamicRiskManager(self.config)
          
        self.bear_market_filter = BearMarketFilter(self.config)
        self.ethical_filter = EthicalFilter(self.config)
          
        self.mode_manager = ModeManager(self)
        self.command_handler = TelegramCommandHandler(self)
          
        self.previous_scores = defaultdict(float)  
          
        self.api_failures = defaultdict(int)  
          
        for path in ['logs', 'data/active_trades', 'data/closed_trades', 'data/executions', 'reports']:  
            os.makedirs(path, exist_ok=True)  
          
        self.logger = Logger.setup('StableBot')  
        self._load_active_trades()  
          
        self.logger.info(f"🚀 بدء StableBotPro v4.1 - وضع: {trading_mode.value}")  
          
        if trading_mode == TradingMode.LIVE:  
            self.logger.warning("⚠️  وضع التداول الحي مفعّل!")  
            success, msg = self._sync_live_balance()  
            if not success:  
                self.logger.error(f"❌ فشل مزامنة الرصيد: {msg}")  
            else:  
                self.logger.info(f"✅ تم مزامنة الرصيد من المنصة: ${self.live_available_balance:.2f}")  
          
        self.logger.info(f"✅ تم تحميل الفلاتر الجديدة:")  
        self.logger.info(f"   • Bear Market Filter: {'مفعل' if self.bear_market_filter.enabled else 'معطل'}")  
        self.logger.info(f"   • Ethical Filter: {'مفعل' if self.ethical_filter.enabled else 'معطل'}")  
        self.logger.info(f"   • Mode Manager: جاهز")  
        self.logger.info(f"   • Max Total Drawdown: {self.config.MAX_TOTAL_DRAWDOWN*100:.1f}%")  

    def _generate_trade_id(self) -> str:
        # إصلاح حرج #1 – مولّد trade_id آمن وفريد عالمياً (nanoseconds + random قوي)
        ns = time.time_ns()
        rand_part = random.randint(100000, 999999)
        return f"TRD-{ns}-{rand_part}"

    def _sync_live_balance(self) -> Tuple[bool, str]:  
        if self.trading_mode != TradingMode.LIVE:  
            return True, "وضع المحاكاة - لا حاجة للمزامنة"  
      
        try:  
            with self.state_lock:  
                balance = self._retry_api(self.exchange.fetch_balance)  
                usdt_balance = balance.get('USDT', {})  
                free_balance = usdt_balance.get('free', 0)  
                total_balance = usdt_balance.get('total', 0)  
                  
                if free_balance <= 0:  
                    return False, "الرصيد المتاح غير صالح أو صفري"  
                  
                self.live_available_balance = free_balance  
                self.live_total_balance = total_balance  
                self.initial_capital_snapshot = total_balance  
                  
                self.logger.info(f"💰 تم مزامنة الرصيد: ${free_balance:.2f} متاح، ${total_balance:.2f} إجمالي")  
                return True, f"تم المزامنة: ${free_balance:.2f}"  
                  
        except Exception as e:  
            return False, f"خطأ في مزامنة الرصيد: {str(e)}"  
  
    def _update_live_balance(self) -> bool:  
        if self.trading_mode != TradingMode.LIVE:  
            return True  
      
        try:  
            balance = self._retry_api(self.exchange.fetch_balance)  
            usdt_balance = balance.get('USDT', {})  
            free_balance = usdt_balance.get('free', 0)  
            total_balance = usdt_balance.get('total', 0)  
              
            with self.state_lock:  
                self.live_available_balance = free_balance  
                self.live_total_balance = total_balance  
                  
                if self.initial_capital_snapshot > 0:  
                    self.total_drawdown_percent = ((self.initial_capital_snapshot - total_balance) /   
                                                  self.initial_capital_snapshot * 100)  
          
            return True  
              
        except Exception as e:  
            self.logger.error(f"خطأ في تحديث الرصيد: {e}")  
            return False  
  
    def _check_total_drawdown(self) -> bool:  
        max_drawdown = self.config.MAX_TOTAL_DRAWDOWN * 100  
          
        with self.state_lock:  
            if self.total_drawdown_percent >= max_drawdown:  
                self.logger.critical(f"🚨 تم تجاوز الحد الأقصى للخسارة الكلية: {self.total_drawdown_percent:.1f}% >= {max_drawdown:.1f}%")  
                self._send_notification(  
                    f"🚨 *توقف النظام - تجاوز الحد الأقصى للخسارة*\\n"  
                    f"• الخسارة الحالية: `{self.total_drawdown_percent:.1f}%`\\n"  
                    f"• الحد الأقصى: `{max_drawdown:.1f}%`\\n"  
                    f"• تم إيقاف جميع التداولات الجديدة"  
                )  
                return False  
          
            elif self.total_drawdown_percent >= max_drawdown * 0.8:  
                self.logger.warning(f"⚠️  اقتراب من الحد الأقصى للخسارة: {self.total_drawdown_percent:.1f}%")  
          
        return True  
  
    def _load_active_trades(self):  
        path = 'data/active_trades'  
        if not os.path.exists(path):  
            return  
          
        loaded = 0  
        for filename in os.listdir(path):  
            if filename.endswith('.json'):  
                try:  
                    with open(f"{path}/{filename}", 'r') as f:  
                        data = json.load(f)  
                        if data.get('take_profit') == 'inf' or data.get('take_profit') is None:  
                            data['take_profit'] = None  
                        trade = TradeRecord(**data)  
                        self.active_trades[trade.trade_id] = trade  
                        loaded += 1  
                except Exception as e:  
                    self.logger.error(f"خطأ في تحميل {filename}: {e}")  
          
        self.logger.info(f"تم تحميل {loaded} صفقة من الذاكرة")  
  
    def _save_trade(self, trade: TradeRecord):  
        try:  
            filename = f"data/active_trades/{trade.trade_id}.json"  
            trade_dict = asdict(trade)  
            if trade_dict['take_profit'] is None:  
                trade_dict['take_profit'] = None  
            with open(filename, 'w', encoding='utf-8') as f:  
                json.dump(trade_dict, f, indent=2, default=str, ensure_ascii=False)  
        except Exception as e:  
            self.logger.error(f"خطأ في حفظ الصفقة: {e}")  
  
    def _save_execution(self, execution: TradeExecution):  
        try:  
            filename = f"data/executions/{execution.trade_id}.json"  
            with open(filename, 'w', encoding='utf-8') as f:  
                json.dump(asdict(execution), f, indent=2, default=str, ensure_ascii=False)  
        except Exception as e:  
            self.logger.error(f"خطأ في حفظ سجل التنفيذ: {e}")  
  
    def _move_to_closed(self, trade_id: str):  
        try:  
            src = f"data/active_trades/{trade_id}.json"  
            dst = f"data/closed_trades/{trade_id}.json"  
            if os.path.exists(src):  
                os.rename(src, dst)  
        except Exception as e:  
            self.logger.error(f"خطأ في نقل الصفقة: {e}")  
  
    def _send_notification(self, message: str):  
        if not self.config.TELEGRAM_TOKEN or not self.config.TELEGRAM_CHAT_ID:  
            return  
      
        try:  
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_TOKEN}/sendMessage"  
            payload = {  
                "chat_id": self.config.TELEGRAM_CHAT_ID,  
                "text": f"🤖 *StableBot v4.1:*\\n{message}",  
                "parse_mode": "Markdown"  
            }  
            requests.post(url, json=payload, timeout=10)  
        except Exception as e:  
            self.logger.error(f"خطأ تلغرام: {e}")  
  
    def _retry_api(self, func, *args, **kwargs):  
        backoff = TradingConfig.API_RETRY_BACKOFF  
        symbol = args[0] if args and isinstance(args[0], str) else "UNKNOWN"  
        for attempt in range(TradingConfig.API_RETRY_MAX):  
            try:  
                result = func(*args, **kwargs)  
                self.api_failures[symbol] = 0  
                return result  
            except Exception as e:  
                self.logger.warning(f"API فشل محاولة {attempt + 1}: {e}")  
                self.api_failures[symbol] += 1  
                if self.api_failures[symbol] >= TradingConfig.CIRCUIT_BREAKER_MAX_FAILS:  
                    self.logger.error(f"circuit breaker: تجاهل {symbol} مؤقتًا")  
                    raise  
                time.sleep(backoff)  
                backoff *= 2  
        raise  
  
    def _check_liquidity_and_spread(self, symbol, df: pd.DataFrame) -> Tuple[bool, str]:  
        avg_volume = df['volume'].rolling(20).mean().iloc[-1] * df['close'].iloc[-1]  
        if avg_volume < TradingConfig.MIN_LIQUIDITY_USDT:  
            return False, "LOW_LIQUIDITY"  
          
        ticker = self._retry_api(self.exchange.fetch_ticker, symbol)  
        bid = ticker['bid']  
        ask = ticker['ask']  
        if bid and ask:  
            mid = (bid + ask) / 2  
            spread = (ask - bid) / mid  
            if spread > TradingConfig.MAX_SPREAD_PERCENT:  
                return False, "HIGH_SPREAD"  
          
        return True, "OK"  
  
    def _get_ai_confidence(self, df: pd.DataFrame) -> float:  
        close = df['close']  
        price_std = close.iloc[-10:].std() / close.iloc[-1]  
          
        volume = df['volume']  
        volume_std = volume.iloc[-10:].std() / volume.iloc[-1] if volume.iloc[-1] > 0 else 1.0  
          
        atr_percent = self.market_filter.analyze_market_regime(df)[1].get('atr_percent', 0)  
          
        confidence = 1.0  
        if price_std > 0.02:  
            confidence -= 0.3  
        if volume_std > 0.5:  
            confidence -= 0.2  
        if atr_percent < 0.01:  
            confidence -= 0.25  
          
        confidence = max(0.35, min(1.0, confidence))  
          
        return confidence  
  
    def _check_cooling_off(self) -> bool:  
        if time.time() < self.cooling_off_end:  
            return False  
        return True  
  
    def _update_loss_streak(self, pnl: float):  
        if pnl < 0:  
            self.loss_streak += 1  
            if self.loss_streak >= TradingConfig.COOLING_OFF_LOSS_STREAK:  
                self.cooling_off_end = time.time() + TradingConfig.COOLING_OFF_DURATION  
                self.loss_streak = 0  
                self.logger.warning("⚠️ تفعيل النوم القسري لـ 24 ساعة")  
                self._send_notification("⚠️ تفعيل النوم القسري بعد خسائر متتالية")  
        else:  
            self.loss_streak = 0  
  
    def _check_max_trade_age(self, trade: TradeRecord) -> bool:  
        age = time.time() - trade.open_time_timestamp  
        if age > TradingConfig.MAX_TRADE_AGE:  
            return True  
        return False  
  
    def calculate_score(self, df: pd.DataFrame) -> float:  
        try:  
            if len(df) < 30:  
                return 0  
              
            close = df['close']  
              
            net_move = abs(close.iloc[-1] - close.iloc[-10])  
            total_path = close.diff().abs().iloc[-10:].sum()  
            efficiency = (net_move / total_path * 40) if total_path > 0 else 0  
              
            sma_20 = close.rolling(20).mean().iloc[-1]  
            trend = 20 if close.iloc[-1] > sma_20 else 5  
              
            last_candle = df.iloc[-1]  
            candle_range = last_candle['high'] - last_candle['low']  
              
            if candle_range > 0:  
                upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])  
                upper_wick_ratio = upper_wick / candle_range  
                penalty = upper_wick_ratio * 30  
            else:  
                penalty = 0  
              
            volume_avg = df['volume'].iloc[-10:].mean()  
            volume_current = df['volume'].iloc[-1]  
            volume_score = 10 if volume_current > volume_avg else 5  
              
            score = efficiency + trend + volume_score - penalty  
            return max(0, min(100, round(score, 2)))  
              
        except Exception as e:  
            self.logger.error(f"خطأ في حساب السكور: {e}")  
            return 0  
  
    def scan_symbols(self) -> Dict[str, MarketAnalysis]:  
        scan_results = {}  
        scanned = 0  
        rejected = 0  
          
        for symbol in self.config.SYMBOLS[:20]:  
            try:  
                if self.api_failures[symbol] >= TradingConfig.CIRCUIT_BREAKER_MAX_FAILS:  
                    rejected += 1  
                    self.logger.info(f"[SCAN] {symbol} REJECTED: CIRCUIT_BREAKER")  
                    continue  
                  
                ohlcv = self._retry_api(self.exchange.fetch_ohlcv, symbol, '15m', limit=100)  
                if len(ohlcv) < 50:  
                    rejected += 1  
                    self.logger.info(f"[SCAN] {symbol} REJECTED: DATA_INSUFFICIENT")  
                    continue  
                  
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])  
                  
                score = self.calculate_score(df)  
                  
                is_tradable, market_analysis = self.market_filter.analyze_market_regime(df)  
                  
                analysis = MarketAnalysis(  
                    symbol=symbol,  
                    score=score,  
                    price=df['close'].iloc[-1],  
                    atr_percent=market_analysis.get('atr_percent', 0),  
                    ema_slope=market_analysis.get('ema_slope', 0),  
                    is_sideways=market_analysis.get('is_sideways', False),  
                    correlation_group=self.correlation_filter.get_symbol_group(symbol),  
                    last_ohlcv=df  
                )  
                  
                analysis.avg_volume = df['volume'].rolling(20).mean().iloc[-1] * analysis.price  
                  
                ticker = self._retry_api(self.exchange.fetch_ticker, symbol)  
                bid = ticker['bid']  
                ask = ticker['ask']  
                if bid and ask:  
                    mid = (bid + ask) / 2  
                    analysis.spread = (ask - bid) / mid  
                else:  
                    analysis.spread = 0.0  
                  
                scan_results[symbol] = analysis  
                  
                scanned += 1  
                time.sleep(self.config.API_DELAY)  
                  
            except Exception as e:  
                rejected += 1  
                self.logger.info(f"[SCAN] {symbol} REJECTED: API_FAILURE - {e}")  
          
        self.logger.info(f"[SUMMARY] Scanned: {scanned + rejected}, Rejected: {rejected}, Eligible: {scanned}, Executed: 0")  
        return scan_results  
  
    def filter_symbols(self, scan_results: Dict[str, MarketAnalysis]) -> Dict[str, MarketAnalysis]:  
        filtered_results = {}  
          
        for symbol, analysis in scan_results.items():  
            reason = "OK"  
            if analysis.score < self.config.MIN_SCORE:  
                reason = "SCORE_TOO_LOW"  
            else:  
                can_trade, corr_reason = self.correlation_filter.can_trade_symbol(symbol, self.active_trades)  
                if not can_trade:  
                    reason = "CORRELATION_BLOCK"  
              
            if reason == "OK":  
                risk_params = self.risk_manager.get_risk_parameters(analysis.score)  
                if not risk_params.get('can_trade', False):  
                    reason = "SCORE_TOO_LOW"  
              
            if reason == "OK":  
                filtered_results[symbol] = analysis  
            self.logger.info(f"[TRACE] {symbol} → {reason}")  
          
        self.logger.info(f"✅ بقي بعد الفلترة {len(filtered_results)} عملة")  
        return filtered_results  
  
    def rank_symbols(self, filtered_results: Dict[str, MarketAnalysis]) -> List[Tuple[str, MarketAnalysis]]:  
        ranking_list = []  
          
        for symbol, analysis in filtered_results.items():  
            ranking_score = analysis.score + (analysis.atr_percent * 100)  
            ranking_list.append((symbol, analysis, ranking_score))  
          
        ranked_items = sorted(  
            ranking_list,  
            key=lambda x: (x[2], x[1].atr_percent),  
            reverse=True  
        )  
          
        ranked_symbols = []  
        for i, (symbol, analysis, _) in enumerate(ranked_items):  
            analysis.ranking = i + 1  
            ranked_symbols.append((symbol, analysis))  
          
        if ranked_symbols:  
            top_3 = ranked_symbols[:3]  
            self.logger.info(f"📊 أفضل 3 عملات:")  
            for symbol, analysis in top_3:  
                self.logger.info(f"  #{analysis.ranking} {symbol}: Score={analysis.score:.1f}, ATR%={analysis.atr_percent:.4f}")  
          
        return ranked_symbols  
  
    def _can_open_trade(self) -> bool:  
        with self.state_lock:  
            if self.is_halted:  
                return False  
              
            if not self._check_total_drawdown():  
                return False  
              
            if len(self.active_trades) >= self.config.MAX_OPEN_TRADES:  
                return False  
              
            trade_amount = (self.live_total_balance if self.trading_mode == TradingMode.LIVE else self.paper_total) * self.config.MAX_CAPITAL_PER_TRADE  
            available = self.live_available_balance if self.trading_mode == TradingMode.LIVE else self.paper_available  
            if trade_amount > available:  
                return False  
              
            daily_limit = self.config.INITIAL_CAPITAL * self.config.MAX_DAILY_LOSS  
            if self.daily_pnl < -daily_limit:  
                return False  
              
            return True  
  
    def execute_trades(self, ranked_symbols: List[Tuple[str, MarketAnalysis]]):  
        with self.state_lock:  
            if self.is_halted:  
                self.logger.warning("تم إيقاف التداول – global halt مفعل")  
                return  
              
            current_trades_count = len(self.active_trades)  
            available_slots = self.config.MAX_OPEN_TRADES - current_trades_count  
              
            if available_slots <= 0:  
                return  
              
            trades_opened = 0  
              
            for symbol, analysis in ranked_symbols:  
                reason = "OK"  
                  
                if trades_opened >= available_slots:  
                    break  
                  
                if any(t.symbol == symbol for t in self.active_trades.values()):  
                    continue  
                  
                if analysis.avg_volume < TradingConfig.MIN_LIQUIDITY_USDT:  
                    reason = "LOW_LIQUIDITY"  
                elif analysis.spread > TradingConfig.MAX_SPREAD_PERCENT:  
                    reason = "HIGH_SPREAD"  
                elif analysis.is_sideways or analysis.atr_percent < self.config.MIN_ATR_PERCENT:  
                    reason = "SIZE_REDUCTION"  
                elif not self._check_cooling_off():  
                    reason = "COOLING_OFF"  
                  
                if reason == "COOLING_OFF":  
                    self.logger.info(f"[TRACE] {symbol} → COOLING_OFF")  
                    continue  
                  
                ai_confidence = self._get_ai_confidence(analysis.last_ohlcv)  
                size_multiplier = ai_confidence  
                  
                if reason in ["LOW_LIQUIDITY", "HIGH_SPREAD"]:  
                    size_multiplier *= 0.5  
                elif reason == "SIZE_REDUCTION":  
                    size_multiplier *= 0.7  
                  
                existing_trade = next((t for t in self.active_trades.values() if t.symbol == symbol), None)  
                if existing_trade:  
                    if analysis.score - existing_trade.score >= TradingConfig.PROGRESSIVE_ENTRY_BOOST_THRESHOLD:  
                        boost_usdt = (self.live_total_balance if self.trading_mode == TradingMode.LIVE else self.paper_total) * self.config.MAX_CAPITAL_PER_TRADE * (1 - TradingConfig.PROGRESSIVE_ENTRY_INITIAL_PERCENT)  
                        boost_quantity = boost_usdt / analysis.price  
                        execution, error = self.execution_manager.execute_entry(symbol, boost_quantity, analysis.price)  
                        if error:  
                            self.logger.error(f"فشل تعزيز {symbol}: {error}")  
                            continue  
                          
                        existing_trade.quantity += execution.entry_order.filled  
                        existing_trade.entry_price = (existing_trade.entry_price * (existing_trade.quantity - execution.entry_order.filled) + execution.net_price_entry * execution.entry_order.filled) / existing_trade.quantity  
                        existing_trade.entry_amount_percent = 1.0  
                        existing_trade.score = analysis.score  
                        self._save_trade(existing_trade)  
                        self._save_execution(execution)  
                        self.monitor.update_stats(existing_trade, execution)  
                        self.logger.info(f"تعزيز صفقة {symbol} (ADD-ON / Position Scaling)")  
                        trades_opened += 1  
                        self.logger.info(f"[TRACE] {symbol} → EXECUTED")  
                        continue  
                  
                entry_usdt = (self.live_total_balance if self.trading_mode == TradingMode.LIVE else self.paper_total) * self.config.MAX_CAPITAL_PER_TRADE * TradingConfig.PROGRESSIVE_ENTRY_INITIAL_PERCENT * size_multiplier  
                quantity = entry_usdt / analysis.price  
                  
                execution, error = self.execution_manager.execute_entry(symbol, quantity, analysis.price)  
                  
                if error:  
                    self.logger.error(f"فشل تنفيذ الدخول لـ {symbol}: {error}")  
                    continue  
                  
                trade_id = execution.trade_id  
                
                risk_params = self.risk_manager.get_risk_parameters(analysis.score)  
                risk_level = risk_params.get('level', 'MEDIUM')  
                  
                stop_loss = self.risk_manager.calculate_stop_loss(analysis.price, risk_params)  
                take_profit = analysis.price * (1 + self.config.TAKE_PROFIT_PERCENT)  
                  
                trade = TradeRecord(  
                    trade_id=trade_id,  
                    symbol=symbol,  
                    entry_price=execution.net_price_entry,  
                    entry_time=datetime.now(timezone.utc).isoformat(),  
                    quantity=execution.entry_order.filled,  
                    stop_loss=stop_loss,  
                    take_profit=take_profit,  
                    original_take_profit=take_profit,  
                    phase="ENTRY",  
                    highest_price=execution.net_price_entry,  
                    score=analysis.score,  
                    risk_level=risk_level,  
                    execution_id=execution.trade_id,  
                    entry_amount_percent=TradingConfig.PROGRESSIVE_ENTRY_INITIAL_PERCENT  
                )  
                  
                trade_cost = execution.net_price_entry * execution.entry_order.filled  
                if self.trading_mode == TradingMode.LIVE:  
                    self.live_available_balance -= trade_cost  
                else:  
                    self.paper_available -= trade_cost  
                  
                if self.trading_mode == TradingMode.LIVE:  
                    success, msg = self.execution_manager.create_exchange_orders(trade)  
                    if not success:  
                        self.logger.warning(f"فشل إنشاء أوامر المنصة: {msg}")  
                  
                self.active_trades[trade_id] = trade  
                self._save_trade(trade)  
                self._save_execution(execution)  
                self.monitor.update_stats(trade, execution)  
                  
                self._send_notification(  
                    f"🚀 *صفقة جديدة [{risk_level}]*\\n"  
                    f"• العملة: `{symbol}`\\n"  
                    f"• السعر: `${execution.net_price_entry:.4f}`\\n"  
                    f"• الكمية: `{execution.entry_order.filled:.6f}`\\n"  
                    f"• التكلفة: `${trade_cost:.2f}`\\n"  
                    f"• السكور: `{analysis.score:.1f}`\\n"  
                    f"• Stop-Loss: `${stop_loss:.4f}`\\n"  
                    f"• Take-Profit: `${take_profit:.4f}`"  
                )  
                  
                self.previous_scores[symbol] = analysis.score  
                trades_opened += 1  
                self.logger.info(f"[TRACE] {symbol} → {reason if reason != 'OK' else 'EXECUTED'}")  
          
        if trades_opened > 0:  
            self.logger.info(f"✅ تم فتح {trades_opened}/{available_slots} صفقة جديدة")  
  
    def manage_trades(self):  
        if not self.active_trades:  
            return  
          
        current_prices = {}  
        for trade in self.active_trades.values():  
            try:  
                ticker = self._retry_api(self.exchange.fetch_ticker, trade.symbol)  
                current_prices[trade.symbol] = ticker['last']  
            except Exception as e:  
                self.logger.error(f"خطأ في جلب سعر {trade.symbol}: {e}")  
                continue  
          
        if not current_prices:  
            return  
          
        with self.state_lock:  
            trades_to_close = []  
              
            for trade_id, trade in list(self.active_trades.items()):  
                price = current_prices.get(trade.symbol)  
                if not price:  
                    continue  
                  
                if price > trade.highest_price:  
                    trade.highest_price = price  
                  
                if self._check_max_trade_age(trade):  
                    execution, error = self.execution_manager.execute_exit(  
                        trade.symbol, trade.quantity, trade.trade_id  
                    )  
                      
                    if error:  
                        self.logger.error(f"❌ فشل إغلاق {trade.symbol} بسبب MAX_AGE: {error}")  
                        continue  
                      
                    trade.exit_price = execution.net_price_exit  
                    trade.exit_time = datetime.now(timezone.utc).isoformat()  
                    trade.exit_reason = "MAX_AGE_EXIT"  
                    trade.status = "CLOSED"  
                      
                    if execution.net_price_exit and execution.net_price_entry:  
                        trade.pnl = (execution.net_price_exit - execution.net_price_entry) * trade.quantity  
                      
                    if self.trading_mode == TradingMode.PAPER:  
                        if trade.pnl:  
                            self.paper_total += trade.pnl  
                            self.paper_available += (trade.entry_price * trade.quantity)  
                        self.daily_pnl += trade.pnl  
                        self._update_loss_streak(trade.pnl)  
                      
                    if self.trading_mode == TradingMode.LIVE:  
                        self.execution_manager.cleanup_exchange_orders(trade)  
                      
                    self._save_trade(trade)  
                    self._save_execution(execution)  
                    self._move_to_closed(trade_id)  
                    self.monitor.update_stats(trade, execution)  
                      
                    status = "✅" if trade.pnl > 0 else "❌"  
                    pnl_amount = trade.pnl if trade.pnl else 0  
                      
                    self._send_notification(  
                        f"{status} *إغلاق صفقة*\\n"  
                        f"• العملة: `{trade.symbol}`\\n"  
                        f"• الدخول: `${trade.entry_price:.4f}`\\n"  
                        f"• الخروج: `${execution.net_price_exit:.4f}`\\n"  
                        f"• الكمية: `{trade.quantity:.6f}`\\n"  
                        f"• P&L: `${pnl_amount:.4f}`\\n"  
                        f"• الرسوم: `${execution.total_fees:.4f}`\\n"  
                        f"• السبب: `MAX_AGE_EXIT`"  
                    )  
                      
                    self.logger.info(f"{status} أغلقت صفقة {trade_id}: ${pnl_amount:.4f} (MAX_AGE)")  
                    trades_to_close.append(trade_id)  
                    continue  
                  
                platform_exit_reason = None  
                if self.trading_mode == TradingMode.LIVE:  
                    platform_exit_reason = self.execution_manager.check_exchange_order_status(trade)  
                  
                if platform_exit_reason:  
                    execution, error = self.execution_manager.execute_exit(  
                        trade.symbol, trade.quantity, trade.trade_id  
                    )  
                      
                    if error:  
                        self.logger.error(f"❌ فشل تنفيذ الخروج لـ {trade.symbol}: {error}")  
                        continue  
                      
                    trade.exit_price = execution.net_price_exit  
                    trade.exit_time = datetime.now(timezone.utc).isoformat()  
                    trade.exit_reason = platform_exit_reason  
                    trade.status = "CLOSED"  
                      
                    if execution.net_price_exit and execution.net_price_entry:  
                        trade.pnl = (execution.net_price_exit - execution.net_price_entry) * trade.quantity  
                      
                    if self.trading_mode == TradingMode.PAPER:  
                        if trade.pnl:  
                            self.paper_total += trade.pnl  
                            self.paper_available += (trade.entry_price * trade.quantity)  
                        self.daily_pnl += trade.pnl  
                        self._update_loss_streak(trade.pnl)  
                      
                    if self.trading_mode == TradingMode.LIVE:  
                        self.execution_manager.cleanup_exchange_orders(trade)  
                      
                    self._save_trade(trade)  
                    self._save_execution(execution)  
                    self._move_to_closed(trade_id)  
                    self.monitor.update_stats(trade, execution)  
                      
                    status = "✅" if trade.pnl > 0 else "❌"  
                    pnl_amount = trade.pnl if trade.pnl else 0  
                      
                    self._send_notification(  
                        f"{status} *إغلاق صفقة (منصة)*\\n"  
                        f"• العملة: `{trade.symbol}`\\n"  
                        f"• الدخول: `${trade.entry_price:.4f}`\\n"  
                        f"• الخروج: `${execution.net_price_exit:.4f}`\\n"  
                        f"• الكمية: `{trade.quantity:.6f}`\\n"  
                        f"• P&L: `${pnl_amount:.4f}`\\n"  
                        f"• الرسوم: `${execution.total_fees:.4f}`\\n"  
                        f"• السبب: `{platform_exit_reason}`"  
                    )  
                      
                    self.logger.info(f"{status} أغلقت صفقة {trade_id} من المنصة: ${pnl_amount:.4f}")  
                    trades_to_close.append(trade_id)  
                    continue  
                  
                if (trade.phase == "ENTRY" and   
                    price >= trade.entry_price * (1 + self.config.BREAKEVEN_TRIGGER)):  
                    trade.stop_loss = trade.entry_price  
                    trade.phase = "BREAKEVEN"  
                    trade.stop_loss_modified = True  
                    self.logger.info(f"{trade_id} انتقل لنقطة التعادل")  
                  
                if (trade.phase == "BREAKEVEN" and   
                    price >= trade.entry_price * (1 + self.config.TRAILING_ACTIVATION)):  
                    trade.phase = "TRAILING"  
                    trade.take_profit = None  
                    trade.take_profit_modified = True  
                    self.logger.info(f"{trade_id} تفعيل التوقف المتحرك وإلغاء TP")  
                  
                if trade.phase == "TRAILING":  
                    trailing_distance = self.config.TRAILING_DISTANCE  
                    if trade.risk_level and self.config.ENABLE_DYNAMIC_RISK:  
                        risk_params = self.risk_manager.get_risk_parameters(trade.score or 0)  
                        trailing_distance = self.risk_manager.calculate_trailing_distance(risk_params)  
                          
                    new_stop = trade.highest_price * (1 - trailing_distance)  
                    # IMPORTANT: We NEVER lower stop_loss in trailing mode (safety)
                    if new_stop > trade.stop_loss:  
                        trade.stop_loss = new_stop  
                        trade.stop_loss_modified = True  
                  
                if trade.stop_loss_modified or trade.take_profit_modified:  
                    success, msg = self.execution_manager.update_exchange_orders(trade)  
                    if success:  
                        self.logger.info(f"🔄 تم تحديث أوامر المنصة للصفقة {trade_id}: {msg}")  
                    else:  
                        self.logger.warning(f"⚠️ فشل تحديث أوامر المنصة للصفقة {trade_id}: {msg}")  
                  
                exit_reason = None  
                  
                if price <= trade.stop_loss:  
                    exit_reason = "STOP_LOSS"  
                  
                elif (trade.take_profit is not None and   
                      trade.phase in ["ENTRY", "BREAKEVEN"] and   
                      price >= trade.take_profit):  
                    exit_reason = "TAKE_PROFIT"  
                  
                if exit_reason:  
                    execution, error = self.execution_manager.execute_exit(  
                        trade.symbol, trade.quantity, trade.trade_id  
                    )  
                      
                    if error:  
                        self.logger.error(f"❌ فشل تنفيذ الخروج لـ {trade.symbol}: {error}")  
                        continue  
                      
                    trade.exit_price = execution.net_price_exit  
                    trade.exit_time = datetime.now(timezone.utc).isoformat()  
                    trade.exit_reason = exit_reason  
                    trade.status = "CLOSED"  
                      
                    if execution.net_price_exit and execution.net_price_entry:  
                        trade.pnl = (execution.net_price_exit - execution.net_price_entry) * trade.quantity  
                      
                    if self.trading_mode == TradingMode.PAPER:  
                        if trade.pnl:  
                            self.paper_total += trade.pnl  
                            self.paper_available += (trade.entry_price * trade.quantity)  
                        self.daily_pnl += trade.pnl  
                        self._update_loss_streak(trade.pnl)  
                      
                    if self.trading_mode == TradingMode.LIVE:  
                        self.execution_manager.cleanup_exchange_orders(trade)  
                      
                    self._save_trade(trade)  
                    self._save_execution(execution)  
                    self._move_to_closed(trade_id)  
                    self.monitor.update_stats(trade, execution)  
                      
                    status = "✅" if trade.pnl > 0 else "❌"  
                    pnl_amount = trade.pnl if trade.pnl else 0  
                      
                    self._send_notification(  
                        f"{status} *إغلاق صفقة*\\n"  
                        f"• العملة: `{trade.symbol}`\\n"  
                        f"• الدخول: `${trade.entry_price:.4f}`\\n"  
                        f"• الخروج: `${execution.net_price_exit:.4f}`\\n"  
                        f"• الكمية: `{trade.quantity:.6f}`\\n"  
                        f"• P&L: `${pnl_amount:.4f}`\\n"  
                        f"• الرسوم: `${execution.total_fees:.4f}`\\n"  
                        f"• السبب: `{exit_reason}`"  
                    )  
                      
                    self.logger.info(f"{status} أغلقت صفقة {trade_id}: ${pnl_amount:.4f} ({exit_reason})")  
                    trades_to_close.append(trade_id)  
                    continue  
              
            for trade_id in trades_to_close:  
                del self.active_trades[trade_id]  
  
    def run(self):  
        self.logger.info("بدء تشغيل البوت الرئيسي...")  
        while True:  
            try:  
                if self.is_halted:  
                    self.logger.critical("البوت متوقف – global halt")  
                    time.sleep(300)  # 5 دقائق  
                    continue  
                  
                scan_results = self.scan_symbols()  
                filtered = self.filter_symbols(scan_results)  
                ranked = self.rank_symbols(filtered)  
                market_condition = self.bear_market_filter.analyze_market_condition(self.exchange)  
                ranked = self.bear_market_filter.apply_filter(ranked, market_condition)  
                ranked = self.ethical_filter.filter_symbols(ranked)  
                self.execute_trades(ranked)  
                self.manage_trades()  
                  
                self.monitor.update_platform_stats(self.exchange_interface.get_execution_stats())  
                  
                if self.trading_mode == TradingMode.LIVE:  
                    self._update_live_balance()  
                  
                time.sleep(self.config.SCAN_INTERVAL)  
            except Exception as e:  
                self.logger.error(f"خطأ في الدورة الرئيسية: {e}")  
                self.api_failure_count += 1  
                if self.api_failure_count >= self.max_api_failures_before_halt:  
                    self.is_halted = True  
                    self._send_notification("🚨 GLOBAL HALT ACTIVATED – critical loop failure")  
                time.sleep(60)  

if __name__ == "__main__":
    bot = StableBotPro(TradingMode.PAPER)  
    bot.run()
