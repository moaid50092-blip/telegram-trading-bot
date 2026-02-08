#!/usr/bin/env python3
"""
StableBotPro v4.1 - نظام التداول الآلي المتكامل مع Hardening
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
        # "TRX/USDT"  تم حذفه لحل التعارض مع BLACKLIST في EthicalFilter
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
    
    # ... (باقي الإعدادات كما هي بدون تغيير)

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

    # ... (باقي الكلاس كما هو)

# باقي الـ dataclasses بدون تغيير
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

# ... (باقي الـ dataclasses كما هي)

class EnhancedExchangeInterface:
    # ...

    def format_price(self, symbol: str, price: float) -> float:
        try:
            market = self.exchange.market(symbol)
            # ──────────────── إصلاح حرج ────────────────
            # استخدام tickSize الحقيقي من PRICE_FILTER بدلاً من precision.price
            for f in market.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    tick_size = float(f['tickSize'])
                    return tick_size * round(price / tick_size)
            # fallback إذا لم يتم العثور على الفلتر
            precision = market['precision'].get('price', 8)
            return round(price, precision)
        except Exception:
            return price

    # ... (باقي الكلاس بدون تغيير)

class ExecutionManager:
    # ...

    def execute_exit(self, symbol: str, amount: float, trade_id: str = None, price: float = None, order_type: OrderType = None) -> Tuple[Optional[TradeExecution], Optional[str]]:
        # ──────────────── إصلاح حرج ────────────────
        # تم تغيير المعامل من execution_id إلى trade_id
        # واستخدامه كمفتاح في القاموس
        execution = None
        if trade_id and trade_id in self.trade_executions:
            execution = self.trade_executions[trade_id]
            symbol = execution.symbol
            amount = execution.entry_order.filled

        # ... (باقي الدالة كما هي)

    # ... (باقي الكلاس)

class StableBotPro:
    def __init__(self, trading_mode: TradingMode = TradingMode.PAPER):
        # ... (الباقي كما هو)

    def _generate_trade_id(self) -> str:
        # ──────────────── إصلاح حرج #1 ────────────────
        # مولد trade_id موحد وآمن جدًا
        # يستخدم timestamp بالنانو ثانية + جزء عشوائي قوي
        ns = time.time_ns()
        rand_part = random.randint(100000, 999999)
        return f"TRD-{ns}-{rand_part}"

    def _sync_live_balance(self) -> Tuple[bool, str]:
        # ... (كما هو)

    def _update_live_balance(self) -> bool:
        # ... (كما هو)

    def execute_trades(self, ranked_symbols: List[Tuple[str, MarketAnalysis]]):
        with self.lock:
            # ... (البداية كما هي)

            for symbol, analysis in ranked_symbols:
                # ...

                # ──────────────── إصلاح حرج #1 ────────────────
                # استخدام المولد الموحد بدلاً من الإنشاء اليدوي
                trade_id = self._generate_trade_id()

                # ... (باقي الكود حتى إنشاء TradeRecord)

                trade = TradeRecord(
                    trade_id=trade_id,   # ← المولد الجديد
                    # ...
                )

                # ... (باقي الدالة)

    def manage_trades(self):
        if not self.active_trades:
            return

        # ... (جلب الأسعار)

        with self.lock:
            trades_to_close = []

            for trade_id, trade in list(self.active_trades.items()):
                # ...

                platform_exit_reason = None
                if self.trading_mode == TradingMode.LIVE:
                    platform_exit_reason = self.execution_manager.check_exchange_order_status(trade)

                if platform_exit_reason:
                    # ──────────────── إصلاح حرج #2 ────────────────
                    # تمرير trade.trade_id بدلاً من trade.execution_id
                    execution, error = self.execution_manager.execute_exit(
                        trade.symbol,
                        trade.quantity,
                        trade.trade_id   # ← التعديل الحاسم
                    )
                    # ... (باقي المنطق)

                # ... (باقي الحالات: max age, stop loss, take profit)

                if exit_reason:
                    # ──────────────── إصلاح حرج #2 ────────────────
                    # نفس التعديل هنا أيضًا
                    execution, error = self.execution_manager.execute_exit(
                        trade.symbol,
                        trade.quantity,
                        trade.trade_id   # ← التعديل الحاسم
                    )
                    if error:
                        # ...

                    # ──────────────── إصلاح حرج #3 ────────────────
                    # حماية رأس المال في وضع LIVE
                    if self.trading_mode == TradingMode.PAPER:
                        if trade.pnl:
                            self.current_capital += trade.pnl
                            self.available_capital += (trade.entry_price * trade.quantity)
                    # في LIVE → نعتمد فقط على _update_live_balance() الدوري

                    # ... (باقي منطق الإغلاق والإشعار)

            # ... (حذف الصفقات المغلقة)

    def run(self):
        self.logger.info("بدء تشغيل البوت الرئيسي...")
        while True:
            try:
                # ... (الدورة كما هي)

                # يُفضل إضافة استدعاء دوري خفيف هنا (اختياري ولكن مفيد)
                if self.trading_mode == TradingMode.LIVE:
                    self._update_live_balance()

                time.sleep(self.config.SCAN_INTERVAL)
            except Exception as e:
                self.logger.error(f"خطأ في الدورة الرئيسية: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = StableBotPro(TradingMode.PAPER)  
    bot.run()
