import os
import time
import ccxt  
import pandas as pd
import numpy as np
import requests
import threading
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# تهيئة البيئة
load_dotenv()

# ==================== الإعدادات الثابتة ====================
class TradingConfig:
    # القائمة الذهبية المقترحة (أفضل 12 عملة سيولة وتحليلاً)
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", 
        "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT",
        "LINK/USDT", "NEAR/USDT", "MATIC/USDT", "LTC/USDT"
    ]
    
    # إدارة رأس المال
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1000))
    MAX_CAPITAL_PER_TRADE = 0.10
    MAX_OPEN_TRADES = 3
    
    # وقف الخسارة وجني الأرباح (الاستراتيجية الأصلية كما هي)
    STOP_LOSS_PERCENT = 0.02
    TAKE_PROFIT_PERCENT = 0.04
    MAX_DAILY_LOSS = 0.05
    
    # نظام المراحل
    BREAKEVEN_TRIGGER = 0.012
    TRAILING_ACTIVATION = 0.03
    TRAILING_DISTANCE = 0.01
    
    # وضع التداول (افتراضياً تجريبي للأمان)
    # للتحويل لحقيقي مستقبلاً، يتم تغيير القيمة في ملف .env إلى REAL
    TRADE_MODE = os.getenv('TRADE_MODE', 'PAPER') 
    
    # التواصل
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # إعدادات الأداء
    MIN_SCORE = 45
    SCAN_INTERVAL = 180  
    API_DELAY = 0.5

# ==================== هياكل البيانات ====================
@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    entry_price: float
    entry_time: str
    quantity: float
    stop_loss: float
    take_profit: float
    phase: str
    status: str = "ACTIVE"
    highest_price: float = 0.0
    score: Optional[float] = None
    mode: str = "PAPER"

# ==================== البوت الرئيسي ====================
class StableBotPro:
    def __init__(self):
        self.config = TradingConfig
        
        # ربط الـ API (للقراءة حالياً في وضع PAPER)
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.active_trades: Dict[str, TradeRecord] = {}
        self.lock = threading.Lock()
        self.daily_pnl = 0.0
        self.current_capital = self.config.INITIAL_CAPITAL
        self.available_capital = self.config.INITIAL_CAPITAL
        
        for path in ['logs', 'data/active_trades', 'data/closed_trades']:
            if not os.path.exists(path):
                os.makedirs(path)
        
        self._setup_logging()
        self._load_active_trades()
        
        mode_text = "حقيقي ⚠️" if self.config.TRADE_MODE == 'REAL' else "تجريبي (آمن) ✅"
        self.logger.info(f"🚀 بدء StableBotPro | الوضع: {mode_text} | رأس مال: ${self.current_capital}")
    
    def _setup_logging(self):
        log_formatter = logging.Formatter('%(asctime)s UTC - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler = RotatingFileHandler('logs/trading.log', maxBytes=10*1024*1024, backupCount=5)
        handler.setFormatter(log_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger = logging.getLogger('StableBot')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console_handler)

    def _load_active_trades(self):
        path = 'data/active_trades'
        if not os.path.exists(path): return
        loaded = 0
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                try:
                    with open(f"{path}/{filename}", 'r') as f:
                        data = json.load(f)
                        trade = TradeRecord(**data)
                        self.active_trades[trade.trade_id] = trade
                        loaded += 1
                except Exception as e:
                    self.logger.error(f"خطأ تحميل: {e}")
        self.logger.info(f"تم تحميل {loaded} صفقة من الذاكرة")

    def _save_trade(self, trade: TradeRecord):
        try:
            filename = f"data/active_trades/{trade.trade_id}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(asdict(trade), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"خطأ حفظ: {e}")

    def _move_to_closed(self, trade_id: str):
        try:
            src = f"data/active_trades/{trade_id}.json"
            dst = f"data/closed_trades/{trade_id}.json"
            if os.path.exists(src): os.rename(src, dst)
        except Exception as e:
            self.logger.error(f"خطأ نقل: {e}")

    def _send_notification(self, message: str):
        if not self.config.TELEGRAM_TOKEN or not self.config.TELEGRAM_CHAT_ID: return
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": self.config.TELEGRAM_CHAT_ID, "text": f"🤖 *StableBot:*\n{message}", "parse_mode": "Markdown"}
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            self.logger.error(f"خطأ تلغرام: {e}")

    def calculate_score(self, df: pd.DataFrame) -> float:
        """استراتيجية السكور الأصلية (بدون تعديل)"""
        try:
            if len(df) < 30: return 0
            close = df['close']
            net_move = abs(close.iloc[-1] - close.iloc[-10])
            total_path = close.diff().abs().iloc[-10:].sum()
            efficiency = (net_move / total_path * 40) if total_path > 0 else 0
            sma_20 = close.rolling(20).mean().iloc[-1]
            trend = 20 if close.iloc[-1] > sma_20 else 5
            last_candle = df.iloc[-1]
            candle_range = last_candle['high'] - last_candle['low']
            penalty = 0
            if candle_range > 0:
                upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                penalty = (upper_wick / candle_range) * 30
            volume_avg = df['vol'].iloc[-10:].mean()
            volume_score = 10 if df['vol'].iloc[-1] > volume_avg else 5
            score = efficiency + trend + volume_score - penalty
            return max(0, min(100, round(score, 2)))
        except Exception as e:
            self.logger.error(f"خطأ سكور: {e}"); return 0

    def _can_open_trade(self) -> bool:
        with self.lock:
            if len(self.active_trades) >= self.config.MAX_OPEN_TRADES: return False
            trade_amount = self.current_capital * self.config.MAX_CAPITAL_PER_TRADE
            if trade_amount > self.available_capital: return False
            if self.daily_pnl < -(self.config.INITIAL_CAPITAL * self.config.MAX_DAILY_LOSS): return False
            return True

    def open_trade(self, symbol: str, price: float, score: float):
        if not self._can_open_trade(): return
        with self.lock:
            trade_id = f"T{int(time.time())}_{symbol.replace('/', '')}"
            trade_amount = self.current_capital * self.config.MAX_CAPITAL_PER_TRADE
            quantity = trade_amount / price
            
            trade = TradeRecord(
                trade_id=trade_id, symbol=symbol, entry_price=price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                quantity=quantity, stop_loss=price * (1 - self.config.STOP_LOSS_PERCENT),
                take_profit=price * (1 + self.config.TAKE_PROFIT_PERCENT),
                phase="ENTRY", highest_price=price, score=score, mode=self.config.TRADE_MODE
            )
            
            self.available_capital -= trade_amount
            self.active_trades[trade_id] = trade
            self._save_trade(trade)
            
            mode_tag = "⚠️ حقيقي" if self.config.TRADE_MODE == 'REAL' else "🧪 تجريبي"
            self._send_notification(f"🚀 *صفقة جديدة ({mode_tag})*\n• العملة: `{symbol}`\n• السعر: `${price:.4f}`\n• السكور: `{score:.1f}`")
            self.logger.info(f"✅ فتح صفقة {trade_id}")

    def manage_trades(self, current_prices: Dict[str, float]):
        with self.lock:
            trades_to_close = []
            for trade_id, trade in list(self.active_trades.items()):
                price = current_prices.get(trade.symbol)
                if not price: continue
                if price > trade.highest_price: trade.highest_price = price

                if trade.phase == "ENTRY" and price >= trade.entry_price * (1 + self.config.BREAKEVEN_TRIGGER):
                    trade.stop_loss = trade.entry_price
                    trade.phase = "BREAKEVEN"
                elif trade.phase == "BREAKEVEN" and price >= trade.entry_price * (1 + self.config.TRAILING_ACTIVATION):
                    trade.phase = "TRAILING"
                
                if trade.phase == "TRAILING":
                    new_stop = trade.highest_price * (1 - self.config.TRAILING_DISTANCE)
                    if new_stop > trade.stop_loss: trade.stop_loss = new_stop

                exit_reason = None
                if price <= trade.stop_loss: exit_reason = "STOP_LOSS"
                elif trade.phase in ["ENTRY", "BREAKEVEN"] and price >= trade.take_profit: exit_reason = "TAKE_PROFIT"

                if exit_reason:
                    pnl = (price - trade.entry_price) * trade.quantity
                    self.current_capital += pnl
                    self.available_capital += (trade.entry_price * trade.quantity)
                    self.daily_pnl += pnl
                    trade.status = "CLOSED"; trade.exit_price = price
                    self._save_trade(trade); self._move_to_closed(trade_id)
                    
                    status_icon = "✅" if pnl > 0 else "❌"
                    self._send_notification(f"{status_icon} *إغلاق صفقة*\n• العملة: `{trade.symbol}`\n• الربح/الخسارة: `${pnl:.2f}`\n• السبب: `{exit_reason}`")
                    trades_to_close.append(trade_id)
                else:
                    self._save_trade(trade)
            for tid in trades_to_close:
                if tid in self.active_trades: del self.active_trades[tid]

    def run_cycle(self):
        if datetime.now(timezone.utc).hour in self.config.AVOID_HOURS and not self.active_trades: return
        prices = {}
        for symbol in self.config.SYMBOLS:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=40)
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                price = df['close'].iloc[-1]
                prices[symbol] = price
                if len(self.active_trades) < self.config.MAX_OPEN_TRADES:
                    if not any(t.symbol == symbol for t in self.active_trades.values()):
                        score = self.calculate_score(df)
                        if score >= self.config.MIN_SCORE: self.open_trade(symbol, price, score)
                time.sleep(self.config.API_DELAY)
            except Exception as e:
                self.logger.error(f"خطأ {symbol}: {e}")
        if self.active_trades: self.manage_trades(prices)

    def run(self):
        while True:
            try:
                self.run_cycle()
                time.sleep(self.config.SCAN_INTERVAL)
            except Exception as e:
                self.logger.error(f"🚨 خطأ رئيسي: {e}"); time.sleep(60)

# ==================== واجهة التلغرام ====================
class TelegramInterface:
    def __init__(self, bot: StableBotPro):
        self.bot = bot
        self.token = TradingConfig.TELEGRAM_TOKEN
    
    def start_polling(self):
        offset = 0
        while True:
            try:
                url = f"https://api.telegram.org/bot{self.token}/getUpdates"
                response = requests.get(url, params={"offset": offset, "timeout": 30}, timeout=35)
                if response.status_code == 200:
                    for update in response.json().get("result", []):
                        offset = update["update_id"] + 1
                        if "message" in update: self.handle_command(update["message"])
                time.sleep(1)
            except: time.sleep(10)
    
    def handle_command(self, message: dict):
        text = message.get("text", "").strip()
        chat_id = message["chat"]["id"]
        if text == "/status":
            mode = "⚠️ حقيقي" if self.bot.config.TRADE_MODE == 'REAL' else "🧪 تجريبي"
            msg = f"📊 *حالة النظام ({mode})*\n• رأس المال: `${self.bot.current_capital:.2f}`\n• الصفقات: `{len(self.bot.active_trades)}`\n• ربح اليوم: `${self.bot.daily_pnl:.2f}`"
            self.send_message(chat_id, msg)
        elif text == "/trades":
            if not self.bot.active_trades: self.send_message(chat_id, "📭 لا توجد صفقات")
            else:
                msg = "📋 *الصفقات النشطة:*\n"
                for t in self.bot.active_trades.values():
                    msg += f"• `{t.symbol}` | سكور: `{t.score}` | مرحلة: `{t.phase}`\n"
                self.send_message(chat_id, msg)

    def send_message(self, chat_id, text):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

if __name__ == "__main__":
    bot = StableBotPro()
    threading.Thread(target=TelegramInterface(bot).start_polling, daemon=True).start()
    bot.run()
