import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
import winsound
import requests
from pathlib import Path
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Global configuration"""
    # Telegram Settings (Optional)
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    TELEGRAM_ENABLED = False  # Set to True to enable
    
    # Alert Settings
    SOUND_ENABLED = True
    POPUP_ENABLED = False
    
    # Trading Log
    LOG_FILE = "trade_signals_log.csv"
    
    # HTF Bias Lock - DISABLED FOR MORE SIGNALS
    HTF_LOCK_ENABLED = False  # Changed to False for more flexibility
    HTF_TIMEFRAME = "H1"
    
    # Signal Strictness (NEW)
    REQUIRE_ALL_CONDITIONS = False  # If False, signals trigger with partial conditions
    MIN_CONDITIONS_MET = 3  # Minimum conditions needed (out of 6)
    
    # Session Times (UTC)
    SESSIONS = {
        'Asian': {'start': 0, 'end': 9},
        'London': {'start': 7, 'end': 16},
        'NewYork': {'start': 12, 'end': 21}
    }
    
    # Trading Settings
    SCALP_MODE = True  # Enable quick scalping signals
    SWING_MODE = True  # Enable swing trading signals

# ============================================================================
# ALERT & LOGGING SYSTEM
# ============================================================================

class AlertSystem:
    """Handle alerts and notifications"""
    
    @staticmethod
    def play_sound(signal_type):
        """Play sound alert"""
        if not Config.SOUND_ENABLED:
            return
        
        try:
            if signal_type == "BUY":
                frequency = 1000
                duration = 200
                winsound.Beep(frequency, duration)
            elif signal_type == "SELL":
                frequency = 600
                duration = 200
                winsound.Beep(frequency, duration)
        except Exception as e:
            print(f"Sound alert error: {e}")
    
    @staticmethod
    def send_telegram(message):
        """Send Telegram notification"""
        if not Config.TELEGRAM_ENABLED or not Config.TELEGRAM_BOT_TOKEN:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": Config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    @staticmethod
    def show_popup(title, message):
        """Show popup notification"""
        if Config.POPUP_ENABLED:
            messagebox.showinfo(title, message)

class TradeLogger:
    """Log trade signals for backtesting"""
    
    def __init__(self, log_file=Config.LOG_FILE):
        self.log_file = Path(log_file)
        self.init_log_file()
    
    def init_log_file(self):
        """Create log file with headers if doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Timeframe', 'Symbol', 'Signal', 'Type', 'Session',
                    'HTF_Bias', 'Entry_Low', 'Entry_High', 'Stop_Loss',
                    'Take_Profit', 'Risk_Reward', 'Confidence', 'Conditions_Met', 'Notes'
                ])
    
    def log_signal(self, signal: 'TradeSignal'):
        """Log a trade signal"""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signal.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    signal.timeframe,
                    signal.symbol,
                    signal.signal,
                    signal.trade_type,
                    signal.session,
                    signal.htf_bias,
                    signal.entry_zone[0] if signal.entry_zone else 0,
                    signal.entry_zone[1] if signal.entry_zone else 0,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.risk_reward,
                    signal.confidence,
                    signal.conditions_met,
                    '; '.join(signal.reason)
                ])
            return True
        except Exception as e:
            print(f"Logging error: {e}")
            return False
    
    def get_stats(self):
        """Get basic statistics from log"""
        if not self.log_file.exists():
            return {}
        
        try:
            df = pd.read_csv(self.log_file)
            stats = {
                'total_signals': len(df),
                'buy_signals': len(df[df['Signal'] == 'BUY']),
                'sell_signals': len(df[df['Signal'] == 'SELL']),
                'scalp_signals': len(df[df['Type'] == 'SCALP']),
                'swing_signals': len(df[df['Type'] == 'SWING']),
                'avg_risk_reward': df['Risk_Reward'].mean(),
                'avg_confidence': df['Confidence'].mean()
            }
            return stats
        except Exception as e:
            print(f"Stats error: {e}")
            return {}

# ============================================================================
# SESSION DETECTION
# ============================================================================

class SessionDetector:
    """Detect trading sessions"""
    
    @staticmethod
    def get_current_session():
        """Get current trading session based on UTC time"""
        current_hour = datetime.utcnow().hour
        
        sessions = []
        for session_name, times in Config.SESSIONS.items():
            start = times['start']
            end = times['end']
            
            if start <= end:
                if start <= current_hour < end:
                    sessions.append(session_name)
            else:
                if current_hour >= start or current_hour < end:
                    sessions.append(session_name)
        
        if len(sessions) == 0:
            return "Pre-Market"
        elif len(sessions) == 1:
            return sessions[0]
        else:
            return f"{sessions[0]}/{sessions[1]}"
    
    @staticmethod
    def is_high_impact_session():
        """Check if currently in high-impact session"""
        session = SessionDetector.get_current_session()
        return 'London' in session or 'NewYork' in session

# ============================================================================
# DATA CLASSES
# ============================================================================

class TrendState(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    RANGE = "Range"

class StructureType(Enum):
    BOS = "BOS"
    CHOCH = "CHoCH"
    NONE = "None"

@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: str
    timestamp: datetime

@dataclass
class FVG:
    fvg_type: str
    top: float
    bottom: float
    midpoint: float
    start_index: int
    is_mitigated: bool
    mitigation_index: Optional[int] = None

@dataclass
class LiquidityZone:
    zone_type: str
    price: float
    count: int
    is_swept: bool
    sweep_index: Optional[int] = None

@dataclass
class TradeSignal:
    """Trade signal with full context"""
    timestamp: datetime
    timeframe: str
    symbol: str
    signal: str
    trade_type: str  # SCALP or SWING
    bias: str
    session: str
    htf_bias: str
    entry_zone: List[float]
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: int  # Percentage 0-100
    conditions_met: int  # How many conditions were met
    reason: List[str]
    confirmed: bool = False

# ============================================================================
# CORE ICT ANALYZER CLASS
# ============================================================================

class ICTMarketAnalyzer:
    def __init__(self):
        self.symbol = "XAUUSDm"
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1
        }
        self.current_trend_state = {}
        self.htf_bias = None
        self.last_signals = {}
        self.logger = TradeLogger()
        self.alert_system = AlertSystem()
        
    def connect_mt5(self):
        """Connect to MT5 terminal"""
        if not mt5.initialize():
            return False
        return True
    
    def get_candle_data(self, timeframe_str, num_candles=500):
        """Fetch OHLCV data from MT5"""
        if timeframe_str not in self.timeframes:
            return None
        
        timeframe = self.timeframes[timeframe_str]
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, num_candles)
        
        if rates is None or len(rates) == 0:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    # ========================================================================
    # SWING DETECTION
    # ========================================================================
    
    def detect_swings(self, df, swing_period=5, min_separation=5):
        """Detect swing highs and lows - LESS STRICT"""
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        
        last_high_idx = -min_separation - 1
        last_low_idx = -min_separation - 1
        
        for i in range(swing_period, len(df) - swing_period):
            # Swing high
            if i - last_high_idx >= min_separation:
                is_swing_high = True
                for j in range(i - swing_period, i + swing_period + 1):
                    if j != i and highs[j] >= highs[i]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_highs.append(SwingPoint(
                        index=i,
                        price=highs[i],
                        swing_type='high',
                        timestamp=df.iloc[i]['time']
                    ))
                    last_high_idx = i
            
            # Swing low
            if i - last_low_idx >= min_separation:
                is_swing_low = True
                for j in range(i - swing_period, i + swing_period + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append(SwingPoint(
                        index=i,
                        price=lows[i],
                        swing_type='low',
                        timestamp=df.iloc[i]['time']
                    ))
                    last_low_idx = i
        
        return swing_highs, swing_lows
    
    # ========================================================================
    # MARKET STRUCTURE - FLEXIBLE
    # ========================================================================
    
    def classify_structure(self, df, swing_highs: List[SwingPoint], 
                          swing_lows: List[SwingPoint], lookback=3):
        """Classify market structure - MORE FLEXIBLE"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendState.RANGE, "Insufficient swings"
        
        recent_highs = swing_highs[-min(lookback, len(swing_highs)):]
        recent_lows = swing_lows[-min(lookback, len(swing_lows)):]
        
        # Simple momentum check
        high_trend = 0
        if len(recent_highs) >= 2:
            if recent_highs[-1].price > recent_highs[-2].price:
                high_trend = 1
            else:
                high_trend = -1
        
        low_trend = 0
        if len(recent_lows) >= 2:
            if recent_lows[-1].price > recent_lows[-2].price:
                low_trend = 1
            else:
                low_trend = -1
        
        # Determine bias
        if high_trend > 0 and low_trend >= 0:
            bias = TrendState.BULLISH
            structure = "Higher Highs (Bullish)"
        elif high_trend <= 0 and low_trend < 0:
            bias = TrendState.BEARISH
            structure = "Lower Lows (Bearish)"
        elif high_trend > 0:
            bias = TrendState.BULLISH
            structure = "Bullish Momentum"
        elif low_trend < 0:
            bias = TrendState.BEARISH
            structure = "Bearish Momentum"
        else:
            bias = TrendState.RANGE
            structure = "Ranging/Consolidation"
        
        return bias, structure
    
    # ========================================================================
    # BOS & CHoCH
    # ========================================================================
    
    def detect_bos_choch(self, df, swing_highs: List[SwingPoint], 
                        swing_lows: List[SwingPoint], timeframe: str):
        """Detect BOS and CHoCH"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"type": StructureType.NONE, "index": None, "price": None}
        
        if timeframe not in self.current_trend_state:
            self.current_trend_state[timeframe] = TrendState.RANGE
        
        previous_trend = self.current_trend_state[timeframe]
        
        latest_high = swing_highs[-1]
        latest_low = swing_lows[-1]
        prev_high = swing_highs[-2]
        prev_low = swing_lows[-2]
        
        result = {"type": StructureType.NONE, "index": None, "price": None}
        
        # Check for breaks
        if latest_high.price > prev_high.price:
            if previous_trend in [TrendState.BULLISH, TrendState.RANGE]:
                result = {
                    "type": StructureType.BOS,
                    "index": latest_high.index,
                    "price": latest_high.price,
                    "direction": "Bullish"
                }
            else:
                result = {
                    "type": StructureType.CHOCH,
                    "index": latest_high.index,
                    "price": latest_high.price,
                    "direction": "Bullish"
                }
            self.current_trend_state[timeframe] = TrendState.BULLISH
        
        elif latest_low.price < prev_low.price:
            if previous_trend in [TrendState.BEARISH, TrendState.RANGE]:
                result = {
                    "type": StructureType.BOS,
                    "index": latest_low.index,
                    "price": latest_low.price,
                    "direction": "Bearish"
                }
            else:
                result = {
                    "type": StructureType.CHOCH,
                    "index": latest_low.index,
                    "price": latest_low.price,
                    "direction": "Bearish"
                }
            self.current_trend_state[timeframe] = TrendState.BEARISH
        
        return result
    
    # ========================================================================
    # PREMIUM/DISCOUNT
    # ========================================================================
    
    def calculate_premium_discount(self, df, swing_highs: List[SwingPoint], 
                                   swing_lows: List[SwingPoint], bias: TrendState):
        """Calculate premium/discount zones"""
        if len(swing_highs) == 0 or len(swing_lows) == 0:
            return "Unknown", None, None, None
        
        current_price = df.iloc[-1]['close']
        
        # Use recent range
        recent_high = max([s.price for s in swing_highs[-3:]])
        recent_low = min([s.price for s in swing_lows[-3:]])
        
        equilibrium = (recent_high + recent_low) / 2
        
        # More flexible zones
        upper_premium = equilibrium + (recent_high - equilibrium) * 0.5
        lower_discount = equilibrium - (equilibrium - recent_low) * 0.5
        
        if current_price > upper_premium:
            zone = "Premium"
        elif current_price < lower_discount:
            zone = "Discount"
        else:
            zone = "Fair Value"
        
        return zone, recent_high, recent_low, equilibrium
    
    # ========================================================================
    # LIQUIDITY DETECTION
    # ========================================================================
    
    def detect_liquidity(self, df, swing_highs: List[SwingPoint], 
                        swing_lows: List[SwingPoint], lookback=10):
        """Detect liquidity zones - MORE FLEXIBLE"""
        current_price = df.iloc[-1]['close']
        tolerance_pct = 0.001  # 0.1% tolerance (more flexible)
        
        liquidity_zones = {
            'BSL': [],
            'SSL': [],
            'BSL_swept': False,
            'SSL_swept': False,
            'BSL_near': False,
            'SSL_near': False,
            'active_bsl': None,
            'active_ssl': None
        }
        
        # Equal highs (BSL)
        recent_highs = swing_highs[-min(lookback, len(swing_highs)):]
        for i in range(len(recent_highs)):
            base_price = recent_highs[i].price
            
            # Check if swept
            is_swept = current_price > base_price * (1 + 0.0002)
            # Check if near
            is_near = abs(current_price - base_price) / base_price < 0.002
            
            zone = LiquidityZone(
                zone_type='BSL',
                price=base_price,
                count=1,
                is_swept=is_swept
            )
            liquidity_zones['BSL'].append(zone)
            
            if is_swept:
                liquidity_zones['BSL_swept'] = True
            if is_near:
                liquidity_zones['BSL_near'] = True
            if not is_swept and liquidity_zones['active_bsl'] is None:
                liquidity_zones['active_bsl'] = zone
        
        # Equal lows (SSL)
        recent_lows = swing_lows[-min(lookback, len(swing_lows)):]
        for i in range(len(recent_lows)):
            base_price = recent_lows[i].price
            
            is_swept = current_price < base_price * (1 - 0.0002)
            is_near = abs(current_price - base_price) / base_price < 0.002
            
            zone = LiquidityZone(
                zone_type='SSL',
                price=base_price,
                count=1,
                is_swept=is_swept
            )
            liquidity_zones['SSL'].append(zone)
            
            if is_swept:
                liquidity_zones['SSL_swept'] = True
            if is_near:
                liquidity_zones['SSL_near'] = True
            if not is_swept and liquidity_zones['active_ssl'] is None:
                liquidity_zones['active_ssl'] = zone
        
        return liquidity_zones
    
    # ========================================================================
    # FVG DETECTION
    # ========================================================================
    
    def detect_fvg(self, df, lookback=50):
        """Detect Fair Value Gaps"""
        fvgs = []
        
        for i in range(len(df) - 3 - lookback, len(df) - 3):
            if i < 0:
                continue
            
            candle1 = df.iloc[i]
            candle2 = df.iloc[i+1]
            candle3 = df.iloc[i+2]
            
            # Bullish FVG
            if candle1['high'] < candle3['low']:
                fvg_top = candle3['low']
                fvg_bottom = candle1['high']
                
                is_mitigated = False
                mitigation_idx = None
                
                for j in range(i + 3, len(df)):
                    if df.iloc[j]['low'] <= fvg_top:
                        is_mitigated = True
                        mitigation_idx = j
                        break
                
                fvgs.append(FVG(
                    fvg_type='Bullish',
                    top=fvg_top,
                    bottom=fvg_bottom,
                    midpoint=(fvg_top + fvg_bottom) / 2,
                    start_index=i,
                    is_mitigated=is_mitigated,
                    mitigation_index=mitigation_idx
                ))
            
            # Bearish FVG
            elif candle1['low'] > candle3['high']:
                fvg_top = candle1['low']
                fvg_bottom = candle3['high']
                
                is_mitigated = False
                mitigation_idx = None
                
                for j in range(i + 3, len(df)):
                    if df.iloc[j]['high'] >= fvg_bottom:
                        is_mitigated = True
                        mitigation_idx = j
                        break
                
                fvgs.append(FVG(
                    fvg_type='Bearish',
                    top=fvg_top,
                    bottom=fvg_bottom,
                    midpoint=(fvg_top + fvg_bottom) / 2,
                    start_index=i,
                    is_mitigated=is_mitigated,
                    mitigation_index=mitigation_idx
                ))
        
        return fvgs
    
    # ========================================================================
    # SIGNAL GENERATION - LESS STRICT, MORE SIGNALS
    # ========================================================================
    
    def is_price_near_fvg(self, current_price, fvg: FVG, tolerance=2.0):
        """Check if price is near or in FVG zone - MORE FLEXIBLE"""
        return (fvg.bottom - tolerance) <= current_price <= (fvg.top + tolerance)
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def generate_signal(self, analysis_result, df):
        """
        Generate trade signal - LESS STRICT, MORE PRACTICAL
        Can generate both SCALP and SWING signals
        """
        signal = "NO TRADE"
        trade_type = "NONE"
        entry_zone = [0, 0]
        stop_loss = 0
        take_profit = 0
        reason = []
        conditions_met = 0
        total_conditions = 6
        
        bias = analysis_result['bias']
        pd_zone = analysis_result['premium_discount']
        liquidity = analysis_result['liquidity']
        fvgs = analysis_result['fvgs']
        current_price = analysis_result['current_price']
        bos_choch = analysis_result['bos_choch']
        timeframe = analysis_result['timeframe']
        
        current_session = SessionDetector.get_current_session()
        is_high_impact = SessionDetector.is_high_impact_session()
        
        atr = self.calculate_atr(df)
        
        # Filter FVGs
        bullish_fvgs = [f for f in fvgs if f.fvg_type == 'Bullish' and not f.is_mitigated]
        bearish_fvgs = [f for f in fvgs if f.fvg_type == 'Bearish' and not f.is_mitigated]
        
        reason.append(f"📅 Session: {current_session}")
        
        # ====================================================================
        # BUY SETUP - FLEXIBLE SCORING SYSTEM
        # ====================================================================
        if bias == TrendState.BULLISH.value or bias == TrendState.RANGE.value:
            buy_score = 0
            buy_reasons = []
            
            # 1. Bullish structure (2 points)
            if bias == TrendState.BULLISH.value:
                buy_score += 2
                buy_reasons.append("✓ Bullish structure")
                conditions_met += 1
            elif bias == TrendState.RANGE.value:
                buy_score += 1
                buy_reasons.append("~ Range (neutral)")
            
            # 2. HTF alignment (1 point) - OPTIONAL
            if Config.HTF_LOCK_ENABLED and self.htf_bias:
                if self.htf_bias == TrendState.BULLISH.value:
                    buy_score += 1
                    buy_reasons.append(f"✓ HTF aligned: {self.htf_bias}")
                    conditions_met += 1
            else:
                conditions_met += 1  # Don't penalize if HTF lock disabled
            
            # 3. Discount/Fair Value zone (1 point)
            if pd_zone in ["Discount", "Fair Value"]:
                buy_score += 1
                buy_reasons.append(f"✓ Good zone: {pd_zone}")
                conditions_met += 1
            else:
                buy_reasons.append(f"~ Premium zone (caution)")
            
            # 4. Liquidity (2 points)
            if liquidity['SSL_swept']:
                buy_score += 2
                buy_reasons.append("✓ SSL swept (strong)")
                conditions_met += 1
            elif liquidity['SSL_near']:
                buy_score += 1
                buy_reasons.append("~ Price near SSL")
                conditions_met += 1
            
            # 5. FVG availability (1 point)
            if len(bullish_fvgs) > 0:
                buy_score += 1
                buy_reasons.append(f"✓ {len(bullish_fvgs)} Bullish FVG(s)")
                conditions_met += 1
            
            # 6. Price near FVG (2 points)
            price_near_fvg = False
            target_fvg = None
            
            for fvg in bullish_fvgs:
                if self.is_price_near_fvg(current_price, fvg):
                    price_near_fvg = True
                    target_fvg = fvg
                    buy_score += 2
                    buy_reasons.append("✓ Price near/in FVG")
                    conditions_met += 1
                    break
            
            if not price_near_fvg and len(bullish_fvgs) > 0:
                # Use nearest FVG anyway
                target_fvg = min(bullish_fvgs, key=lambda f: abs(f.midpoint - current_price))
                buy_score += 1
                buy_reasons.append("~ Using nearest FVG")
            
            # SIGNAL DECISION
            min_score = 4 if Config.REQUIRE_ALL_CONDITIONS else 3
            
            if buy_score >= min_score and target_fvg:
                # Determine if SCALP or SWING
                if timeframe in ['M5', 'M15'] and Config.SCALP_MODE:
                    signal = "BUY"
                    trade_type = "SCALP"
                    entry_zone = [target_fvg.bottom, target_fvg.midpoint]
                    stop_loss = target_fvg.bottom - (atr * 1.0)
                    take_profit = current_price + (atr * 2.0)
                    buy_reasons.append(f"💰 SCALP Setup (Score: {buy_score}/9)")
                elif timeframe in ['M30', 'H1'] and Config.SWING_MODE:
                    signal = "BUY"
                    trade_type = "SWING"
                    entry_zone = [target_fvg.bottom, target_fvg.top]
                    stop_loss = target_fvg.bottom - (atr * 1.5)
                    take_profit = analysis_result['structure_high']
                    buy_reasons.append(f"📈 SWING Setup (Score: {buy_score}/9)")
                elif timeframe in ['M5', 'M15']:
                    signal = "BUY"
                    trade_type = "SCALP"
                    entry_zone = [target_fvg.bottom, target_fvg.midpoint]
                    stop_loss = target_fvg.bottom - (atr * 1.0)
                    take_profit = current_price + (atr * 2.0)
                    buy_reasons.append(f"💰 SCALP Setup (Score: {buy_score}/9)")
                
                reason = buy_reasons
            else:
                buy_reasons.append(f"❌ Insufficient score ({buy_score}/9, need {min_score})")
                reason = buy_reasons
        
        # ====================================================================
        # SELL SETUP - FLEXIBLE SCORING SYSTEM
        # ====================================================================
        elif bias == TrendState.BEARISH.value or (bias == TrendState.RANGE.value and signal == "NO TRADE"):
            sell_score = 0
            sell_reasons = []
            
            # 1. Bearish structure
            if bias == TrendState.BEARISH.value:
                sell_score += 2
                sell_reasons.append("✓ Bearish structure")
                conditions_met += 1
            elif bias == TrendState.RANGE.value:
                sell_score += 1
                sell_reasons.append("~ Range (neutral)")
            
            # 2. HTF alignment
            if Config.HTF_LOCK_ENABLED and self.htf_bias:
                if self.htf_bias == TrendState.BEARISH.value:
                    sell_score += 1
                    sell_reasons.append(f"✓ HTF aligned: {self.htf_bias}")
                    conditions_met += 1
            else:
                conditions_met += 1
            
            # 3. Premium/Fair Value zone
            if pd_zone in ["Premium", "Fair Value"]:
                sell_score += 1
                sell_reasons.append(f"✓ Good zone: {pd_zone}")
                conditions_met += 1
            else:
                sell_reasons.append(f"~ Discount zone (caution)")
            
            # 4. Liquidity
            if liquidity['BSL_swept']:
                sell_score += 2
                sell_reasons.append("✓ BSL swept (strong)")
                conditions_met += 1
            elif liquidity['BSL_near']:
                sell_score += 1
                sell_reasons.append("~ Price near BSL")
                conditions_met += 1
            
            # 5. FVG availability
            if len(bearish_fvgs) > 0:
                sell_score += 1
                sell_reasons.append(f"✓ {len(bearish_fvgs)} Bearish FVG(s)")
                conditions_met += 1
            
            # 6. Price near FVG
            price_near_fvg = False
            target_fvg = None
            
            for fvg in bearish_fvgs:
                if self.is_price_near_fvg(current_price, fvg):
                    price_near_fvg = True
                    target_fvg = fvg
                    sell_score += 2
                    sell_reasons.append("✓ Price near/in FVG")
                    conditions_met += 1
                    break
            
            if not price_near_fvg and len(bearish_fvgs) > 0:
                target_fvg = min(bearish_fvgs, key=lambda f: abs(f.midpoint - current_price))
                sell_score += 1
                sell_reasons.append("~ Using nearest FVG")
            
            # SIGNAL DECISION
            min_score = 4 if Config.REQUIRE_ALL_CONDITIONS else 3
            
            if sell_score >= min_score and target_fvg:
                if timeframe in ['M5', 'M15'] and Config.SCALP_MODE:
                    signal = "SELL"
                    trade_type = "SCALP"
                    entry_zone = [target_fvg.midpoint, target_fvg.top]
                    stop_loss = target_fvg.top + (atr * 1.0)
                    take_profit = current_price - (atr * 2.0)
                    sell_reasons.append(f"💰 SCALP Setup (Score: {sell_score}/9)")
                elif timeframe in ['M30', 'H1'] and Config.SWING_MODE:
                    signal = "SELL"
                    trade_type = "SWING"
                    entry_zone = [target_fvg.bottom, target_fvg.top]
                    stop_loss = target_fvg.top + (atr * 1.5)
                    take_profit = analysis_result['structure_low']
                    sell_reasons.append(f"📈 SWING Setup (Score: {sell_score}/9)")
                elif timeframe in ['M5', 'M15']:
                    signal = "SELL"
                    trade_type = "SCALP"
                    entry_zone = [target_fvg.midpoint, target_fvg.top]
                    stop_loss = target_fvg.top + (atr * 1.0)
                    take_profit = current_price - (atr * 2.0)
                    sell_reasons.append(f"💰 SCALP Setup (Score: {sell_score}/9)")
                
                reason = sell_reasons
            else:
                sell_reasons.append(f"❌ Insufficient score ({sell_score}/9, need {min_score})")
                reason = sell_reasons
        
        # Avoid duplicate signals
        signal_key = f"{timeframe}_{signal}_{current_price:.0f}"
        if signal in ["BUY", "SELL"]:
            if signal_key in self.last_signals:
                last_time = self.last_signals[signal_key]
                if (datetime.now() - last_time).seconds < 180:  # 3 minutes cooldown
                    return signal, entry_zone, stop_loss, take_profit, reason, trade_type, conditions_met
            
            self.last_signals[signal_key] = datetime.now()
            self.alert_system.play_sound(signal)
            
            # Calculate confidence
            confidence = int((conditions_met / total_conditions) * 100)
            
            # Calculate R:R
            rr = abs(take_profit - entry_zone[0]) / abs(stop_loss - entry_zone[0]) if stop_loss != entry_zone[0] else 0
            
            # Create signal object
            trade_signal = TradeSignal(
                timestamp=datetime.now(),
                timeframe=timeframe,
                symbol=self.symbol.replace('m', ''),
                signal=signal,
                trade_type=trade_type,
                bias=bias,
                session=current_session,
                htf_bias=self.htf_bias if self.htf_bias else "N/A",
                entry_zone=entry_zone,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=rr,
                confidence=confidence,
                conditions_met=conditions_met,
                reason=reason,
                confirmed=False
            )
            
            self.logger.log_signal(trade_signal)
            
            if Config.TELEGRAM_ENABLED:
                message = f"<b>🚨 {signal} {trade_type}</b>\n\n"
                message += f"Symbol: {trade_signal.symbol}\n"
                message += f"Timeframe: {timeframe}\n"
                message += f"Confidence: {confidence}%\n"
                message += f"Entry: ${entry_zone[0]:.2f} - ${entry_zone[1]:.2f}\n"
                message += f"SL: ${stop_loss:.2f}\n"
                message += f"TP: ${take_profit:.2f}\n"
                message += f"R:R: 1:{rr:.2f}\n"
                self.alert_system.send_telegram(message)
        
        return signal, entry_zone, stop_loss, take_profit, reason, trade_type, conditions_met
    
    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================
    
    def analyze_timeframe(self, timeframe_str):
        """Complete ICT analysis"""
        df = self.get_candle_data(timeframe_str)
        if df is None or len(df) < 50:
            return None
        
        swing_highs, swing_lows = self.detect_swings(df)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        bias, structure = self.classify_structure(df, swing_highs, swing_lows)
        
        if timeframe_str == Config.HTF_TIMEFRAME:
            self.htf_bias = bias.value
        
        bos_choch = self.detect_bos_choch(df, swing_highs, swing_lows, timeframe_str)
        
        pd_zone, struct_high, struct_low, equilibrium = \
            self.calculate_premium_discount(df, swing_highs, swing_lows, bias)
        
        liquidity = self.detect_liquidity(df, swing_highs, swing_lows)
        fvgs = self.detect_fvg(df)
        current_session = SessionDetector.get_current_session()
        
        analysis = {
            "symbol": self.symbol.replace('m', ''),
            "timeframe": timeframe_str,
            "bias": bias.value,
            "structure": structure,
            "bos_choch": bos_choch,
            "premium_discount": pd_zone,
            "liquidity": liquidity,
            "fvgs": fvgs[-10:] if fvgs else [],
            "current_price": df.iloc[-1]['close'],
            "equilibrium": equilibrium,
            "structure_high": struct_high,
            "structure_low": struct_low,
            "swing_highs_count": len(swing_highs),
            "swing_lows_count": len(swing_lows),
            "session": current_session,
            "htf_bias": self.htf_bias if self.htf_bias else "N/A"
        }
        
        signal, entry_zone, stop_loss, take_profit, reason, trade_type, conditions_met = \
            self.generate_signal(analysis, df)
        
        analysis.update({
            "signal": signal,
            "trade_type": trade_type,
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": reason,
            "conditions_met": conditions_met
        })
        
        return analysis

# ============================================================================
# GUI CLASS (Same as before with small updates)
# ============================================================================

class ICTAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XAUUSD ICT/SMC Analyzer - Less Strict Mode")
        self.root.geometry("1400x900")
        
        try:
            self.root.state('zoomed')
        except:
            pass
        
        self.analyzer = ICTMarketAnalyzer()
        
        if not self.analyzer.connect_mt5():
            messagebox.showerror("MT5 Error", "Cannot connect to MT5. Please ensure MT5 is running.")
            self.root.destroy()
            return
        
        self.setup_ui()
        self.root.after(1500, self.safe_analyze_all)
    
    def setup_ui(self):
        """Setup GUI"""
        self.root.configure(bg='#1e1e1e')
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(
            main_frame,
            text="🏆 XAUUSD ICT/SMC Analyzer - Flexible Mode",
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=6, pady=(0, 15))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=6, pady=(0, 15))
        
        self.refresh_btn = ttk.Button(
            control_frame,
            text="🔄 Refresh",
            command=self.analyze_all,
            width=15
        )
        self.refresh_btn.grid(row=0, column=0, padx=5)
        
        self.export_btn = ttk.Button(
            control_frame,
            text="💾 Export",
            command=self.export_analysis,
            width=15
        )
        self.export_btn.grid(row=0, column=1, padx=5)
        
        self.stats_btn = ttk.Button(
            control_frame,
            text="📊 Stats",
            command=self.show_stats,
            width=15
        )
        self.stats_btn.grid(row=0, column=2, padx=5)
        
        self.session_var = tk.StringVar(value="Session: N/A")
        ttk.Label(control_frame, textvariable=self.session_var, font=("Arial", 10, "bold")).grid(row=0, column=3, padx=10)
        
        self.htf_bias_var = tk.StringVar(value="HTF: N/A")
        ttk.Label(control_frame, textvariable=self.htf_bias_var, font=("Arial", 10, "bold")).grid(row=0, column=4, padx=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=0, column=5, padx=20)
        
        # Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        self.timeframe_frames = {}
        for tf in ['M5', 'M15', 'M30', 'H1']:
            frame = ttk.Frame(self.notebook, padding="10")
            self.timeframe_frames[tf] = frame
            self.notebook.add(frame, text=f"  {tf}  ")
            self.setup_timeframe_frame(frame, tf)
        
        # Summary
        summary_frame = ttk.LabelFrame(main_frame, text="📊 Market Summary", padding="15")
        summary_frame.grid(row=3, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=10, width=120, font=("Consolas", 10))
        self.summary_text.grid(row=0, column=0)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def setup_timeframe_frame(self, parent, timeframe):
        """Setup timeframe display"""
        # Left column
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.W), padx=10)
        
        left_metrics = [
            ("Symbol:", "symbol"),
            ("Price:", "current_price"),
            ("Bias:", "bias"),
            ("Structure:", "structure"),
            ("Zone:", "zone"),
            ("Swings:", "swings"),
        ]
        
        for i, (label, key) in enumerate(left_metrics):
            ttk.Label(left_frame, text=label, font=("Arial", 10, "bold")).grid(row=i, column=0, sticky=tk.W, pady=3)
            value_var = tk.StringVar(value="...")
            ttk.Label(left_frame, textvariable=value_var, font=("Arial", 10)).grid(row=i, column=1, sticky=tk.W, padx=10, pady=3)
            setattr(self, f"{timeframe}_{key}_var", value_var)
        
        # Signal frame
        signal_frame = ttk.LabelFrame(parent, text="Trade Signal", padding="10")
        signal_frame.grid(row=0, column=1, sticky=(tk.N, tk.W), padx=10)
        
        signal_var = tk.StringVar(value="ANALYZING...")
        signal_label = ttk.Label(signal_frame, textvariable=signal_var, font=("Arial", 16, "bold"))
        signal_label.grid(row=0, column=0)
        setattr(self, f"{timeframe}_signal_display", signal_label)
        setattr(self, f"{timeframe}_signal_var", signal_var)
        
        type_var = tk.StringVar(value="")
        ttk.Label(signal_frame, textvariable=type_var, font=("Arial", 10)).grid(row=1, column=0)
        setattr(self, f"{timeframe}_type_var", type_var)
        
        entry_var = tk.StringVar(value="")
        ttk.Label(signal_frame, textvariable=entry_var, font=("Arial", 10)).grid(row=2, column=0, pady=5)
        setattr(self, f"{timeframe}_entry_var", entry_var)
        
        confidence_var = tk.StringVar(value="")
        ttk.Label(signal_frame, textvariable=confidence_var, font=("Arial", 10, "bold")).grid(row=3, column=0)
        setattr(self, f"{timeframe}_confidence_var", confidence_var)
        
        # Reason display
        reason_frame = ttk.LabelFrame(parent, text="Analysis", padding="10")
        reason_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10, padx=10)
        
        reason_text = scrolledtext.ScrolledText(reason_frame, height=8, width=80, font=("Consolas", 9))
        reason_text.grid(row=0, column=0)
        setattr(self, f"{timeframe}_reason_text", reason_text)
    
    def update_timeframe_display(self, timeframe, analysis):
        """Update display"""
        if not analysis:
            getattr(self, f"{timeframe}_signal_var").set("ERROR")
            return
        
        try:
            getattr(self, f"{timeframe}_symbol_var").set(analysis["symbol"])
            getattr(self, f"{timeframe}_current_price_var").set(f"${analysis['current_price']:.2f}")
            getattr(self, f"{timeframe}_bias_var").set(analysis["bias"])
            getattr(self, f"{timeframe}_structure_var").set(analysis["structure"])
            getattr(self, f"{timeframe}_zone_var").set(analysis['premium_discount'])
            getattr(self, f"{timeframe}_swings_var").set(f"H:{analysis['swing_highs_count']} L:{analysis['swing_lows_count']}")
            
            signal = analysis['signal']
            trade_type = analysis.get('trade_type', 'NONE')
            
            getattr(self, f"{timeframe}_signal_var").set(signal)
            
            if trade_type != 'NONE':
                getattr(self, f"{timeframe}_type_var").set(f"[{trade_type}]")
            
            signal_label = getattr(self, f"{timeframe}_signal_display")
            if signal == "BUY":
                signal_label.config(foreground="green")
            elif signal == "SELL":
                signal_label.config(foreground="red")
            else:
                signal_label.config(foreground="orange")
            
            if signal != "NO TRADE":
                entry_text = f"Entry: ${analysis['entry_zone'][0]:.2f} - ${analysis['entry_zone'][1]:.2f}\n"
                entry_text += f"SL: ${analysis['stop_loss']:.2f} | TP: ${analysis['take_profit']:.2f}"
                rr = abs(analysis['take_profit'] - analysis['entry_zone'][0]) / abs(analysis['stop_loss'] - analysis['entry_zone'][0])
                entry_text += f"\nR:R = 1:{rr:.2f}"
                getattr(self, f"{timeframe}_entry_var").set(entry_text)
                
                conf = int((analysis.get('conditions_met', 0) / 6) * 100)
                getattr(self, f"{timeframe}_confidence_var").set(f"Confidence: {conf}%")
            else:
                getattr(self, f"{timeframe}_entry_var").set("No setup")
                getattr(self, f"{timeframe}_confidence_var").set("")
            
            reason_text = getattr(self, f"{timeframe}_reason_text")
            reason_text.delete('1.0', tk.END)
            if 'reason' in analysis:
                reason_text.insert('1.0', "\n".join(analysis['reason']))
        
        except Exception as e:
            print(f"Error: {e}")
    
    def safe_analyze_all(self):
        try:
            self.analyze_all()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def analyze_all(self):
        """Analyze all timeframes"""
        self.refresh_btn.config(state="disabled")
        self.status_var.set("Analyzing...")
        thread = threading.Thread(target=self._analyze_thread)
        thread.start()
    
    def _analyze_thread(self):
        """Analysis thread"""
        all_analysis = {}
        
        for timeframe in ['M5', 'M15', 'M30', 'H1']:
            try:
                analysis = self.analyzer.analyze_timeframe(timeframe)
                all_analysis[timeframe] = analysis
                self.root.after(0, self.update_timeframe_display, timeframe, analysis)
            except Exception as e:
                print(f"Error {timeframe}: {e}")
        
        self.root.after(0, self.update_summary, all_analysis)
        self.root.after(0, lambda: self.refresh_btn.config(state="normal"))
        self.root.after(0, lambda: self.status_var.set("Complete"))
        
        # Update session info
        session = SessionDetector.get_current_session()
        self.root.after(0, lambda: self.session_var.set(f"Session: {session}"))
        if self.analyzer.htf_bias:
            self.root.after(0, lambda: self.htf_bias_var.set(f"HTF: {self.analyzer.htf_bias}"))
    
    def update_summary(self, all_analysis):
        """Update summary"""
        summary = []
        summary.append("=" * 100)
        summary.append("MARKET SUMMARY - FLEXIBLE MODE")
        summary.append("=" * 100)
        
        signals = {'BUY': [], 'SELL': [], 'NO TRADE': []}
        scalps = []
        swings = []
        
        for tf, analysis in all_analysis.items():
            if analysis:
                signals[analysis['signal']].append(tf)
                if analysis.get('trade_type') == 'SCALP':
                    scalps.append(f"{tf}:{analysis['signal']}")
                elif analysis.get('trade_type') == 'SWING':
                    swings.append(f"{tf}:{analysis['signal']}")
        
        summary.append(f"\n📊 SIGNALS:")
        summary.append(f"   BUY: {len(signals['BUY'])} {signals['BUY']}")
        summary.append(f"   SELL: {len(signals['SELL'])} {signals['SELL']}")
        summary.append(f"\n💰 SCALP: {len(scalps)} {scalps}")
        summary.append(f"📈 SWING: {len(swings)} {swings}")
        
        summary.append(f"\n💡 RECOMMENDATION:")
        if len(signals['BUY']) >= 2:
            summary.append("   ✅ BULLISH BIAS - Look for BUY entries")
        elif len(signals['SELL']) >= 2:
            summary.append("   ✅ BEARISH BIAS - Look for SELL entries")
        else:
            summary.append("   ⚠️ MIXED SIGNALS - Trade with caution")
        
        summary.append(f"\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}")
        summary.append("=" * 100)
        
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', "\n".join(summary))
    
    def export_analysis(self):
        """Export to JSON"""
        messagebox.showinfo("Export", "Analysis export feature - check log file")
    
    def show_stats(self):
        """Show statistics"""
        stats = self.analyzer.logger.get_stats()
        if not stats:
            messagebox.showinfo("Stats", "No signals logged yet")
            return
        
        msg = f"""Total Signals: {stats.get('total_signals', 0)}
Buy: {stats.get('buy_signals', 0)}
Sell: {stats.get('sell_signals', 0)}
Scalps: {stats.get('scalp_signals', 0)}
Swings: {stats.get('swing_signals', 0)}
Avg R:R: {stats.get('avg_risk_reward', 0):.2f}
Avg Confidence: {stats.get('avg_confidence', 0):.0f}%"""
        
        messagebox.showinfo("Statistics", msg)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        root.withdraw()
        app = ICTAnalysisGUI(root)
        root.deiconify()
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if mt5.initialize():
            mt5.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print("XAUUSD ICT/SMC Analyzer - Flexible Mode")
    print("=" * 60)
    print("Less strict, more signals, follows trend better")
    print()
    
    if not mt5.initialize():
        print("ERROR: Cannot connect to MT5")
        input("Press Enter to exit...")
    else:
        print("✓ MT5 connected")
        mt5.shutdown()
        print("✓ Starting...")
        main()
