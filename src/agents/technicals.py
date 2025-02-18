import math
from typing import Dict

from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
import json

def technical_analyst_agent(state: AgentState):
    """
    Simple technical analysis system for minute-level forex data using 3 core indicators:
    1. EMA (Exponential Moving Average) - for trend direction
    2. RSI (Relative Strength Index) - for overbought/oversold conditions
    3. MACD (Moving Average Convergence Divergence) - for momentum and trend confirmation
    """
    # print('technical_analyst_agent')
    # print('#'*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    time = data["time"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    prices_df = data["prices"]
    
    # Calculate short-term and medium-term EMAs
    ema_5 = calculate_ema(prices_df['close'], 5)   # Fast EMA for quick trends
    ema_20 = calculate_ema(prices_df['close'], 20) # Slow EMA for trend confirmation
    
    # Calculate RSI with 14 periods
    rsi = calculate_rsi(prices_df['close'], 14)
    
    # Calculate MACD (12,26,9)
    macd_line, signal_line = calculate_macd(prices_df['close'])
    
    # Analysis and Signal Generation
    signals = []
    reasoning = {}
    
    # 1. EMA Analysis - Trend Direction
    ema_trend = analyze_ema_trend(ema_5, ema_20)
    signals.append(ema_trend['signal'])
    reasoning['EMA'] = {
        'signal': ema_trend['signal'],
        'details': ema_trend['details']
    }
    
    # 2. RSI Analysis - Overbought/Oversold
    rsi_signal = analyze_rsi(rsi)
    signals.append(rsi_signal['signal'])
    reasoning['RSI'] = {
        'signal': rsi_signal['signal'],
        'details': rsi_signal['details']
    }
    
    # 3. MACD Analysis - Momentum
    macd_signal = analyze_macd(macd_line, signal_line)
    signals.append(macd_signal['signal'])
    reasoning['MACD'] = {
        'signal': macd_signal['signal'],
        'details': macd_signal['details']
    }
    
    # Determine overall signal and confidence
    overall_signal, confidence = calculate_overall_signal(signals)
    
    # Generate analysis report
    analysis_report = {
        "signal": overall_signal,
        "confidence": confidence,
        "market_state": determine_market_state(ema_trend, rsi_signal, macd_signal),
        "reasoning": reasoning
    }
    
    message = HumanMessage(
        content=json.dumps(analysis_report),
        name="technical_analyst_agent",
    )
    
    if show_reasoning:
        show_agent_reasoning(analysis_report, "Technical Analyst", 'Technical', time, symbol, start_date, end_date)
        
    return {
        "messages": [message],
        "data": data,
    }

def calculate_ema(series, periods):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=periods, adjust=False).mean()

def calculate_rsi(series, periods=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    """Calculate MACD and Signal Line"""
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def analyze_ema_trend(ema_5, ema_20):
    """
    Analyze trend direction using EMAs
    Returns bullish if fast EMA > slow EMA, bearish if opposite
    """
    current_fast = ema_5.iloc[-1]
    current_slow = ema_20.iloc[-1]
    prev_fast = ema_5.iloc[-2]
    prev_slow = ema_20.iloc[-2]
    
    if current_fast > current_slow and prev_fast > prev_slow:
        return {'signal': 'bullish', 'details': 'Strong uptrend: Fast EMA above Slow EMA'}
    elif current_fast < current_slow and prev_fast < prev_slow:
        return {'signal': 'bearish', 'details': 'Strong downtrend: Fast EMA below Slow EMA'}
    elif current_fast > current_slow and prev_fast < prev_slow:
        return {'signal': 'bullish', 'details': 'Potential trend reversal: Fast EMA crossed above Slow EMA'}
    elif current_fast < current_slow and prev_fast > prev_slow:
        return {'signal': 'bearish', 'details': 'Potential trend reversal: Fast EMA crossed below Slow EMA'}
    else:
        return {'signal': 'neutral', 'details': 'Sideways movement: EMAs close together'}

def analyze_rsi(rsi):
    """
    Analyze RSI values
    Oversold < 30, Overbought > 70
    """
    current_rsi = rsi.iloc[-1]
    
    if current_rsi < 30:
        return {'signal': 'bullish', 'details': f'Oversold conditions: RSI at {current_rsi:.1f}'}
    elif current_rsi > 70:
        return {'signal': 'bearish', 'details': f'Overbought conditions: RSI at {current_rsi:.1f}'}
    else:
        return {'signal': 'neutral', 'details': f'Neutral RSI: {current_rsi:.1f}'}

def analyze_macd(macd_line, signal_line):
    """
    Analyze MACD crossovers and momentum
    """
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        return {'signal': 'bullish', 'details': 'MACD crossed above signal line'}
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        return {'signal': 'bearish', 'details': 'MACD crossed below signal line'}
    elif macd_line.iloc[-1] > signal_line.iloc[-1]:
        return {'signal': 'bullish', 'details': 'MACD above signal line'}
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return {'signal': 'bearish', 'details': 'MACD below signal line'}
    else:
        return {'signal': 'neutral', 'details': 'MACD and signal line converging'}

def calculate_overall_signal(signals):
    """Calculate overall signal and confidence based on indicator agreement"""
    bullish_count = signals.count('bullish')
    bearish_count = signals.count('bearish')
    total_signals = len(signals)
    
    if bullish_count > bearish_count:
        return 'bullish', f"{(bullish_count/total_signals*100):.0f}%"
    elif bearish_count > bullish_count:
        return 'bearish', f"{(bearish_count/total_signals*100):.0f}%"
    else:
        return 'neutral', "50%"

def determine_market_state(ema_trend, rsi_signal, macd_signal):
    """
    Determine current market state (trending up/down or ranging)
    """
    if ema_trend['signal'] == 'bullish' and (rsi_signal['signal'] == 'bullish' or macd_signal['signal'] == 'bullish'):
        return "Upward Trending"
    elif ema_trend['signal'] == 'bearish' and (rsi_signal['signal'] == 'bearish' or macd_signal['signal'] == 'bearish'):
        return "Downward Trending"
    elif ema_trend['signal'] == 'neutral':
        return "Ranging/Sideways"
    else:
        return "Mixed Signals"

