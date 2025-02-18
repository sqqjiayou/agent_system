from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
import json

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes forex fundamental data and generates trading signals."""
    # print('fundamentals_agent')
    # print('#'*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["macro_metrics"][0]  # 使用新的macro_metrics替代financial_metrics
    currency = data["ticker"][:3]  # 现在使用单个货币代码

    signals = []
    reasoning = {}
    
    # 1. Interest Rate Analysis
    interest_rate_diff = metrics.get("interest_rate_differential")
    currency_usd_policy = metrics.get("monetary_policy_stance", {})
    
    rate_score = 0
    if interest_rate_diff > 0:  # 目标货币利率更高
        rate_score += 1
    if currency_usd_policy.get("currency") == "tightening":
        rate_score += 1
    if currency_usd_policy.get("usd") == "easing":
        rate_score += 1
        
    signals.append('bullish' if rate_score >= 2 else 'bearish' if rate_score == 0 else 'neutral')
    reasoning["interest_rate_signal"] = {
        "signal": signals[0],
        "details": (
            f"Interest Rate Differential: {interest_rate_diff:.2f}%, " +
            f"Currency Policy: {currency_usd_policy.get('currency', 'unknown')}, " +
            f"USD Policy: {currency_usd_policy.get('usd', 'unknown')}"
        )
    }
    
    # 2. Economic Growth Analysis
    gdp_growth_diff = metrics.get("gdp_growth_differential")
    economic_indicators = metrics.get("economic_indicators", {})
    currency_indicators = economic_indicators.get("currency", {})
    usd_indicators = economic_indicators.get("usd", {})
    
    growth_score = 0
    if gdp_growth_diff > 0:  # 目标货币GDP增长更快
        growth_score += 1
    if currency_indicators.get("unemployment", 100) < usd_indicators.get("unemployment", 0):
        growth_score += 1
    if currency_indicators.get("trade_balance", -100) > usd_indicators.get("trade_balance", 0):
        growth_score += 1
        
    signals.append('bullish' if growth_score >= 2 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning["growth_signal"] = {
        "signal": signals[1],
        "details": (
            f"GDP Growth Differential: {gdp_growth_diff:.2f}%, " +
            f"{currency} Unemployment: {currency_indicators.get('unemployment', 'N/A')}%, " +
            f"USD Unemployment: {usd_indicators.get('unemployment', 'N/A')}%"
        )
    }
    
    # 3. Inflation Analysis
    inflation_diff = metrics.get("inflation_differential")
    
    inflation_score = 0
    if abs(inflation_diff) < 1:  # 通胀差异较小
        inflation_score += 1
    if inflation_diff > 0 and currency_usd_policy.get("currency") == "tightening":
        inflation_score += 1
    if inflation_diff < 0 and currency_usd_policy.get("usd") == "tightening":
        inflation_score += 1
        
    signals.append('bullish' if inflation_score >= 2 else 'bearish' if inflation_score == 0 else 'neutral')
    reasoning["inflation_signal"] = {
        "signal": signals[2],
        "details": f"Inflation Differential: {inflation_diff:.2f}%"
    }
    
    # 4. Trade Balance Analysis
    trade_balance_currency = currency_indicators.get("trade_balance")
    trade_balance_usd = usd_indicators.get("trade_balance")
    
    trade_score = 0
    if trade_balance_currency and trade_balance_usd:
        if trade_balance_currency > 0:  # 目标货币有贸易顺差
            trade_score += 1
        if trade_balance_currency > trade_balance_usd:  # 目标货币贸易状况更好
            trade_score += 1
            
    signals.append('bullish' if trade_score >= 1 else 'bearish' if trade_score == 0 else 'neutral')
    reasoning["trade_balance_signal"] = {
        "signal": signals[3],
        "details": (
            f"{currency} Trade Balance: {trade_balance_currency:,.1f}B, " +
            f"USD Trade Balance: {trade_balance_usd:,.1f}B"
        )
    }
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }
    
    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamentals_agent",
    )
    
    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Forex Fundamental Analysis Agent")
    
    return {
        "messages": [message],
        "data": data,
    }