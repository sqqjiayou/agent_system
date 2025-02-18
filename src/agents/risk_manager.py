import math
from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
import json
import ast


def risk_management_agent(state: AgentState):
    """Evaluates forex trading risk based on minute-level forex data analysis."""
    # print('risk_management_agent')
    # print('#'*50)
    show_agent_reasoning_flag = state["metadata"].get("show_agent_reasoning", state["metadata"].get("show_reasoning"))
    data = state["data"]
    symbol = data["ticker"]
    time = data["time"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    prices_df = data["prices"]
    
    # Get messages from other agents
    technical_message = next(msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(msg for msg in state["messages"] if msg.name == "valuation_agent")

    try:
        agent_signals = {
            "fundamental": json.loads(fundamentals_message.content),
            "technical": json.loads(technical_message.content),
            "sentiment": json.loads(sentiment_message.content),
            "valuation": json.loads(valuation_message.content)
        }
    except Exception:
        agent_signals = {
            "fundamental": ast.literal_eval(fundamentals_message.content),
            "technical": ast.literal_eval(technical_message.content),
            "sentiment": ast.literal_eval(sentiment_message.content),
            "valuation": ast.literal_eval(valuation_message.content)
        }

    # Calculate minute-based volatility metrics
    returns = prices_df['close'].pct_change().dropna()
    
    # Calculate pip volatility (average high-low range)
    pip_value = 0.01 if data["ticker"] == "JPY" else 0.0001
    pip_volatility = (prices_df['high'] - prices_df['low']).mean() / pip_value
    
    # Calculate rolling volatility (60-minute window)
    rolling_vol = returns.rolling(window=60).std() * math.sqrt(1440)  # Annualized
    current_vol = rolling_vol.iloc[-1]
    
    # Calculate relative volatility percentile
    vol_percentile = (current_vol - rolling_vol.mean()) / rolling_vol.std()
    
    # Calculate short-term VaR (2-hour window)
    var_95 = returns.rolling(window=120).quantile(0.05).iloc[-1]
    
    # Calculate recent maximum drawdown (4-hour window)
    rolling_max = prices_df['close'].rolling(window=240).max()
    drawdown = (prices_df['close'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Aggregate signal score (-1 to 1 scale)
    def get_signal_score(signal_data):
        signal_map = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0
        }
        signal = signal_data.get('signal', 'neutral').lower()
        confidence = float(str(signal_data.get('confidence', '0')).strip('%')) / 100
        return signal_map.get(signal, 0) * confidence

    combined_signal = sum(get_signal_score(signal) for signal in agent_signals.values())

    # Normalize each risk component to 0-2.5 scale, sum to max 10
    normalized_risk_components = {
        "volatility_risk": min(2.5, abs(vol_percentile) * 1.25),
        "pip_volatility_risk": min(2.5, (pip_volatility / 50) * 1.25),
        "var_risk": min(2.5, abs(var_95) * 200),
        "drawdown_risk": min(2.5, abs(max_drawdown) * 25)
    }
    
    risk_score = sum(normalized_risk_components.values())

    # Interpret risk levels for each component
    risk_interpretations = {
        "volatility": "High" if current_vol > 0.001 else "Moderate" if current_vol > 0.0005 else "Low",
        "pip_volatility": "High" if pip_volatility > 75 else "Moderate" if pip_volatility > 40 else "Low",
        "var": "High" if abs(var_95) > 0.005 else "Moderate" if abs(var_95) > 0.002 else "Low",
        "drawdown": "High" if abs(max_drawdown) > 0.004 else "Moderate" if abs(max_drawdown) > 0.002 else "Low"
    }

    # Signal strength interpretation (combined_signal ranges from -4 to +4)
    signal_strength = (
        "Strong Bullish" if combined_signal > 2 else
        "Moderate Bullish" if combined_signal > 1 else
        "Weak Bullish" if combined_signal > 0 else
        "Neutral" if combined_signal == 0 else
        "Weak Bearish" if combined_signal > -1 else
        "Moderate Bearish" if combined_signal > -2 else
        "Strong Bearish"
    )

    # Calculate maximum position size based on risk score
    base_position = 1
    max_position = (
        base_position * 0.3 if risk_score > 7 else
        base_position * 0.5 if risk_score > 4 else
        base_position
    )

    message_content = {
        "risk_score": float(risk_score),
        "max_position_size": float(max_position),  # Added this field
        "risk_metrics": {
            "volatility": float(current_vol),  # Higher indicates more risk
            "pip_volatility": float(pip_volatility),  # Higher indicates more risk
            "value_at_risk_95": float(var_95),  # More negative indicates more risk
            "max_drawdown": float(max_drawdown),  # More negative indicates more risk
            "combined_signal": float(combined_signal)  # Range: -4 to +4, positive indicates bullish
        },
        "normalized_components": normalized_risk_components,
        "reasoning": (
            f"Overall Risk Score: {risk_score:.1f}/10\n"
            f"Recommended Position Size: {max_position:.1f}x (base position)\n"  # Added this line
            f"Risk Breakdown:\n"
            f"- Volatility Risk ({risk_interpretations['volatility']}): {current_vol:.6f} "
            f"(normalized: {normalized_risk_components['volatility_risk']:.1f}/2.5)\n"
            f"- Pip Volatility Risk ({risk_interpretations['pip_volatility']}): {pip_volatility:.1f} pips "
            f"(normalized: {normalized_risk_components['pip_volatility_risk']:.1f}/2.5)\n"
            f"- Value at Risk ({risk_interpretations['var']}): {var_95:.6f} "
            f"(normalized: {normalized_risk_components['var_risk']:.1f}/2.5)\n"
            f"- Drawdown Risk ({risk_interpretations['drawdown']}): {max_drawdown:.6f} "
            f"(normalized: {normalized_risk_components['drawdown_risk']:.1f}/2.5)\n"
            f"Market Signal: {signal_strength} (Score: {combined_signal:.2f})\n"
            f"Market Condition Summary: "
            f"{'High risk environment - extra caution advised' if risk_score > 7 else 'Moderate risk environment' if risk_score > 4 else 'Lower risk environment'} "
            f"- Position size reduced to {max_position:.1f}x"  # Added position size context
        )
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_agent_reasoning_flag:
        show_agent_reasoning(message_content, "Forex Risk Management Agent", 'Risk', time, symbol, start_date, end_date)

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }