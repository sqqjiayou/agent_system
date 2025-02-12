import math
from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
from tools.api import prices_to_df
import json
import ast

def risk_management_agent(state: AgentState):
    """Evaluates forex trading risk and sets position limits based on comprehensive risk analysis."""

    show_agent_reasoning_flag = state["metadata"]["show_agent_reasoning"] if "show_agent_reasoning" in state["metadata"] else state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    
    prices_df = prices_to_df(data["prices"])
    
    # 获取其他代理的消息
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")

    try:
        fundamental_signals = json.loads(fundamentals_message.content)
        technical_signals = json.loads(technical_message.content)
        sentiment_signals = json.loads(sentiment_message.content)
        valuation_signals = json.loads(valuation_message.content)
    except Exception as e:
        fundamental_signals = ast.literal_eval(fundamentals_message.content)
        technical_signals = ast.literal_eval(technical_message.content)
        sentiment_signals = ast.literal_eval(sentiment_message.content)
        valuation_signals = ast.literal_eval(valuation_message.content)

    agent_signals = {
        "fundamental": fundamental_signals,
        "technical": technical_signals,
        "sentiment": sentiment_signals,
        "valuation": valuation_signals
    }

    # 1. 计算风险指标
    returns = prices_df['close'].pct_change().dropna()
    
    # 计算pip波动
    pip_value = 0.0001 if data["ticker"] != "JPY" else 0.01
    pip_volatility = (prices_df['high'] - prices_df['low']).mean() / pip_value
    
    # 日内波动率和年化波动率
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)
    
    # 计算波动率分布
    rolling_std = returns.rolling(window=24).std() * (252 ** 0.5)  # 使用24小时窗口
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = (volatility - volatility_mean) / volatility_std
    
    # 计算VaR和最大回撤
    var_95 = returns.quantile(0.05)
    max_drawdown = (prices_df['close'] / prices_df['close'].rolling(window=24).max() - 1).min()

    # 2. 市场风险评估
    market_risk_score = 0
    
    # 波动率评分
    if volatility_percentile > 1.5:
        market_risk_score += 2
    elif volatility_percentile > 1.0:
        market_risk_score += 1
        
    # Pip波动评分
    if pip_volatility > 100:  # 高pip波动
        market_risk_score += 2
    elif pip_volatility > 50:
        market_risk_score += 1
    
    # VaR评分
    if var_95 < -0.01:  # 外汇市场通常波动较小
        market_risk_score += 2
    elif var_95 < -0.005:
        market_risk_score += 1
        
    # 最大回撤评分
    if max_drawdown < -0.05:
        market_risk_score += 2
    elif max_drawdown < -0.02:
        market_risk_score += 1

    # 3. 头寸限制计算
    current_position_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_position_value
    
    # 基础头寸大小（考虑杠杆）
    leverage = data.get("leverage", 1)  # 默认杠杆为1
    base_position_size = total_portfolio_value * 0.02 * leverage  # 使用2%风险
    
    # 根据风险调整头寸
    if market_risk_score >= 4:
        max_position_size = base_position_size * 0.3  # 高风险大幅减少
    elif market_risk_score >= 2:
        max_position_size = base_position_size * 0.5  # 中等风险适度减少
    else:
        max_position_size = base_position_size

    # 4. 风险调整信号分析
    def parse_confidence(conf_str):
        try:
            if isinstance(conf_str, str):
                return float(conf_str.replace('%', '')) / 100.0
            return float(conf_str)
        except:
            return 0.0

    # 检查低置信度信号
    low_confidence = any(parse_confidence(
        signal['confidence']) < 0.20 for signal in agent_signals.values())
    
    # 检查信号分歧
    unique_signals = set(signal['signal'] for signal in agent_signals.values())
    signal_divergence = (2 if len(unique_signals) == 3 else 0)
    
    # 计算最终风险分数
    risk_score = market_risk_score + (2 if low_confidence else 0) + signal_divergence
    risk_score = min(round(risk_score), 10)

    # -----------------------------------------------------------------
    # 修改交易行动判断：根据四个agent的信号 (fundamental, technical, sentiment, valuation)
    # 以及 risk_score 采用以下规则：
    # 若 risk_score <= 3:
    #    Long条件: (至少1个agent显示 long 且其他agent中没有 short)
    #               或至少2个agent显示 long 且剩余agent不全为 short
    #               或至少3个agent显示 long
    #    Short条件: 同理。
    # 若 risk_score 在 (3,6]:
    #    Long条件: 如果有 3 个agent显示 long（或全部4个）且剩下的agent不是 short，则可 long
    #    Short条件: 如果有 3 个agent显示 short（或全部4个）且剩下的agent不是 long，则可 short
    #    此区间可满仓
    # 若 risk_score 在 (6,8]:
    #    同 (3,6] 的条件，但必须半仓
    # 若 risk_score > 8:
    #    动作设为 neutral
    #
    # 同时根据各agent的 confidence（取支持该方向的平均值）：
    # 当平均 confidence >= 0.5 时，允许满仓；否则半仓。
    # -----------------------------------------------------------------

    # 先对各agent的信号做标准化映射
    signal_mapping = {
        'bullish': 'long',
        'bearish': 'short',
        'neutral': 'neutral',
        'buy': 'long',
        'sell': 'short',
        'hold': 'neutral'
    }
    for agent_name, agent_data in agent_signals.items():
        if 'signal' in agent_data:
            original_signal = agent_data['signal'].lower()
            agent_data['signal'] = signal_mapping.get(original_signal, 'neutral')
    
    # 统计四个agent信号数量
    signals = [agent_data.get("signal", "neutral") for agent_data in agent_signals.values()]
    bullish_count = signals.count("long")
    bearish_count = signals.count("short")
    
    # 辅助函数：计算目标信号的平均 confidence
    def avg_confidence(target):
        confs = []
        for agent_data in agent_signals.values():
            if agent_data.get("signal", "neutral") == target:
                confs.append(parse_confidence(agent_data.get("confidence", "0%")))
        if confs:
            return sum(confs) / len(confs)
        else:
            return 0.0
    
    long_avg_conf = avg_confidence("long")
    short_avg_conf = avg_confidence("short")
    
    # 初始化决策变量
    trading_action = "neutral"
    position_size = "full"  # "full" 表示满仓，"half" 表示半仓
    
    if risk_score > 8:
        trading_action = "neutral"
    elif risk_score <= 3:
        # 长仓条件
        if (bullish_count >= 1 and bearish_count == 0) or (bullish_count >= 2 and bearish_count < 2) or (bullish_count >= 3):
            trading_action = "long"
            position_size = "full" if long_avg_conf >= 0.5 else "half"
        # 短仓条件
        elif (bearish_count >= 1 and bullish_count == 0) or (bearish_count >= 2 and bullish_count < 2) or (bearish_count >= 3):
            trading_action = "short"
            position_size = "full" if short_avg_conf >= 0.5 else "half"
        else:
            trading_action = "neutral"
    elif risk_score > 3 and risk_score <= 6:
        # 对于风险中低水平，要求至少3个agent支持（且剩余的agent无反对）
        if bullish_count == 4 or (bullish_count == 3 and bearish_count == 0):
            trading_action = "long"
            position_size = "full"
        elif bearish_count == 4 or (bearish_count == 3 and bullish_count == 0):
            trading_action = "short"
            position_size = "full"
        else:
            trading_action = "neutral"
    elif risk_score > 6 and risk_score <= 8:
        if bullish_count == 4 or (bullish_count == 3 and bearish_count == 0):
            trading_action = "long"
            position_size = "half"
        elif bearish_count == 4 or (bearish_count == 3 and bullish_count == 0):
            trading_action = "short"
            position_size = "half"
        else:
            trading_action = "neutral"

    # -----------------------------------------------------------------
    # 计算建议止损点位
    current_price = prices_df['close'].iloc[-1]
    atr = prices_df['high'].rolling(14).max() - prices_df['low'].rolling(14).min()
    pip_val = 0.01 if state["data"]["ticker"] == "JPY" else 0.0001
    stop_loss_pips = int(atr.iloc[-1] / pip_val)
    
    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "position_size": position_size,
        "risk_metrics": {
            "volatility": float(volatility),
            "pip_volatility": float(pip_volatility),
            "value_at_risk_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score,
            "stop_loss_pips": stop_loss_pips
        },
        "reasoning": (
            f"Risk Score {risk_score}/10: Market Risk={market_risk_score}, "
            f"Pip Vol={pip_volatility:.1f}, Daily Vol={volatility:.2%}, Stop Loss={stop_loss_pips} pips. "
            f"Signals: {signals}; bullish_count: {bullish_count}, bearish_count: {bearish_count}. "
            f"Long Avg Confidence: {long_avg_conf:.2f}, Short Avg Confidence: {short_avg_conf:.2f}."
        )
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_agent_reasoning_flag:
        show_agent_reasoning(message_content, "Forex Risk Management Agent")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }