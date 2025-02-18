from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
import json

def valuation_agent(state: AgentState):
    """Performs forex valuation analysis using multiple methodologies."""
    # print('valuation_agent')
    # print('#'*50)
    try:
        show_reasoning = state["metadata"]["show_reasoning"]
        data = state["data"]
        symbol = data["ticker"]
        time = data["time"]
        start_date = data["start_date"]
        end_date = data["end_date"]
        
        # 获取当前汇率
        current_rate = data["prices"].iloc[-1]["close"]
        
        # 获取宏观指标
        metrics = data.get("macro_metrics", [])
        if not metrics:
            raise ValueError("No macro metrics data available")
            
        current_metrics = metrics[0]  # 使用最新的指标数据
        
        # 安全地获取通胀率和利率数据
        currency_data = current_metrics.get("economic_indicators", {}).get("currency", {})
        usd_data = current_metrics.get("economic_indicators", {}).get("usd", {})
        
        # 获取利率差异
        interest_differential = current_metrics.get("interest_rate_differential", 0.0)
        inflation_differential = current_metrics.get("inflation_differential", 0.0)
        
        # 计算PPP值
        ppp_value = calculate_ppp_value(
            currency_inflation=inflation_differential,  # 直接使用差异值
            usd_inflation=0,  # 差异值已经包含了USD的影响
            current_rate=current_rate
        )
        
        # 计算利率平价值
        interest_parity_value = calculate_interest_parity(
            currency_interest_rate=interest_differential,  # 直接使用差异值
            usd_interest_rate=0,  # 差异值已经包含了USD的影响
            current_rate=current_rate,
            forward_points=0
        )
        
        # 检查当前汇率有效性
        if current_rate <= 0:
            raise ValueError("Invalid current rate")
        
        # 计算价值偏离度
        ppp_gap = (ppp_value - current_rate) / current_rate if ppp_value > 0 else 0
        interest_gap = (interest_parity_value - current_rate) / current_rate if interest_parity_value > 0 else 0
        
        # 综合估值偏离度
        valuation_gap = (ppp_gap + interest_gap) / 2 if ppp_gap != 0 and interest_gap != 0 else 0
        
        # 确定信号
        if valuation_gap > 0.02:  # 超过2%低估
            signal = 'bullish'
        elif valuation_gap < -0.02:  # 超过2%高估
            signal = 'bearish'
        else:
            signal = 'neutral'
            
        # 记录分析原因
        reasoning = {
            "ppp_analysis": {
                "signal": "bullish" if ppp_gap > 0.02 else "bearish" if ppp_gap < -0.02 else "neutral",
                "details": f"PPP Value: {ppp_value:.4f}, Current Rate: {current_rate:.4f}, Gap: {ppp_gap:.2%}"
            },
            "interest_parity_analysis": {
                "signal": "bullish" if interest_gap > 0.02 else "bearish" if interest_gap < -0.02 else "neutral",
                "details": f"IRP Value: {interest_parity_value:.4f}, Current Rate: {current_rate:.4f}, Gap: {interest_gap:.2%}"
            },
            "market_conditions": {
                "inflation_differential": inflation_differential,
                "interest_rate_differential": interest_differential
            }
        }
        
        message_content = {
            "signal": signal,
            "confidence": f"{abs(valuation_gap):.1%}",
            "reasoning": reasoning
        }
        
    except Exception as e:
        print(f"Debug: Error in valuation analysis: {str(e)}")
        message_content = {
            "signal": "neutral",
            "confidence": "0%",
            "reasoning": {"error": f"Analysis error: {str(e)}"}
        }
    
    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )
    
    if show_reasoning:
        show_agent_reasoning(message_content, "FX Valuation Analysis Agent", 'Valuation', time, symbol, start_date, end_date)
        
    return {
        "messages": [message],
        "data": data,
    }

def calculate_ppp_value(
    currency_inflation: float,
    usd_inflation: float,
    current_rate: float,
    weight_cpi: float = 1.0  # 简化模型，仅使用CPI
) -> float:
    """
    计算购买力平价理论汇率
    
    Args:
        currency_inflation: 目标货币通胀率
        usd_inflation: 美元通胀率
        current_rate: 当前汇率
        weight_cpi: CPI权重
    
    Returns:
        float: PPP理论汇率
    """
    if current_rate <= 0:
        return 0
        
    # 计算相对通胀
    relative_inflation = currency_inflation - usd_inflation
        
    # 计算PPP理论汇率
    ppp_rate = current_rate * (1 + relative_inflation)
    
    return ppp_rate

def calculate_interest_parity(
    currency_interest_rate: float,
    usd_interest_rate: float,
    current_rate: float,
    forward_points: float,
    time_period: float = 1.0  # 年
) -> float:
    """
    计算利率平价理论汇率
    
    Args:
        currency_interest_rate: 目标货币利率
        usd_interest_rate: 美元利率
        current_rate: 当前即期汇率
        forward_points: 远期点数
        time_period: 时间周期(年)
    
    Returns:
        float: 利率平价理论汇率
    """
    if current_rate <= 0:
        return 0
        
    # 计算利率差
    interest_differential = currency_interest_rate - usd_interest_rate
    
    # 计算理论远期汇率
    theoretical_forward = current_rate * (
        1 + currency_interest_rate * time_period
    ) / (1 + usd_interest_rate * time_period)
    
    # 考虑实际远期点数的影响
    actual_forward = current_rate + forward_points
    
    # 返回加权平均值
    return (theoretical_forward + actual_forward) / 2