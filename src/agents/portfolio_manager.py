from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tools.openrouter_config import get_chat_completion
from agents.state import AgentState, show_agent_reasoning
import json

def portfolio_management_agent(state: AgentState):
    """Makes final forex trading decisions and generates orders"""

    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    currency = state["data"]["ticker"]
    
    # Get all agent messages
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # Create system message with forex-specific guidelines.
    # 修改提示：quantity 必须为三个选项之一：满仓、半仓或 0。
    system_message = {
        "role": "system",
        "content": f'''You are a forex portfolio manager making final trading decisions for {currency}.
Your job is to make trading decisions based on the team's analysis while considering risk management guidelines.
...
RISK MANAGEMENT GUIDELINES:
- Strictly follow position size limits from risk management
- Never exceed maximum leverage ratios
- Consider volatility and market conditions
- Monitor currency correlations
- Use appropriate stop-loss levels

Signal Weighting for Forex Markets:
1. Technical Analysis (35% weight)
   - Critical for forex trading
   - Primary driver for entry/exit timing
   - Focus on price action and momentum
   - Key for identifying support/resistance levels
2. Fundamental Analysis (25% weight)
   - Economic indicators and central bank policies
   - Interest rate differentials
   - GDP and employment data
   - Trade balances
3. Sentiment Analysis (20% weight)
   - Market positioning and flows
   - News impact and market sentiment
   - Institutional activity
   - Retail trader positioning
4. Valuation Analysis (20% weight)
   - Purchasing power parity
   - Interest rate parity
   - Fair value metrics
   - Economic differentials

Decision Process for Forex:
1. Analyze technical signals for market conditions and timing
2. Validate with fundamental factors and economic data
3. Consider market sentiment and positioning
4. Confirm with valuation metrics
5. Apply risk management rules
6. Make final position decision

Provide the following in your output:
- "action": "long" | "short" | "neutral",
- "quantity": <target position: full (i.e. available cash-based full position plus current holding), half (half of full position), or 0>,
- "confidence": <float between 0 and 1>,
- "agent_signals": <object mapping each agent name to its signal and confidence>,
- "reasoning": <concise explanation of the decision including how you weighted the signals>.

Output must be a valid JSON object containing exactly the above keys and no additional text.'''
    }

    # Create user message with forex-specific portfolio info
    user_message = {
        "role": "user",
        "content": f"""Analyze the following signals and make a forex trading decision for {currency}:

Technical Analysis: {technical_message.content}
Fundamental Analysis: {fundamentals_message.content}
Sentiment Analysis: {sentiment_message.content}
Valuation Analysis: {valuation_message.content}
Risk Management: {risk_message.content}

Current Portfolio:
Available Balance: {portfolio['cash']:.2f}
Current Position: {portfolio['stock']} units  # Use 'stock' (not 'position')

Provide a comprehensive analysis and final decision following the specified format,
including detailed reasoning for each component and final decision logic."""
    }

    # Get completion from OpenRouter
    result = get_chat_completion([system_message, user_message])
    
    # 如果 API 调用失败，则使用默认值
    if result is None:
        result = '''
{
    "action": "neutral",
    "quantity": 0,
    "confidence": 0.5,
    "agent_signals": {
        "technical": {"signal": "neutral", "confidence": 0.5, "key_factors": ["API failure"]},
        "fundamental": {"signal": "neutral", "confidence": 0.5, "key_factors": ["API failure"]},
        "sentiment": {"signal": "neutral", "confidence": 0.5, "key_factors": ["API failure"]},
        "valuation": {"signal": "neutral", "confidence": 0.5, "key_factors": ["API failure"]}
    },
    "reasoning": {
        "weighted_signals": "API call failed",
        "risk_considerations": "Unable to assess risk",
        "final_decision": "Defaulting to neutral position due to API failure"
    }
}
        '''
    
    # ----- 预处理 result，去除 Markdown 代码块标记 -----
    result = result.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        result = "\n".join(lines).strip()
    # ----- 结束预处理 -----

    # ----- 后处理部分：根据当前资金和持仓计算满仓、半仓目标，并校验 action 与 quantity 是否对应 -----
    try:
        decision = json.loads(result)
    except Exception as e:
        decision = {"action": "neutral", "quantity": 0, "confidence": 0.5}
    
    # 获取当前价格（假设 state["data"]["prices"] 中最后一条数据有效）
    if "prices" in state["data"] and state["data"]["prices"]:
        current_price = state["data"]["prices"][-1]["close"]
    else:
        current_price = 1

    # 辅助函数：解析 confidence 数值
    def parse_conf(val):
        try:
            if isinstance(val, str):
                if "%" in val:
                    return float(val.strip().replace("%", "")) / 100.0
                else:
                    return float(val)
            else:
                return float(val)
        except:
            return 0.0

    conf = parse_conf(decision.get("confidence", 0))
    action = decision.get("action", "neutral").lower()

    # 根据当前资金和现有持仓，计算满仓目标：
    # 这里满仓的值定义为：可用现金买入的最大仓位加上当前正持仓（如果持仓为正，否则只考虑现金）
    if portfolio["stock"] > 0:
        long_full = int(portfolio["cash"] // current_price) + portfolio["stock"]
    else:
        long_full = int(portfolio["cash"] // current_price)
    long_half = long_full // 2 if long_full >= 2 else long_full

    if action == "long":
        # 校验返回的 quantity 是否为 long_full 或 long_half，否则覆盖
        if decision.get("quantity", None) not in (long_full, long_half):
            decision["quantity"] = long_full if conf >= 0.5 else long_half
    elif action == "short":
        # 对于 short，我们定义满仓数量同 long_full，然后目标仓位应为负值
        short_full = long_full  # 两边可参考同一数值，目标仓位是 -short_full
        short_half = short_full // 2 if short_full >= 2 else short_full
        if decision.get("quantity", None) not in (short_full, short_half):
            decision["quantity"] = short_full if conf >= 0.5 else short_half
    else:
        decision["quantity"] = 0

    result = json.dumps(decision)
    # ----- 后处理结束 -----

    # Create portfolio message
    message = HumanMessage(
        content=result,
        name="portfolio_management",
    )

    if show_reasoning:
        show_agent_reasoning(message.content, "Forex Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}