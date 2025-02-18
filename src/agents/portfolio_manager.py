from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tools.openrouter_config import get_chat_completion
from agents.state import AgentState, show_agent_reasoning
import json
import ast

def portfolio_management_agent(state: AgentState):
    """Makes final forex trading decisions based on weighted analysis of multiple signals with granular position sizing"""
    # print('portfolio_management_agent')
    # print('#'*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    currency = state["data"]["ticker"][:3]
    data = state["data"]
    symbol = data["ticker"]
    time = data["time"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    
    # Get all agent messages
    messages = {
        "technical": next(msg for msg in state["messages"] if msg.name == "technical_analyst_agent"),
        "fundamental": next(msg for msg in state["messages"] if msg.name == "fundamentals_agent"),
        "sentiment": next(msg for msg in state["messages"] if msg.name == "sentiment_agent"),
        "valuation": next(msg for msg in state["messages"] if msg.name == "valuation_agent"),
        "risk": next(msg for msg in state["messages"] if msg.name == "risk_management_agent")
    }

    # Parse messages
    try:
        agent_data = {
            name: json.loads(msg.content) for name, msg in messages.items()
        }
    except:
        agent_data = {
            name: ast.literal_eval(msg.content) for name, msg in messages.items()
        }

    # Get max position size from risk management
    max_position = float(agent_data["risk"]["max_position_size"])

    system_message = {
        "role": "system",
        "content": f'''You are a forex portfolio manager making final trading decisions for {currency}.
        Your task is to determine position size (0-1) based on weighted analysis of multiple signals:

        SIGNAL WEIGHTS:
        - Sentiment Analysis (70%): Primary driver
        * event_similarity direction determines long/short
        * Consider base_sentiment alignment
        * Higher position for higher similarity/sentiment magnitude
        * Check confidence, importance, impact metrics
        * Reduce position for high uniqueness (repeated news)
        * Maximum position limited by risk management ({max_position})

        - Technical Analysis (15%): 
        * Validate sentiment direction
        * Increase position if aligned
        * Decrease if contrary

        - Fundamental Analysis (5%):
        * Further validation
        * Minor position adjustment

        - Valuation Analysis (5%):
        * Final validation
        * Minor position adjustment

        Position sizes must be in increments of 0.1 from 0 to 1, and cannot exceed max_position={max_position}

        Output Format:
        {{
            "action": "long/short/neutral",
            "quantity": "float 0-1 in 0.1 increments",
            "confidence": "float 0-1",
            "agent_signals": "signal summary",
            "reasoning": "decision logic"
        }}'''
    }

    user_message = {
        "role": "user",
        "content": f"""Analyze trading signals for {currency}:

        Sentiment Analysis ({agent_data['sentiment']}):
        - Check event_similarity for direction
        - Validate with base_sentiment
        - Consider uniqueness={agent_data['sentiment']['reasoning'].get('uniqueness', 'N/A')}
        - Review confidence/importance metrics

        Technical Analysis ({agent_data['technical']}):
        - Validate sentiment direction

        Fundamental ({agent_data['fundamental']}) & Valuation ({agent_data['valuation']}):
        - Final validation

        Risk Management:
        - Max position: {max_position:.2f}
        - Risk score: {agent_data['risk']['risk_score']:.1f}/10

        Determine position size (0-1 in 0.1 increments) based on signal analysis."""
    }

    # Get completion from OpenRouter
    result = get_chat_completion([system_message, user_message])
    
    if result is None:
        result = '''
        {
            "action": "neutral",
            "quantity": 0,
            "confidence": 0.5,
            "agent_signals": {
                "sentiment": {"signal": "neutral", "confidence": 0.5},
                "technical": {"signal": "neutral", "confidence": 0.5},
                "fundamental": {"signal": "neutral", "confidence": 0.5},
                "valuation": {"signal": "neutral", "confidence": 0.5}
            },
            "reasoning": "API failure - defaulting to neutral position"
        }'''
            
    # Clean up result
    result = result.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        result = "\n".join(lines).strip()

    # Validate and adjust position size
    try:
        decision = json.loads(result)
        quantity = float(decision.get("quantity", 0))
        # Round to nearest 0.1 and ensure within max_position
        quantity = round(min(quantity, max_position) * 10) / 10
        decision["quantity"] = quantity
        result = json.dumps(decision)
    except Exception as e:
        decision = {"action": "neutral", "quantity": 0, "confidence": 0.5}

    message = HumanMessage(
        content=result,
        name="portfolio_management",
    )

    if show_reasoning:
        show_agent_reasoning(message.content, "Forex Portfolio Management Agent", 'Portfolio', time, symbol, start_date, end_date)

    return {"messages": state["messages"] + [message]}