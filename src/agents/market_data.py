from langchain_core.messages import HumanMessage
from tools.openrouter_config import get_chat_completion
from agents.state import AgentState
from tools.api import get_macro_metrics, get_price_history
from datetime import datetime, timedelta
import pytz

def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing forex market data"""
    messages = state["messages"]
    data = state["data"]

    # Get current_date from state
    current_date = data.get("current_date") or data["end_date"]

    # 对于外汇市场，获取6个月的历史数据
    current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
    min_start_date = (current_date_obj - timedelta(days=180)).strftime('%Y-%m-%d')

    # 使用原始的start_date和min_start_date中较早的那个
    original_start_date = data["start_date"]
    start_date = min(original_start_date, min_start_date) if original_start_date else min_start_date

    # Get currency code
    currency = data["ticker"]  # 现在直接使用货币代码

    try:
        # 获取从start_date到current_date的所有价格数据
        prices = get_price_history(currency, start_date, current_date)

        # 获取当前货币的宏观数据
        macro_metrics = get_macro_metrics(currency)

        return {
            "messages": messages,
            "data": {
                **data,
                "prices": prices,
                "start_date": start_date,
                "end_date": current_date,
                "current_date": current_date,
                "macro_metrics": macro_metrics,  # 宏观经济指标数据
                "trading_session": get_trading_session(current_date_obj)  # 当前交易时段
            }
        }

    except Exception as e:
        print(f"Error in market_data_agent: {str(e)}")
        # 返回默认值
        return {
            "messages": messages,
            "data": {
                **data,
                "prices": [],
                "start_date": start_date,
                "end_date": current_date,
                "current_date": current_date,
                "macro_metrics": [{  # 默认的宏观指标结构
                    "interest_rate_differential": 0.0,
                    "inflation_differential": 0.0,
                    "gdp_growth_differential": 0.0,
                    "monetary_policy_stance": {
                        "currency": "unknown",
                        "usd": "unknown"
                    },
                    "economic_indicators": {
                        "currency": {
                            "unemployment": 0.0,
                            "trade_balance": 0.0
                        },
                        "usd": {
                            "unemployment": 0.0,
                            "trade_balance": 0.0
                        }
                    },
                    "data_timestamp": current_date,
                    "period": "current"
                }],
                "trading_session": "unknown"
            }
        }

def get_trading_session(date_obj: datetime) -> str:
    """
    根据时间判断当前交易时段
    使用UTC时间作为基准进行判断
    
    亚洲时段：00:00-08:00 UTC (晚上8点-早上4点 EST)
    欧洲时段：08:00-16:00 UTC (早上4点-下午12点 EST)
    美洲时段：16:00-24:00 UTC (下午12点-晚上8点 EST)
    """
    # 确保使用UTC时间
    if date_obj.tzinfo is None:
        date_obj = pytz.utc.localize(date_obj)
    elif date_obj.tzinfo != pytz.utc:
        date_obj = date_obj.astimezone(pytz.utc)
    
    hour = date_obj.hour
    
    if 0 <= hour < 8:
        return "Asian Session"
    elif 8 <= hour < 16:
        return "European Session"
    else:
        return "American Session"