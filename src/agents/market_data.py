from langchain_core.messages import HumanMessage
from tools.openrouter_config import get_chat_completion
from agents.state import AgentState
from tools.api import get_macro_metrics, get_market_data_history
from datetime import datetime, timedelta
import pytz
import pandas as pd

def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing forex market data"""
    # print('market_data_agent')
    # print('#'*50)
    messages = state["messages"]
    data = state["data"]

    # Get current_date from state
    time = data.get("time")


    # Get currency code
    currency = data["ticker"]  # 现在直接使用货币代码

    try:
        # 获取从start_date到current_date的所有价格数据
        prices = get_market_data_history(currency, time, num_min=1440)
        symbol = currency[:3]
        # 获取当前货币的宏观数据
        macro_metrics = get_macro_metrics(symbol)

        return {
            "messages": messages,
            "data": {
                **data,
                "prices": prices,
                "time": time,
                "macro_metrics": macro_metrics,  # 宏观经济指标数据
                "trading_session": get_trading_session(pd.to_datetime(time))  # 当前交易时段
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
                "time": time,
                "macro_metrics": [{  # 默认的宏观指标结构
                    "interest_rate_differential": 0.0,
                    "inflation_differential": 0.0,
                    "gdp_growth_differential": 0.0,
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
                    "data_timestamp": time,
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