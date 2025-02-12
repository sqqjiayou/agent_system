from typing import Dict, Any, List
import pandas as pd
# import yfinance as yf
from datetime import datetime, timedelta
import warnings
import numpy as np
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 定义直接报价的货币列表（USD在后）
DIRECT_CURRENCIES = ['EUR', 'GBP', 'AUD', 'NZD']

def get_macro_metrics(currency: str) -> List[Dict[str, Any]]:
    """获取货币相对USD的宏观经济指标，包括最近两期数据"""
    try:
        # 获取当前和上期的宏观数据
        current_data = get_currency_macro_data(currency)
        prev_data = get_currency_macro_data(currency, is_previous=True)
        usd_data = get_currency_macro_data('USD')
        usd_prev_data = get_currency_macro_data('USD', is_previous=True)
        
        # 根据是否为直接报价调整指标正负
        multiplier = 1 if currency in DIRECT_CURRENCIES else -1
        
        # 计算当前期间的指标
        current_metrics = {
            "interest_rate_differential": multiplier * (current_data["interest_rate"] - usd_data["interest_rate"]),
            "inflation_differential": multiplier * (current_data["inflation_rate"] - usd_data["inflation_rate"]),
            "gdp_growth_differential": multiplier * (current_data["gdp_growth"] - usd_data["gdp_growth"]),
            "monetary_policy_stance": {
                "currency": current_data["monetary_policy"],
                "usd": usd_data["monetary_policy"]
            },
            "economic_indicators": {
                "currency": {
                    "unemployment": float(current_data["unemployment"]),
                    "trade_balance": float(current_data["trade_balance"])
                },
                "usd": {
                    "unemployment": float(usd_data["unemployment"]),
                    "trade_balance": float(usd_data["trade_balance"])
                }
            },
            "data_timestamp": datetime.now().strftime("%Y-%m-%d"),
            "period": "current"
        }
        
        # 计算上一期间的指标
        previous_metrics = {
            "interest_rate_differential": multiplier * (prev_data["interest_rate"] - usd_prev_data["interest_rate"]),
            "inflation_differential": multiplier * (prev_data["inflation_rate"] - usd_prev_data["inflation_rate"]),
            "gdp_growth_differential": multiplier * (prev_data["gdp_growth"] - usd_prev_data["gdp_growth"]),
            "monetary_policy_stance": {
                "currency": prev_data["monetary_policy"],
                "usd": usd_prev_data["monetary_policy"]
            },
            "economic_indicators": {
                "currency": {
                    "unemployment": float(prev_data["unemployment"]),
                    "trade_balance": float(prev_data["trade_balance"])
                },
                "usd": {
                    "unemployment": float(usd_prev_data["unemployment"]),
                    "trade_balance": float(usd_prev_data["trade_balance"])
                }
            },
            "data_timestamp": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "period": "previous"
        }
        
        return [current_metrics, previous_metrics]
        
    except Exception as e:
        print(f"Error getting macro metrics: {e}")
        return [get_default_macro_metrics("current"), get_default_macro_metrics("previous")]

def get_default_macro_metrics(period: str) -> Dict[str, Any]:
    """返回默认的宏观指标数据"""
    return {
        "interest_rate_differential": 0.0,
        "inflation_differential": 0.0,
        "gdp_growth_differential": 0.0,
        "monetary_policy_stance": {"currency": "unknown", "usd": "unknown"},
        "economic_indicators": {
            "currency": {"unemployment": 0.0, "trade_balance": 0.0},
            "usd": {"unemployment": 0.0, "trade_balance": 0.0}
        },
        "data_timestamp": None,
        "period": period
    }

def get_currency_macro_data(currency: str, is_previous: bool = False) -> Dict[str, Any]:
    """获取单个货币的宏观经济数据，包括当前和上期数据"""
    current_macro_data = {
        "USD": {
            "interest_rate": 5.50,  # Fed funds rate
            "inflation_rate": 3.2,
            "gdp_growth": 2.1,
            "monetary_policy": "tightening",
            "unemployment": 3.8,
            "trade_balance": -63.3
        },
        "EUR": {
            "interest_rate": 4.50,  # ECB rate
            "inflation_rate": 2.9,
            "gdp_growth": 0.1,
            "monetary_policy": "tightening",
            "unemployment": 6.5,
            "trade_balance": 23.0
        },
        "JPY": {
            "interest_rate": -0.1,  # BOJ rate
            "inflation_rate": 3.3,
            "gdp_growth": 1.2,
            "monetary_policy": "easing",
            "unemployment": 2.6,
            "trade_balance": -34.2
        },
        "GBP": {
            "interest_rate": 5.25,  # BOE rate
            "inflation_rate": 4.0,
            "gdp_growth": 0.3,
            "monetary_policy": "tightening",
            "unemployment": 4.2,
            "trade_balance": -15.8
        },
        "CNH": {  # 离岸人民币
            "interest_rate": 3.45,
            "inflation_rate": 0.1,
            "gdp_growth": 4.9,
            "monetary_policy": "neutral",
            "unemployment": 5.3,
            "trade_balance": 77.7
        },
        "CNY": {  # 在岸人民币
            "interest_rate": 3.45,
            "inflation_rate": 0.1,
            "gdp_growth": 4.9,
            "monetary_policy": "neutral",
            "unemployment": 5.3,
            "trade_balance": 77.7
        },
        "CHF": {
            "interest_rate": 1.75,  # SNB rate
            "inflation_rate": 1.7,
            "gdp_growth": 0.8,
            "monetary_policy": "neutral",
            "unemployment": 2.0,
            "trade_balance": 8.5
        },
        "CAD": {
            "interest_rate": 5.00,  # BOC rate
            "inflation_rate": 3.4,
            "gdp_growth": 1.1,
            "monetary_policy": "tightening",
            "unemployment": 5.8,
            "trade_balance": -2.8
        },
        "AUD": {
            "interest_rate": 4.35,  # RBA rate
            "inflation_rate": 4.1,
            "gdp_growth": 2.1,
            "monetary_policy": "tightening",
            "unemployment": 3.9,
            "trade_balance": 7.5
        },
        "NZD": {
            "interest_rate": 5.50,  # RBNZ rate
            "inflation_rate": 4.7,
            "gdp_growth": 1.6,
            "monetary_policy": "tightening",
            "unemployment": 3.9,
            "trade_balance": -0.9
        }
    }
    
    # 模拟上期数据（添加小幅度变化）
    if is_previous:
        previous_data = current_macro_data.get(currency, {}).copy()
        if previous_data:
            # 使用固定的随机种子确保结果可重现
            np.random.seed(42)
            for key in ["interest_rate", "inflation_rate", "gdp_growth", "unemployment", "trade_balance"]:
                if key in previous_data:
                    # 添加较小的随机变化（±5%）
                    previous_data[key] = previous_data[key] * (1 + np.random.uniform(-0.05, 0.05))
            return previous_data
    
    return current_macro_data.get(currency, {
        "interest_rate": 0.0,
        "inflation_rate": 0.0,
        "gdp_growth": 0.0,
        "monetary_policy": "unknown",
        "unemployment": 0.0,
        "trade_balance": 0.0
    })

def get_price_data(currency: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取外汇价格数据"""
    # Check if data exists in CSV file
    filename = f"data/{currency}_{start_date}_{end_date}.csv"
    try:
        if os.path.exists(filename):
            return pd.read_csv(filename, index_col="Date")
    except:
        pass

    try:
        # 确定正确的报价方式
        if currency in DIRECT_CURRENCIES:
            yf_symbol = f"{currency}USD=X"
        else:
            try:
                # 先尝试货币在前的方式
                yf_symbol = f"{currency}USD=X"
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period="1d")
                if df.empty:
                    raise Exception("Empty data")
            except:
                print(f"Warning: {currency} must be quoted with USD in front")
                yf_symbol = f"USD{currency}=X"
        
        # 转换日期格式
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start == end:
            end = start + timedelta(days=1)
            
        # 获取数据
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            print(f"Warning: No price data found for {currency}")
            return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])
        
        # 格式化数据
        df = df.reset_index()
        df["Date"] = df["Date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        
        # 重命名列
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        if "volume" not in df.columns or df["volume"].isnull().all():
            df["volume"] = 1000000
        
        df = df[["Date", "open", "high", "low", "close", "volume"]]
        df = df.set_index("Date")
        
        # 如果是USD在前的报价方式，需要反转价格
        if yf_symbol.startswith("USD"):
            # 反转OHLC价格
            for col in ['open', 'high', 'low', 'close']:
                df[col] = 1 / df[col]
            # 交换high和low
            df['high'], df['low'] = df['low'], df['high']

        # Save data to CSV
        try:
            os.makedirs("data", exist_ok=True)
            df.to_csv(filename)
        except:
            print(f"Warning: Failed to save data to {filename}")
        
        return df
        
    except Exception as e:
        print(f"Error in get_price_data for {currency}: {str(e)}")
        return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])

def get_price_history(currency: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """获取外汇历史价格数据"""
    try:
        # 确定正确的报价方式和是否需要反转
        if currency in DIRECT_CURRENCIES:
            yf_symbol = f"{currency}USD=X"
            needs_inversion = False
        else:
            # print(f"Warning: {currency} must be quoted with USD in front")
            yf_symbol = f"USD{currency}=X"
            needs_inversion = True

        if not end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        if not start_date:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # 构建文件名
        filename = f"data_history/{currency}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
        
        # 检查文件是否存在
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
        else:
            # 使用 yfinance 获取数据
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # 确保data_history文件夹存在
            os.makedirs('data_history', exist_ok=True)
            
            # 保存数据
            df.to_csv(filename)
            
        # 转换为所需格式
        prices = []
        for date, row in df.iterrows():
            price_data = {
                "time": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row.get("Volume", 1000000))
            }
            
            # 如果需要反转价格
            if needs_inversion:
                for key in ["open", "high", "low", "close"]:
                    price_data[key] = 1 / price_data[key]
                # 交换high和low
                price_data["high"], price_data["low"] = price_data["low"], price_data["high"]
            
            prices.append(price_data)
            
        return prices
        
    except Exception as e:
        print(f"Error in get_price_history: {e}")
        return []

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """将外汇价格列表转换为 DataFrame"""
    try:
        if not prices:
            return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])
            
        df = pd.DataFrame(prices)
        
        # 确保存在必要的列
        required_cols = ["time", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                if col == "volume":
                    df[col] = 1000000
                else:
                    df[col] = 0.0
                    
        # 转换日期列
        df["Date"] = pd.to_datetime(df["time"])
        df = df.drop("time", axis=1)
        df.set_index("Date", inplace=True)
        
        # 确保数值类型正确
        numeric_cols = ["open", "close", "high", "low", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # 按日期排序
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error in prices_to_df: {str(e)}")
        return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])
