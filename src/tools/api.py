from typing import Dict, Any, List
from datetime import datetime, timedelta
import warnings
import numpy as np
import requests
from pathlib import Path
# 忽略警告
warnings.filterwarnings('ignore')
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.language_models import LLM
from openai import OpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
from typing import List, Any
import os
import json
import logging
import time

class DeepSeek2(LLM):
    """Custom LLM class for DeepSeek API integration"""
    @property
    def _llm_type(self) -> str:
        return "DeepSeekllm"
        
    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=os.getenv("DS_API_KEY"),
                # base_url="https://api.openai.com/v1"
                #base_url="https://api.deepseek.com",
                base_url="https://tbnx.plus7.plus/v1"
            )
            
            formatted_prompt = prompt + "\nIMPORTANT: Return only a single integer number, nothing else."

            completion = client.chat.completions.create(
                model="deepseek-chat",
                # model="deepseek-reasoner", 
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in DeepSeek _call: {e}", exc_info=True)
            raise

class DeepSeek(LLM):
    """Custom LLM class for DeepSeek API integration"""
    @property
    def _llm_type(self) -> str:
        return "DeepSeekllm"
        
    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=os.getenv("DS_API_KEY"),
                #base_url="https://api.moonshot.cn/v1",
                # base_url="https://api.openai.com/v1"
                base_url="https://tbnx.plus7.plus/v1"

            )
            # Add explicit JSON formatting instruction
            formatted_prompt = prompt + """

            IMPORTANT: You must return ONLY a JSON object with the specified metrics, without any comments or explanations.
            Your response must be EXACTLY in this format (do not add any comments or explanations):
            
            {
            "similarity_score": 0.8,
            "sentiment_score": 0.5,
            "sentiment_class": 1,
            "confidence": 0.9,
            "relevance": 0.7,
            "impact_length": "month",
            "importance": 0.6
            }

            Do not include any explanations or comments in the JSON. The values should be numbers (except impact_length which should be a string) without any additional text."""

            completion = client.chat.completions.create(
                model="deepseek-chat",
                # model="deepseek-reasoner", 
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in DeepSeek _call: {e}", exc_info=True)
            raise

def create_news_analysis_schema(currency: str) -> List[ResponseSchema]:
    """
    Create schema for news analysis metrics
    
    Args:
        currency: Target currency or currency pair for analysis
    
    Returns:
        List of ResponseSchema objects defining expected output format
    """
    return [
        ResponseSchema(
            name="similarity_score",
            description="""Float between -1 and 1 representing similarity to event:
            1: Highly similar with same direction
            -1: Highly similar with opposite direction
            0: Unrelated
            Use intermediate values for partial similarity""",
            type="float"
        ),
        ResponseSchema(
            name="sentiment_score",
            description=f"""Float between -1 and 1 indicating sentiment impact on {currency}.
            For currency pairs, positive means first currency strengthens vs second
            (e.g., USDCNH up means +USD,-CNH)""",
            type="float"
        ),
        ResponseSchema(
            name="sentiment_class",
            description=f"Must be exactly: 1 (positive), 0 (neutral), or -1 (negative) for {currency}",
            type="integer"
        ),
        ResponseSchema(
            name="confidence",
            description="Float between 0-1 measuring classification confidence, higher for more certain assessments",
            type="float"
        ),
        ResponseSchema(
            name="relevance",
            description=f"Float between 0-1 measuring direct relevance to {currency}, higher for more directly relevant news",
            type="float"
        ),
        ResponseSchema(
            name="impact_length",
            description="One of: 'day','week','month','quarter','year','decade' indicating expected impact duration",
            type="string"
        ),
        ResponseSchema(
            name="importance",
            description=f"Float between 0-1 measuring potential market impact magnitude on {currency}",
            type="float"
        )
    ]

def analyze_news_metrics_with_llm(symbol: str, start_date: str, end_date: str, news_row: pd.Series, event_prompt: str) -> pd.Series:
    """
    Analyze news content using LLM to generate multiple metrics
    
    Args:
        symbol: Currency pair to analyze (e.g., 'EURUSD')
        start_date: Analysis start date
        end_date: Analysis end date
        news_row: Single row from news DataFrame containing title and text
        event_prompt: Reference event for similarity comparison
    
    Returns:
        pd.Series containing all scoring metrics
    """
    # Check existing analysis
    output_path = f"news_data/{symbol}_{start_date}_{end_date}.csv"
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        matched_row = existing_df[existing_df['publishedDate'] == news_row['publishedDate']]
        required_cols = ['similarity_score', 'sentiment_score', 'sentiment_class', 
                        'confidence', 'relevance', 'impact_length', 'importance']
        if not matched_row.empty and all(col in matched_row.columns for col in required_cols):
            return matched_row[required_cols].iloc[0]

    # Initialize LLM
    llm = DeepSeek()
    
    # Create parser and prompt
    output_parser = StructuredOutputParser.from_response_schemas(create_news_analysis_schema(symbol))
    format_instructions = output_parser.get_format_instructions()
    
    template = """Analyze the following news content considering multiple aspects:

    Title: {title}
    Content: {text}
    Reference Event: {event_prompt}
    Currency/Pair to analyze: {symbol}

    Requirements:
    1. Similarity Score (-1 to 1):
    - Compare content similarity to reference event
    - 1: Highly similar content with same directional impact
    - -1: Highly similar content with opposite directional impact
    - 0: Unrelated content
    - Use intermediate values only for partial similarity

    2. Currency Impact Analysis for {symbol}:
    - sentiment_score (-1 to 1): Direct market impact strength and direction
        For currency pairs: +1 means strong positive impact on first currency
        (e.g., for EURUSD: +1 means strong EUR strength/USD weakness)
        -1 means strong negative impact on first currency
        0 means neutral impact
    
    - sentiment_class: Strict classification of impact
        Must be exactly: +1 (strengthens first currency)
        0 (neutral)
        -1 (weakens first currency)

    3. Additional Metrics:
    - confidence (0-1): Certainty level of the impact assessment
        1.0: Extremely clear and certain impact
        0.5: Moderate certainty
        0.0: Highly uncertain impact
    
    - relevance (0-1): Direct connection to {symbol}
        1.0: Directly about {symbol} or major driving factors
        0.5: Indirectly related
        0.0: Minimal connection
    
    - impact_length: Expected duration of market impact
        Choose one: 'day','week','month','quarter','year','decade'
    
    - importance (0-1): Potential magnitude of market impact
        1.0: Major market-moving event
        0.5: Moderate market impact
        0.0: Minimal market impact

    Provide scores following these format instructions:
    {format_instructions}"""

    template = template + """

    IMPORTANT: Your response must be EXACTLY in the specified JSON format without any comments or explanations.
    Do not add any explanatory text or comments to the JSON values.
    """
    
    prompt = PromptTemplate(
        input_variables=["title", "text", "event_prompt", "symbol"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )

    try:
        # 生成prompt
        prompt_value = prompt.format(
            title=news_row['title'],
            text=news_row['text'],
            event_prompt=event_prompt,
            symbol=symbol
        )
        
        # 使用invoke而不是直接调用
        llm_output = llm.invoke(prompt_value)
        
        try:
            # 尝试解析输出
            parsed_output = output_parser.parse(llm_output)
        except Exception as parse_error:
            logging.error(f"Initial parse failed: {parse_error}")
            # 清理输出并重试
            cleaned_output = llm_output.strip()
            if '```' in cleaned_output:
                # 提取markdown代码块中的内容
                start_idx = cleaned_output.find('```') + 3
                end_idx = cleaned_output.rfind('```')
                if start_idx < end_idx:
                    # 移除可能的language标识符（如 ```json）
                    json_content = cleaned_output[start_idx:end_idx].strip()
                    if json_content.startswith('json'):
                        json_content = json_content[4:].strip()
                    cleaned_output = json_content            
            try:
                # 移除任何可能的注释（以//开头的行）
                cleaned_lines = [line.split('//')[0].strip() for line in cleaned_output.split('\n')]
                cleaned_output = ''.join(cleaned_lines)
                parsed_output = json.loads(cleaned_output)
            except json.JSONDecodeError as json_error:
                error_msg = f"Failed to parse LLM output: {json_error}\nOutput was: {cleaned_output}"
                logging.error(error_msg)
                raise Exception(error_msg)
        
        try:
            # Prepare metrics from LLM output
            metrics_to_add = {
                'similarity_score': parsed_output['similarity_score'],
                'sentiment_score': parsed_output['sentiment_score'],
                'sentiment_class': parsed_output['sentiment_class'],
                'confidence': parsed_output['confidence'],
                'relevance': parsed_output['relevance'],
                'impact_length': parsed_output['impact_length'],
                'importance': parsed_output['importance']
            }
            
            # Ensure news_data directory exists
            os.makedirs('news_data', exist_ok=True)
            
            # Read existing CSV or create new one
            if os.path.exists(output_path):
                df_existing = pd.read_csv(output_path)
                
                # Add missing columns if needed
                for col in metrics_to_add.keys():
                    if col not in df_existing.columns:
                        df_existing[col] = None
                
                # Update metrics for matching publishedDate
                df_existing['publishedDate'] = pd.to_datetime(df_existing['publishedDate'])
                # df_existing['publishedDate'] = df_existing['publishedDate'].dt.tz_localize(None) 
                mask = df_existing['publishedDate'] == news_row['publishedDate']
                if any(mask):
                    # Update existing row
                    for col, value in metrics_to_add.items():
                        df_existing.loc[mask, col] = value
                else:
                    # Add new row if publishedDate not found
                    new_row = news_row.copy()
                    new_row.update(metrics_to_add)
                    df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
                
                # Save updated dataframe
                df_existing.to_csv(output_path, index=False)
            else:
                # Create new dataframe with all required columns
                news_row_with_metrics = news_row.copy()
                news_row_with_metrics.update(metrics_to_add)
                pd.DataFrame([news_row_with_metrics]).to_csv(output_path, index=False)
            
            return pd.Series(metrics_to_add)
            
        except Exception as e:
            logging.error(f"Error saving news metrics: {e}", exc_info=True)
            raise
        
    except Exception as e:
        logging.error(f"Error analyzing news metrics: {e}", exc_info=True)
        raise

def get_forex_news(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch forex news for a specified symbol within a date range.
    First checks if local data covers the requested date range, then falls back to API if needed.
    
    Args:
        symbol: Trading pair symbol (e.g. "EURUSD")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        
    Returns:
        pandas DataFrame containing forex news data, or None if request fails
    """
    
    # Validate date formats
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD format")
        return None

    # Create news_data directory if it doesn't exist
    news_dir = Path('news_data')
    news_dir.mkdir(exist_ok=True)
    
    # Check for existing files with matching symbol
    existing_files = list(news_dir.glob(f"{symbol}_*.csv"))
    
    use_local_data = False
    local_file_path = None
    
    for file_path in existing_files:
        # Extract dates from filename
        filename = file_path.stem
        _, file_start, file_end = filename.split('_')
        
        file_start_dt = datetime.strptime(file_start, "%Y-%m-%d")
        file_end_dt = datetime.strptime(file_end, "%Y-%m-%d")
        
        # Check if local file covers the requested date range
        if file_start_dt <= start_dt and file_end_dt >= end_dt:
            use_local_data = True
            local_file_path = file_path
            break
    
    try:
        # Use local data if available and covers the date range
        if use_local_data:
            print(f"Found local news data covering the requested period for {symbol}")
            df = pd.read_csv(local_file_path)
            df['publishedDate'] = pd.to_datetime(df['publishedDate'])
            df = df[df['publishedDate'].between(start_dt, end_dt)]
            return df
            
        # If local data not suitable, fetch from API using pagination
        all_data = []
        current_dt = end_dt
        
        while current_dt >= start_dt:
            to_dt = current_dt
            from_dt = current_dt - timedelta(days=1)
            
            # Adjust from_dt if it's earlier than start_dt
            if from_dt < start_dt:
                from_dt = start_dt
                
            params = {
                "symbol": symbol,
                "from": from_dt.strftime("%Y-%m-%d"),
                "to": to_dt.strftime("%Y-%m-%d"),
                "apikey": os.getenv("FMP_API_KEY")
            }
            
            base_url = "https://financialmodelingprep.com/api/v4/forex_news"
            response = requests.get(url=base_url, params=params)
            
            if response.status_code != 200:
                print(f"API request failed for period {from_dt} to {to_dt}: Status {response.status_code}")
                print(f"Response: {response.text}")
                continue
                
            data = response.json()
            if data:
                all_data.extend(data)
                
            print(f"Downloaded data for {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}")
            
            current_dt = current_dt - timedelta(days=2)
            time.sleep(1)  # Rate limiting pause
            
        # Check if any data was collected
        if not all_data:
            print(f"No news data found for {symbol} between {start_date} and {end_date}")
            return None
            
        # Convert to DataFrame and sort
        df = pd.DataFrame(all_data)
        df['publishedDate'] = pd.to_datetime(df['publishedDate'])
        df['publishedDate'] = df['publishedDate'].dt.tz_localize(None) 
        df.sort_values(by='publishedDate', ascending=True, inplace=True)
        # Save to local storage
        new_file_name = f"{symbol}_{start_date}_{end_date}.csv"
        new_file_path = news_dir / new_file_name
        df.to_csv(new_file_path, index=False)
        print(f"Saved new data to {new_file_name}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

# Usage example
def test_forex_news():
    symbol = "EURUSD"
    start_date = "2025-02-01"
    end_date = "2025-02-10"
    
    news_df = get_forex_news(symbol, start_date, end_date)
    if news_df is not None:
        print(news_df)

import pandas as pd
from typing import Optional
import os
import glob

import pandas as pd
from typing import Optional, Tuple
import os
import glob

def get_forex_news_history(symbol: str, start_date: str, end_date: str, time: str, num_hour: int) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Get historical forex news data within specified time range.
    
    Args:
        symbol: Currency pair symbol (e.g., 'EURUSD')
        start_date: Start date of the data file
        end_date: End date of the data file
        time: Target time point
        num_hour: Number of hours to look back
    
    Returns:
        Tuple containing:
        - str or None: Path of the suitable file
        - DataFrame or None: DataFrame containing news within the time range
    """
    # Convert input dates to datetime for comparison
    try:
        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)
    except Exception as e:
        print(f"Error: Invalid date format for input dates: {e}")
        return None, None

    # Find all files starting with the symbol in news_data directory
    file_pattern = os.path.join('news_data', f"{symbol}_*.csv")
    available_files = glob.glob(file_pattern)
    
    if not available_files:
        print(f"Error: No files found for symbol {symbol}")
        return None, None
    
    # Find suitable file by checking date range in filename
    suitable_file = None
    for file_path in available_files:
        # Extract dates from filename
        try:
            filename = os.path.basename(file_path)
            file_dates = filename.replace(f"{symbol}_", "").replace(".csv", "").split("_")
            file_start = pd.to_datetime(file_dates[0])
            file_end = pd.to_datetime(file_dates[1])
            
            # Check if target date range is within file's date range
            if file_start <= target_start and file_end >= target_end:
                suitable_file = file_path
                break
        except Exception as e:
            print(f"Warning: Skipping file {filename} due to invalid format: {e}")
            continue
    
    if suitable_file is None:
        print(f"Error: No suitable file found containing date range from {start_date} to {end_date}")
        return None, None
    
    try:
        # Read CSV file
        df = pd.read_csv(suitable_file)
        
        if df.empty:
            print(f"Error: No data in file: {suitable_file}")
            return suitable_file, None
            
        # Convert publishedDate column to datetime
        try:
            df['publishedDate'] = pd.to_datetime(df['publishedDate'])
        except Exception as e:
            print(f"Error: Failed to convert publishedDate to datetime format: {e}")
            return suitable_file, None
        
        # Convert input time to datetime
        try:
            target_time = pd.to_datetime(time)
        except Exception as e:
            print(f"Error: Invalid time format for input time {time}: {e}")
            return suitable_file, None
            
        # Calculate start time
        start_time = target_time - pd.Timedelta(hours=num_hour)
        
        # Get the earliest time in the dataset
        earliest_time = df['publishedDate'].min()
        
        # If start_time is earlier than the earliest available data,
        # use the earliest available time
        if start_time < earliest_time:
            print(f"Warning: Requested start time ({start_time}) is earlier than available data. Using earliest available time: {earliest_time}")
            start_time = earliest_time
        
        # Filter data within time range
        mask = (df['publishedDate'] >= start_time) & (df['publishedDate'] <= target_time)
        filtered_df = df[mask]
        
        if filtered_df.empty:
            print(f"Warning: No data found between {start_time} and {target_time}")
            return suitable_file, None
        
        # Sort by time
        filtered_df = filtered_df.sort_values('publishedDate')
        
        print(f"Successfully retrieved {len(filtered_df)} news entries from file: {os.path.basename(suitable_file)}")
        return suitable_file, filtered_df
    
    except Exception as e:
        print(f"Error: Unexpected error while processing file: {e}")
        return suitable_file, None
    
def get_market_data(symbol: str, timestamp: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Get market data for the nearest minute greater than or equal to the specified timestamp
    
    Args:
        symbol: Trading pair symbol (e.g. "EURUSD")
        timestamp: Timestamp string with seconds (format: "YYYY-MM-DD HH:MM:SS")
        api_key: API key
    
    Returns:
        pandas Series containing market data, or None if no data found
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Convert input time to datetime and round up to next minute
    # timestamp = str(timestamp).replace("T", " ")
    # 移除毫秒和时区信息
    # clean_timestamp = timestamp.split('.')[0]
    dt = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
    if dt.second > 0:
        dt = dt + timedelta(minutes=1)
    target_minute = dt.replace(second=0, microsecond=0)
    date_str = target_minute.strftime("%Y-%m-%d")
    
    # Construct filename
    file_name = f"{symbol}_{date_str}.csv"
    file_path = data_dir / file_name
    
    try:
        # Try to read data from local file
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Found local data for {symbol} on {date_str}")
        else:
            # If local file doesn't exist, fetch from API
            base_url = "https://financialmodelingprep.com/api/v3/historical-chart/1min"
            params = {
                "apikey": os.getenv("FMP_API_KEY"),
                "from": date_str,
                "to": date_str
            }
            url = f"{base_url}/{symbol}"
            
            response = requests.get(url=url, params=params)
            if response.status_code != 200:
                print(f"API request failed: HTTP Status Code {response.status_code}")
                return None, None
                
            df = pd.DataFrame(response.json())
            if df.empty:
                print(f"Warning: No trading data available for {symbol} on {date_str}")
                return None, None
                
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values(by='date', ascending=True, inplace=True)    
            # Save to local storage
            df.to_csv(file_path, index=False)
        
        # Find the nearest data point after target time
        mask = df['date'] >= target_minute
        if not mask.any():
            print(f"No data found after {target_minute}")
            return None, None
            
        return df[mask].iloc[0], df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Usage example
def test_market_data():
    symbol = "CNHUSD"
    timestamps = [
        "2025-01-01 14:30:45",
        "2025-01-01 14:35:20",
        "2025-01-01 14:40:15"
    ]

    for timestamp in timestamps:
        print(f"\nQuerying timestamp: {timestamp}")
        result = get_market_data(symbol, timestamp)
        if result is not None:
            print(f"Market data:\n{result}")


def get_market_data_history(symbol: str, timestamp: str, num_min: int) -> pd.DataFrame:
    """
    Get historical market data for specified number of minutes before the timestamp
    
    Args:
        symbol: Trading pair symbol (e.g. "EURUSD")
        timestamp: Timestamp string with seconds (format: "YYYY-MM-DD HH:MM:SS")
        num_min: Number of minutes of historical data to retrieve
    
    Returns:
        pandas DataFrame containing historical market data, or None if no data found
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Convert input time to datetime and round down to minute
    end_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(minutes=num_min)
    
    # Construct filename
    file_name = f"{symbol}_{start_dt.strftime('%Y-%m-%d')}_{end_dt.strftime('%Y-%m-%d')}.csv"
    file_path = data_dir / file_name
    
    try:
        # Try to read from local file first
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Found local data for {symbol} from {start_dt} to {end_dt}")
            
            # Verify if local data covers the required range
            if df['date'].min() <= start_dt and df['date'].max() >= end_dt:
                mask = df['date'].between(start_dt, end_dt)
                return df[mask].sort_values(by='date', ascending=True)
        
        # If local file doesn't exist or doesn't cover the range, fetch from API
        all_data = []
        current_dt = end_dt
        
        while current_dt >= start_dt:
            to_date = current_dt.strftime("%Y-%m-%d")
            from_date = (current_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            
            base_url = "https://financialmodelingprep.com/api/v3/historical-chart/1min"
            params = {
                "apikey": os.getenv("FMP_API_KEY"),
                "from": from_date,
                "to": to_date
            }
            url = f"{base_url}/{symbol}"
            
            response = requests.get(url=url, params=params)
            if response.status_code != 200:
                print(f"API request failed for period {from_date} to {to_date}: Status {response.status_code}")
                current_dt = current_dt - timedelta(days=2)
                time.sleep(1)
                continue
                
            data = response.json()
            if data:
                all_data.extend(data)
                
            print(f"Downloaded data for {from_date} to {to_date}")
            current_dt = current_dt - timedelta(days=2)
            time.sleep(1)  # Rate limiting pause
            
        if not all_data:
            print(f"No market data found for {symbol} between {start_dt} and {end_dt}")
            return None
            
        # Convert to DataFrame and process
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', ascending=True, inplace=True)

        # Save to local storage
        df.to_csv(file_path, index=False)

        # Filter for required time range
        mask = df['date'].between(start_dt, end_dt)
        result_df = df[mask].copy()
        
        return result_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    

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
            "interest_rate": 4.5,  # Fed Interest Rate Decision (30/1)
            "inflation_rate": 3.2, # Core Inflation Rate YoY DEC
            "gdp_growth": 3.1, # GDP Growth Rate QoQ Final Q3
            "unemployment": 4.1, # Unemployment Rate DEC
            "trade_balance": -98.4 # Balance of Trade DEC (in billion)
        },
        "EUR": {
            "interest_rate": 2.9,  # ECB Interest Rate Decision (30/1)
            "inflation_rate": 2.4, #Inflation Rate YoY Flash DEC
            "gdp_growth": 0.9, # GDP Growth Rate YoY 2nd Est Q3
            "unemployment": 6.3, # Unemployment Rate DEC
            "trade_balance": 16.4 # Balance of Trade Nov (in billion) 
        },
        "JPY": {
            "interest_rate": 0.5,  # BoJ Interest Rate Decision (24/1)
            "inflation_rate": 3.6, # Inflation Rate YoY DEC
            "gdp_growth": 1.2,  # GDP Growth Annualized Final Q3
            "unemployment": 2.4, # Unemployment Rate DEC
            "trade_balance": 130.9 # Balance of Trade DEC (in billion)
        },
        "GBP": {
            "interest_rate": 4.5,  # BoE Interest Rate Decision (6/2)
            "inflation_rate": 2.5, # Inflation Rate YoY DEC
            "gdp_growth": 0.9, # GDP Growth Rate YoY Final Q3
            "unemployment": 4.4, # Unemployment Rate Nov 
            "trade_balance": -2.82 # Balance of Trade DEC (in billion)
        },
        "CNH": {  # 离岸人民币
            "interest_rate": 3.1, # Loan Prime Rate 1Y (20/1)
            "inflation_rate": 0.1, #Inflation Rate YoY DEC
            "gdp_growth": 5.4, # GDP Growth Rate YoY Q4
            "unemployment": 5.1, # Unemployment Rate DEC
            "trade_balance": 104.84 # Balance of Trade DEC (in billion)
        },
        "CNY": {  # 在岸人民币
            "interest_rate": 3.1, # Loan Prime Rate 1Y (20/1)
            "inflation_rate": 0.1, #Inflation Rate YoY DEC
            "gdp_growth": 5.4, # GDP Growth Rate YoY Q4
            "unemployment": 5.1, # Unemployment Rate DEC
            "trade_balance": 104.84 # Balance of Trade DEC (in billion)
        },
        "CHF": {
            "interest_rate": 0.5,  #SNB Interest Rate Decision (12/12/2024)
            "inflation_rate": 0.7, # Inflation rate YoY DEC
            "gdp_growth": 2, # GDP Growth Rate YoY Q3
            "unemployment": 2.8, # Unemployment Rate DEC
            "trade_balance": 4.3 # Balance of Trade DEC (in billion)
        },
        "CAD": {
            "interest_rate": 3,  # BoC Interest Rate Decision (29/1)
            "inflation_rate": 1.8, # Inflation Rate YoY DEC
            "gdp_growth": 1, # GDP Growth Rate Annualized Q3
            "unemployment": 6.7, # Unemployment Rate DEC
            "trade_balance": 0.71 # Balance of Trade DEC (in billion)
        },
        "AUD": {
            "interest_rate": 4.35,  # RBA Interest Rate Decision (10/12/2024)
            "inflation_rate": 2.4, # Inflation Rate YoY Q4
            "gdp_growth": 0.8, # GDP Growth Rate YoY Q3
            "unemployment": 4, # Unemployment Rate DEC
            "trade_balance": 5.085 # Balance of Trade DEC (in billion)
        },
        "NZD": {
            "interest_rate": 4.25,  # RBNZ Interest Rate Decision (27/11)
            "inflation_rate": 2.2,  # Inflation Rate YoY Q4
            "gdp_growth": -1.5, # GDP Growth Rate YoY Q3
            "unemployment": 5.1, # Unemployment Rate Q4
            "trade_balance": 0.219 # Balance of Trade DEC (in billion)
        }
    }
    
    previous_macro_data = {
        "USD": {
            "interest_rate": 4.5,  # Fed Interest Rate Decision
            "inflation_rate": 3.3, # Core Inflation Rate YoY Nov
            "gdp_growth": 3, # GDP Growth Rate QoQ Final Q2
            "unemployment": 4.2, # Unemployment Rate Nov
            "trade_balance": -78.9 # Balance of Trade Nov (in billion)
        },
        "EUR": {
            "interest_rate": 3.15,  # ECB Interest Rate Decision 
            "inflation_rate": 2.2, #Inflation Rate YoY Flash Nov
            "gdp_growth": 0.6, # GDP Growth Rate YoY 2nd Est Q2
            "unemployment": 6.2, # Unemployment Rate Nov
            "trade_balance": 8.6 # Balance of Trade Oct (in billion) 
        },
        "JPY": {
            "interest_rate": 0.25,  # BoJ Interest Rate Decision
            "inflation_rate": 2.9, # Inflation Rate YoY Nov
            "gdp_growth": 2.2, #  GDP Growth Annualized Final Q2
            "unemployment": 2.5, # Unemployment Rate DEC
            "trade_balance": -110.3 # # Balance of Trade Nov (in billion)
        },
        "GBP": {
            "interest_rate": 4.75,  # BoE Interest Rate Decision
            "inflation_rate": 2.6,  # Inflation Rate YoY Nov
            "gdp_growth": 0.7, # GDP Growth Rate YoY Final Q2
            "unemployment": 4.3, # unemployment Rate Oct
            "trade_balance": -4.35 # Balance of Trade Nov (in billion)
        },
        "CNH": {  # 离岸人民币
            "interest_rate": 3.1, # Loan Prime Rate 1Y 
            "inflation_rate": 0.2,  #Inflation Rate YoY Nov
            "gdp_growth": 4.6,
            "unemployment": 5, # Unemployment Rate Nov
            "trade_balance": 97.44 # Balance of Trade Nov (in billion)
        },
        "CNY": {  # 在岸人民币
            "interest_rate": 3.1, # Loan Prime Rate 1Y 
            "inflation_rate": 0.2, #Inflation Rate YoY Nov
            "gdp_growth": 4.6,
            "unemployment": 5, # Unemployment Rate Nov
            "trade_balance": 97.44 # Balance of Trade Nov (in billion)
        },
        "CHF": {
            "interest_rate": 1,  #SNB Interest Rate Decision
            "inflation_rate": 0.7, # Inflation rate YoY Nov
            "gdp_growth": 1.5, # GDP Growth Rate YoY Q2
            "unemployment": 2.6, # Unemployment Rate Nov
            "trade_balance": 4.6 # Balance of Trade Nov (in billion)
        },
        "CAD": {
            "interest_rate": 3.25,  # BoC Interest Rate Decision
            "inflation_rate": 1.9,  # Inflation Rate YoY Nov
            "gdp_growth": 2.2,  # GDP Growth Rate Annualized Q2
            "unemployment": 6.8, # Unemployment Rate Nov
            "trade_balance": -0.99 # Balance of Trade Nov (in billion)
        },
        "AUD": {
            "interest_rate": 4.35,  # RBA Interest Rate Decision
            "inflation_rate": 2.8,  # Inflation Rate YoY Q4
            "gdp_growth": 1, # GDP Growth Rate YoY Q2
            "unemployment": 3.9, #Unemployment Rate Nov
            "trade_balance": 6.792 # Balance of Trade Nov (in billion)
        },
        "NZD": {
            "interest_rate": 4.7,  # RBNZ Interest Rate Decision
            "inflation_rate": 2.2, # Inflation Rate YoY Q3
            "gdp_growth": -1.1, # GDP Growth Rate YoY Q2
            "unemployment": 4.8, #  Unemployment Rate Q3
            "trade_balance": -0.435 # # Balance of Trade Nov (in billion)
        }
    }

    # 上期数据
    if is_previous:
        return previous_macro_data.get(currency, {
        "interest_rate": 0.0,
        "inflation_rate": 0.0,
        "gdp_growth": 0.0,
        "monetary_policy": "unknown",
        "unemployment": 0.0,
        "trade_balance": 0.0
    })
    
    return current_macro_data.get(currency, {
        "interest_rate": 0.0,
        "inflation_rate": 0.0,
        "gdp_growth": 0.0,
        "monetary_policy": "unknown",
        "unemployment": 0.0,
        "trade_balance": 0.0
    })

if __name__ == "__main__":
    # test_market_data()
    test_forex_news()
