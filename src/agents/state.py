from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
import json
from datetime import datetime
import pandas as pd
import csv
from filelock import FileLock, Timeout  # 需要先 pip install filelock
import shutil
import os

# 这些辅助函数保持不变
def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """合并两个字典，支持深度合并"""
    merged = a.copy()
    for key, value in b.items():
        if (
            key in merged 
            and isinstance(merged[key], dict) 
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

class AgentState(TypedDict):
    """代理状态类型定义"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]

def format_number(value: float) -> str:
    """Format number output optimized for forex data
    
    Args:
        value: Number to format
    Returns:
        Formatted string with appropriate decimal places
    """
    if abs(value) >= 1:
        return f"{value:,.4f}"
    else:
        return f"{value:.4f}"

def convert_to_serializable(obj):
    """Convert object to serializable format with special handling for forex data
    
    Args:
        obj: Object to be converted to serializable format
    Returns:
        Serializable version of the input object
    """
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (int, float)):
        if isinstance(obj, float):
            return format_number(obj)
        return obj
    elif isinstance(obj, (bool, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        converted_dict = {}
        for key, value in obj.items():
            forex_specific_keys = {
                'price', 'rate', 'stop_loss', 'take_profit',
                'pip_value', 'position_size', 'max_position_size',
                'volatility', 'pip_volatility'
            }
            
            if key in forex_specific_keys and isinstance(value, (int, float)):
                converted_dict[key] = format_number(value)
            else:
                converted_dict[key] = convert_to_serializable(value)
        return converted_dict
    else:
        return str(obj)

def process_value_for_dataframe(value):
    """Process value to ensure it can be stored in DataFrame
    
    Args:
        value: Value to be processed
    Returns:
        String representation of value if it's a dict/list, otherwise original value
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value

def show_agent_reasoning(output, agent_name, agent, time, symbol, start_date, end_date):
    """Display agent reasoning process and update news data with agent outputs
    
    Args:
        output: Agent output to display
        agent_name: Name of the agent 
        agent: Agent identifier
        time: Timestamp to match in news data
        symbol: Currency pair symbol
        start_date: Start date for news data
        end_date: End date for news data
    """
    
    try:
        # Define file paths
        original_news_file = f"{symbol}_{start_date}_{end_date}.csv"
        agent_news_file = f"Agent_{symbol}_{start_date}_{end_date}.csv"
        lock_file = f"results/{agent_news_file}.lock"

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Check if agent news file exists, if not create a copy from original news data
        if not os.path.exists(f"results/{agent_news_file}"):
            try:
                original_df = pd.read_csv(f"news_data/{original_news_file}")
                original_cols = [col for col in original_df.columns if not col.startswith('Unnamed')]
                original_df = original_df[original_cols]
                original_df.to_csv(f"results/{agent_news_file}", index=False)
            except Exception as e:
                print(f"Error copying file: {e}")
                return

        # Use file lock for thread-safe operations
        lock = FileLock(lock_file, timeout=10)
        
        try:
            with lock:
                if not os.path.exists(f"results/{agent_news_file}"):
                    print(f"Error: {agent_news_file} not found after copy")
                    return
                    
                try:
                    df = pd.read_csv(f"results/{agent_news_file}")
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                except pd.errors.EmptyDataError:
                    print("Error: CSV file is empty")
                    return
                except Exception as e:
                    print(f"Error reading CSV: {e}")
                    return

                if df.empty:
                    print("Error: Dataframe is empty")
                    return

                # Process output and update dataframe
                if isinstance(output, (dict, list)):
                    serializable_output = convert_to_serializable(output)
                    formatted_output = json.dumps(serializable_output, indent=2)
                    formatted_dict = json.loads(formatted_output)
                    
                    if isinstance(formatted_dict, dict):
                        df['publishedDate'] = pd.to_datetime(df['publishedDate'])
                        time = pd.to_datetime(time)
                        
                        for key, value in formatted_dict.items():
                            col_name = f"{agent}_{key}"
                            if col_name not in df.columns:
                                df[col_name] = ""
                            mask = df['publishedDate'] == time
                            # Process value before assigning to DataFrame
                            processed_value = process_value_for_dataframe(value)
                            df.loc[mask, col_name] = processed_value
                        
                        df['publishedDate'] = df['publishedDate'].dt.strftime('%Y-%m-%d %H:%M')
                        df.to_csv(f"results/{agent_news_file}", index=False)
                            
                        print(f"\n{'=' * 10} {agent} Agent Decision Details {'=' * 10}")
                        print(formatted_output)
                    else:
                        print(f"\n{'=' * 10} {agent} Agent Decision Details {'=' * 10}")
                        print(formatted_output)
                
                else:
                    try:
                        parsed_output = json.loads(output)
                        formatted_output = json.dumps(parsed_output, indent=2)
                        formatted_dict = json.loads(formatted_output)
                        
                        if isinstance(formatted_dict, dict):
                            df['publishedDate'] = pd.to_datetime(df['publishedDate'])
                            time = pd.to_datetime(time)
                            
                            for key, value in formatted_dict.items():
                                col_name = f"{agent}_{key}"
                                if col_name not in df.columns:
                                    df[col_name] = ""
                                mask = df['publishedDate'] == time
                                # Process value before assigning to DataFrame
                                processed_value = process_value_for_dataframe(value)
                                df.loc[mask, col_name] = processed_value
                            
                            df['publishedDate'] = df['publishedDate'].dt.strftime('%Y-%m-%d %H:%M')
                            df.to_csv(f"results/{agent_news_file}", index=False)
                        print(f"\n{'=' * 10} {agent} Agent Decision Details {'=' * 10}")     
                        print(formatted_output)
                    except json.JSONDecodeError:
                        print(output)

        except Timeout:
            print("Error: Could not acquire file lock")
            return
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    finally:
        print()