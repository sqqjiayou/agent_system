from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
import json
from datetime import datetime

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
    """格式化数字输出，针对外汇数据进行优化"""
    if abs(value) >= 1:
        return f"{value:,.4f}"
    else:
        return f"{value:.4f}"

def show_agent_reasoning(output, agent_name):
    """显示代理推理过程，增强外汇市场相关信息的展示"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    print(f"Timestamp: {timestamp}")
    
    def convert_to_serializable(obj):
        """转换对象为可序列化格式，增加对外汇特定数据的处理"""
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
                # 更新关键字列表，增加外汇特定字段
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
    
    if isinstance(output, (dict, list)):
        serializable_output = convert_to_serializable(output)
        formatted_output = json.dumps(serializable_output, indent=2)
        
        try:
            # 增强交易信号的展示
            if isinstance(output, dict):
                if 'signal' in output or 'action' in output:
                    action = output.get('action', output.get('signal', '')).upper()
                    confidence = output.get('confidence', '')
                    print(f"Signal: {action} | Confidence: {confidence}")
                    if 'reasoning' in output:
                        print("Reasoning:")
                        print(output['reasoning'])
                    print("\nDetails:")
                print(formatted_output)
            else:
                print(formatted_output)
        except:
            print(formatted_output)
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            print(output)
    
    print("=" * 48)
    print()