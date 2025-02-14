from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
from tools.news_crawler import get_stock_news, get_news_sentiment
from tools.openrouter_config import get_chat_completion
import json
from datetime import datetime, timedelta

def sentiment_agent(state: AgentState):
    """Analyzes forex market sentiment and generates trading signals"""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    direction = data["direction"]
    event_prompt = data["event_prompt"]
    current_date = data["end_date"]
    # 获取新闻数量，默认为5
    num_of_news = data.get("num_of_news", 5)

    # 获取新闻数据
    news_list = get_stock_news(symbol, date=current_date, max_news=num_of_news)

    # 过滤最近48小时的新闻（外汇市场对新闻反应更快）
    cutoff_date = datetime.strptime(current_date, "%Y-%m-%d") - timedelta(hours=48)
    recent_news = []
    
    for news in news_list:
        try:
            news_date = datetime.strptime(news['publish_time'], '%Y-%m-%d %H:%M:%S')
            if news_date > cutoff_date:
                recent_news.append(news)
        except Exception as e:
            continue

    # 获取情感分数
    sentiment_score = get_news_sentiment(recent_news, date=current_date, num_of_news=num_of_news)

    # 基于情感分数生成交易信号（降低阈值以适应外汇市场）
    if sentiment_score >= 0.2:  # 从0.5改为0.2
        signal = "bullish"
        confidence = str(round(min(abs(sentiment_score) * 200, 100))) + "%"  # 提高置信度
    elif sentiment_score <= -0.2:  # 从-0.5改为-0.2
        signal = "bearish"
        confidence = str(round(min(abs(sentiment_score) * 200, 100))) + "%"  # 提高置信度
    else:
        signal = "neutral"
        confidence = str(round((1 - abs(sentiment_score * 2)) * 100)) + "%"

    # 计算新闻权重
    news_weight = calculate_news_weight(recent_news)
    
    # 生成分析结果
    reasoning = {
        "sentiment_score": f"{sentiment_score:.2f}",
        "news_count": len(recent_news),
        "news_weight": f"{news_weight:.2f}",
        "time_period": "last 48 hours",
        "news_summary": summarize_news(recent_news[:3])  # 总结最近3条新闻
    }

    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning
    }

    # 显示推理过程
    if show_reasoning:
        show_agent_reasoning(message_content, "FX Sentiment Analysis Agent")

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    return {
        "messages": [message],
        "data": data,
    }

def calculate_news_weight(news_list: list) -> float:
    """
    计算新闻的权重分数
    基于新闻的时效性和重要性
    """
    if not news_list:
        return 0.0
        
    total_weight = 0
    for news in news_list:
        try:
            # 时效性权重（越新越重要）
            news_time = datetime.strptime(news['publish_time'], '%Y-%m-%d %H:%M:%S')
            time_diff = datetime.now() - news_time
            time_weight = max(0, 1 - (time_diff.total_seconds() / (48 * 3600)))  # 48小时内线性衰减
            
            # 重要性权重（基于内容长度和来源）
            content_weight = min(len(news['content']) / 1000, 1)  # 内容长度权重
            source_weight = 1.0  # 可以根据新闻来源调整
            
            # 综合权重
            news_weight = (time_weight * 0.4 + content_weight * 0.3 + source_weight * 0.3)
            total_weight += news_weight
            
        except Exception as e:
            continue
            
    return total_weight / len(news_list) if news_list else 0.0

def summarize_news(news_list: list) -> list:
    """
    总结最近的新闻
    返回关键信息列表
    """
    summaries = []
    for news in news_list:
        try:
            summary = {
                "time": news['publish_time'],
                "title": news['title'],
                "key_points": news['content'][:200] + "..." if len(news['content']) > 200 else news['content']
            }
            summaries.append(summary)
        except Exception as e:
            continue
            
    return summaries