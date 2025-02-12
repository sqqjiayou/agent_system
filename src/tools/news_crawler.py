import os
import json
from datetime import datetime, timedelta
import logging
import pandas as pd
from tools.openrouter_config import get_chat_completion

logger = logging.getLogger(__name__)

def get_stock_news(symbol: str, date: str = None, max_news: int = 10) -> list:
    """从本地news_data文件夹获取新闻数据"""
    try:
        # 如果没有提供日期，使用当前日期
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        # 构建新闻文件路径
        date_no_dash = date.replace("-", "")
        news_file = os.path.join("news_data", f"{symbol}_{date_no_dash}.csv")
        
        if not os.path.exists(news_file):
            logger.warning(f"No news file found for {symbol} on {date}")
            return []
            
        # 读取CSV文件
        df = pd.read_csv(news_file)
        
        # 处理新闻数据
        news_list = []
        for _, row in df.iterrows():
            try:
                # 获取内容
                content = str(row['content']).strip()
                if not content:
                    continue
                    
                # 处理标题
                title = str(row['title']).strip()
                if not title:
                    # 如果标题为空，使用内容的第一句话作为标题
                    first_sentence = content.split('。')[0].split('！')[0].split('？')[0]
                    title = first_sentence[:50] + ('...' if len(first_sentence) > 50 else '')
                
                # 处理时间戳
                update_time = row['updateTimestamp'] if isinstance(row['updateTimestamp'], str) else None
                
                news_item = {
                    "title": title,
                    "content": content,
                    "publish_time": update_time or str(row.get('date', date)),
                    "source": "FX News",
                    "relatedSymbols": str(row.get('relatedSymbols', '')),
                    "trade_date": str(row.get('trade_date', date))
                }
                
                news_list.append(news_item)
                logger.info(f"Successfully processed news: {news_item['title']}")
                
            except Exception as e:
                logger.error(f"Failed to process news item: {e}")
                continue
                
        # 按发布时间排序
        try:
            news_list.sort(key=lambda x: x["publish_time"], reverse=True)
        except Exception as e:
            logger.error(f"Failed to sort news list: {e}")
        
        # 限制返回数量
        return news_list[:max_news]
        
    except Exception as e:
        logger.error(f"Failed to get news data: {e}")
        return []

def get_news_sentiment(news_list: list, date: str = None, num_of_news: int = 5) -> float:
    """分析新闻情感倾向，支持中英文"""
    if not news_list:
        return 0.0

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 检查缓存
    cache_file = "src/data/sentiment_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if date in cache:
                    return cache[date]
        except Exception as e:
            logger.error(f"Failed to read sentiment cache: {e}")
            cache = {}
    else:
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": """You are a professional forex market analyst specializing in news sentiment analysis. 
        You can analyze both English and Chinese news articles and provide a sentiment score between -1 and 1:
        
        Sentiment scale (情感量表):
        - 1: Extremely bullish (极度看多) - 重大利好消息，强有力的政策支持
        - 0.5 to 0.9: Bullish (看多) - 积极的经济指标，支持性货币政策
        - 0.1 to 0.4: Slightly bullish (略微看多) - 小幅利好发展
        - 0: Neutral (中性) - 常规更新，信号混合
        - -0.1 to -0.4: Slightly bearish (略微看空) - 小幅利空发展
        - -0.5 to -0.9: Bearish (看空) - 负面经济数据，紧缩政策
        - -1: Extremely bearish (极度看空) - 重大经济危机，严厉政策收紧

        Focus on (关注重点):
        1. Economic indicators (经济指标)
        2. Monetary policy (货币政策)
        3. Political developments (政治发展)
        4. Market sentiment (市场情绪)
        5. Global trade relations (全球贸易关系)
        6. Currency specific factors (货币特定因素)

        Consider (考虑因素):
        1. News reliability (新闻可靠性)
        2. Market impact (市场影响)
        3. Timing relevance (时效性)
        4. Global market context (全球市场背景)"""
    }

    # 准备新闻内容
    news_content = "\n\n".join([
        f"Title (标题): {news['title']}\n"
        f"Time (时间): {news['publish_time']}\n"
        f"Content (内容): {news['content']}"
        for news in news_list[:num_of_news]
    ])

    user_message = {
        "role": "user",
        "content": f"Please analyze the sentiment of the following forex market news (请分析以下外汇市场新闻的情感倾向):\n\n{news_content}\n\nPlease return only a number between -1 and 1, no explanation needed."
    }

    try:
        # 获取LLM分析结果
        result = get_chat_completion([system_message, user_message])
        if result is None:
            return 0.0

        # 解析结果
        sentiment_score = float(result.strip())
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # 缓存结果
        cache[date] = sentiment_score
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

        return sentiment_score

    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {e}")
        return 0.0