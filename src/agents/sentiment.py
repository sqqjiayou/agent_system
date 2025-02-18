from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
from tools.api import get_forex_news_history, DeepSeek2
from tools.openrouter_config import get_chat_completion
import json
from datetime import datetime, timedelta
import pandas as pd

def sentiment_agent(state: AgentState):
    """Analyzes forex market sentiment and generates trading signals"""
    # print('sentiment_agent')
    # print('#'*50)
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    time = data["time"]
    start_date = data["start_date"]
    end_date = data["end_date"]

    # get news history data
    file_path, news_df = get_forex_news_history(symbol, start_date, end_date, time, num_hour=48)

    """
    Analyze uniqueness of latest news by comparing with previous titles
    and update signal analysis based on metrics
    """
    # Get latest news row and previous titles
    latest_row = news_df.iloc[-1]
    previous_titles = news_df.iloc[:-1]['title'].tolist()
    
    # Read existing CSV and check for uniqueness value
    try:
        df = pd.read_csv(file_path)
        if 'uniqueness' not in df.columns:
            df['uniqueness'] = None
        
        # Check if uniqueness already exists for latest row
        df['publishedDate'] = pd.to_datetime(df['publishedDate'])
        matched_row = df[df['publishedDate'] == latest_row['publishedDate']]       
        if not matched_row.empty and pd.notna(matched_row['uniqueness'].iloc[0]):
            uniqueness_score = int(matched_row['uniqueness'].iloc[0])
        else:
            # Call LLM to analyze uniqueness if value doesn't exist
            llm = DeepSeek2()
            prompt = f"""Compare this news title with previous titles and score its uniqueness:

            Latest title: {latest_row['title']}

            Previous titles from last 48 hours:
            {' | '.join(previous_titles)}

            Score guidelines:
            1 = New story (information not seen before)
            2 or higher = Update to previous story (integer indicating number of similar previous stories)

            Return only a single integer number.
            """
            uniqueness_score = int(llm.invoke(prompt).strip())
            
            # Update uniqueness in CSV
            df.loc[df['publishedDate'] == latest_row['publishedDate'], 'uniqueness'] = uniqueness_score
            df.to_csv(file_path, index=False)

        # Calculate trading signal
        sentiment_impact = latest_row['sentiment_score'] * latest_row['confidence'] * latest_row['importance']
        
        # Adjust impact based on uniqueness
        adjusted_impact = sentiment_impact / uniqueness_score  # Lower impact for repeated news
        
        # Generate signal
        if adjusted_impact >= 0.2:
            signal = "bullish"
            confidence = min(abs(adjusted_impact) * 100, 100)
        elif adjusted_impact <= -0.2:
            signal = "bearish" 
            confidence = min(abs(adjusted_impact) * 100, 100)
        else:
            signal = "neutral"
            confidence = (1 - abs(adjusted_impact)) * 100

        reasoning = {
            "event_similarity": latest_row['similarity_score'],
            "base_sentiment": latest_row['sentiment_score'],
            "uniqueness": uniqueness_score,
            "confidence": latest_row['confidence'],
            "importance": latest_row['importance'],
            "impact_length": latest_row['impact_length'],
            "adjusted_impact": f"{adjusted_impact:.2f}"
        }

        message_content = {
            "signal": signal,
            "confidence": f"{confidence:.0f}%",
            "reasoning": reasoning
        }

       # show_reasoning
        if show_reasoning:
            show_agent_reasoning(message_content, "FX Sentiment Analysis Agent")

        # create the sentiment analysis message
        message = HumanMessage(
            content=json.dumps(message_content),
            name="sentiment_agent",
        )

        return {
            "messages": [message],
            "data": data,
        }
    except Exception as e:
        print(f"Error analyzing news uniqueness: {e}")
        return None
