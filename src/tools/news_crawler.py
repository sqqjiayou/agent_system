import os
import logging
import pandas as pd
from datetime import datetime
from openrouter_config import get_chat_completion

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

import requests
import pandas as pd
from datetime import datetime

# 设定日期范围
start_date = "2025-02-01"
end_date = "2025-02-03"

# 构建 API URL
base_url = "https://financialmodelingprep.com/api/v4/forex_news"
params = {
    "symbol": "EURUSD",
    "from": start_date,
    "to": end_date,
    "apikey": os.getenv("FMP_API_KEY")
}

# 发送请求
response = requests.get(url=base_url, params=params)

# 转换为 DataFrame
df = pd.DataFrame(response.json())

# 显示结果
df

def get_headlines_excel(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Reads the Excel file 'news.xlsx' from the 'news_data' folder and filters
    the news headlines that fall between the specified start and end dates.

    Parameters:
    - start_date: The start date as a string in 'YYYY-MM-DD' format.
    - end_date: The end date as a string in 'YYYY-MM-DD' format.

    Returns:
    - A pandas DataFrame with the filtered news headlines.
    """
    try:
        excel_path = os.path.join("news_data", "news.xlsx")
        if not os.path.exists(excel_path):
            logger.error(f"Excel file {excel_path} does not exist.")
            return pd.DataFrame()
        
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Check for necessary column 'Date'
        if 'Date' not in df.columns:
            logger.error("Column 'Date' not found in the Excel file.")
            return pd.DataFrame()
        
        # Convert the 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert the input dates into datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter rows that fall between start_dt and end_dt (inclusive)
        filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
        logger.info(f"Found {len(filtered_df)} headlines between {start_date} and {end_date}.")
        return filtered_df
    except Exception as e:
        logger.error(f"Error reading or filtering Excel file: {e}")
        return pd.DataFrame()

def update_news_similarity(event_prompt: str, start_date: str, end_date: str, threshold: float = 0.5) -> None:
    """
    Updates the Excel file by performing a similarity analysis for each news headline 
    within the specified date range. Each headline is compared to the provided event_prompt 
    using a LLM, and the similarity score is inserted into a new (or existing) column 
    'Similarity_score'. Headlines that already have a similarity score are skipped.

    For each headline, if the absolute similarity score is greater than the threshold, 
    the Date, Headline, and the resulting similarity score are printed.

    Parameters:
    - event_prompt: A string that describes the event prompt to which the headlines are compared.
    - start_date: The starting date as a string in 'YYYY-MM-DD' format.
    - end_date: The ending date as a string in 'YYYY-MM-DD' format.
    - threshold: A float threshold for printing headlines with significant similarity (default is 0.5).
    """
    try:
        excel_path = os.path.join("news_data", "news.xlsx")
        if not os.path.exists(excel_path):
            logger.error(f"Excel file {excel_path} does not exist.")
            return

        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_path)
        
        # Check for the required columns 'Date' and 'Headline'
        if 'Date' not in df.columns or 'Headline' not in df.columns:
            logger.error("Required columns 'Date' and/or 'Headline' not found in the Excel file.")
            return
        
        # Convert the 'Date' column to datetime type
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create a mask for news within the date range
        date_mask = (df['Date'] >= start_dt) & (df['Date'] <= end_dt)
        df_filtered = df.loc[date_mask].copy()
        
        logger.info(f"Processing {len(df_filtered)} headlines within the date range.")
        
        # If 'Similarity_score' column does not exist, create it with empty values
        if 'Similarity_score' not in df.columns:
            df['Similarity_score'] = pd.NA
        
        # Prepare the system message for similarity analysis with detailed scoring guidelines
        system_message = {
            "role": "system",
            "content": (
                "You are a professional forex market analyst with expertise in data analysis. "
                "You will be provided with a news headline and an event prompt. Analyze the similarity "
                "between the news headline and event prompt based on the following scoring guidelines:\n\n"
                "Scoring Guidelines:\n"
                "1. Score should be between -1 and 1.\n"
                "2. Score 1: Headline is highly similar and has the same directional impact.\n"
                "3. Score -1: Headline is highly similar but has the opposite directional impact.\n"
                "4. Score 0: Headline is unrelated to the event.\n"
                "5. Most headlines should score close to 0 unless clearly related.\n"
                "6. Use intermediate values only when partial similarity exists.\n\n"
                "Return only a number between -1 and 1 with no explanation."
            )
        }
        
        # Process each row in the filtered DataFrame
        for idx in df_filtered.index:
            # Skip headlines that already have a similarity score
            if pd.notna(df.loc[idx, 'Similarity_score']):
                continue

            headline = str(df.loc[idx, 'Headline']).strip()
            if not headline:
                logger.warning(f"Empty headline at index {idx}; skipping similarity analysis.")
                continue

            # Construct the user message for LLM with the headline and event_prompt
            user_message = {
                "role": "user",
                "content": (
                    f"Please analyze the similarity between the following news headline and event prompt.\n\n"
                    f"Headline: {headline}\n\n"
                    f"Event Prompt: {event_prompt}\n\n"
                    "Return only a number between -1 and 1 with no explanation."
                )
            }
            
            try:
                # Get similarity score from the LLM
                result = get_chat_completion([system_message, user_message])
                if result is None:
                    similarity_score = 0.0
                else:
                    similarity_score = float(result.strip())
                    # Ensure score is within the range -1 to 1
                    similarity_score = max(-1.0, min(1.0, similarity_score))
            except Exception as e:
                logger.error(f"Error processing similarity for headline at index {idx}: {e}")
                similarity_score = 0.0
            
            # Update the DataFrame with the similarity score
            df.loc[idx, 'Similarity_score'] = similarity_score
            
            # If the absolute similarity score exceeds the threshold, print its details
            if abs(similarity_score) > threshold:
                news_date = df.loc[idx, 'Date']
                formatted_date = news_date.strftime("%Y-%m-%d") if pd.notna(news_date) else "Unknown Date"
                print(f"Date: {formatted_date} | Headline: {headline} | Similarity Score: {similarity_score}")
                logger.info(f"Significant similarity for row index {idx}: Score {similarity_score}")
                
        # Write the updated DataFrame (including the new similarity scores) back to the Excel file
        df.to_excel(excel_path, index=False)
        logger.info("News Excel file updated with similarity scores.")
        
    except Exception as e:
        logger.error(f"Error processing news similarity: {e}")
        return

if __name__ == "__main__":
    # # Example usage:
    # EVENT_PROMPT = "PBOC sets stronger yuan fixing than expected."
    # START_DATE = "2018-01-01"
    # END_DATE = "2018-02-07"
    # THRESHOLD = 0.5

    # # First, you could retrieve headlines (if needed) using get_headlines_excel()
    # headlines_df = get_headlines_excel(START_DATE, END_DATE)
    # logger.info(f"Retrieved {len(headlines_df)} headlines for the given date range.")
    
    # # Update the Excel file by analyzing similarity between each headline and the event prompt
    # update_news_similarity(EVENT_PROMPT, START_DATE, END_DATE, THRESHOLD)
    # 发送请求
    response = requests.get(url=base_url, params=params)

    # 转换为 DataFrame
    df = pd.DataFrame(response.json())

    # 显示结果
    print(df)