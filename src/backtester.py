import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import matplotlib.pyplot as plt
from tools.api import get_forex_news
from tools.api import get_market_data
from tools.api import analyze_news_metrics_with_llm

class BacktestFramework:
    def __init__(self, start_date, end_date, ticker, event_prompt, similarity_threshold=0.5):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.event_prompt = event_prompt
        self.similarity_threshold = similarity_threshold
        self.news_data = None
        self.trade_records = []

    def get_news_data(self):
        """
        Retrieve news data for specified ticker within date range.
        Returns news data in DataFrame format from either API or local files.
        
        Returns:
            pd.DataFrame: News data including publishedDate, title, text, url etc.
                        Returns None if no data found
        """
        news_df = get_forex_news(
            symbol=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if news_df is None or news_df.empty:
            print(f"No news found for {self.ticker} between {self.start_date} and {self.end_date}")
            return None
        else:
            print(f"Found {len(news_df)} news articles for {self.ticker} between {self.start_date} and {self.end_date}")
            return news_df

    def analyze_news_with_llm(self, news_row: pd.Series) -> pd.Series:
        """
        Analyze news using LLM through API service to calculate metrics
        
        Args:
            news_row (pd.Series): Single news item containing title and text
            event_prompt (str): Reference event for comparison
            
        Returns:
            pd.Series: Analysis metrics including:
                - similarity_score
                - sentiment_score 
                - sentiment_class
                - confidence
                - relevance
                - impact_length
                - importance
                Returns None if analysis fails
        """
        try:
            return analyze_news_metrics_with_llm(
                symbol=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                news_row=news_row,
                event_prompt=self.event_prompt
            )
        except Exception as e:
            print(f"Failed to analyze news with LLM: {str(e)}")
            return None

    def execute_trade(self, timestamp, position_size, analysis_result, last_exit_time):
        """
        Execute and record trade details including exit conditions
        
        Args:
            timestamp: Entry time for the trade
            position_size: Size and direction of position (-1 to 1)
            analysis_result: Dictionary containing analysis metrics
            last_exit_time: Time of last trade exit to check for overlapping trades
        """
        timestamp = pd.Timestamp(timestamp)
        # Get entry price and intraday price data
        entry_price_data, intraday_df = self.get_price(timestamp)
        if entry_price_data is None:
            return None
            
        entry_price = entry_price_data['open']
        
        # Determine if trade can actually be executed
        actual_execution = True
        if last_exit_time and timestamp < last_exit_time:
            actual_execution = False
        intraday_df = intraday_df[intraday_df.date > timestamp]

        # Calculate exit conditions using intraday_df
        intraday_df['returns'] = (intraday_df['close'] - entry_price) / entry_price
        if position_size < 0:
            intraday_df['returns'] = -intraday_df['returns']
            
        # Find exit point based on conditions
        exit_mask = (
            (intraday_df['returns'] <= self.stop_loss_limit) |
            (intraday_df['returns'] >= self.take_profit_limit)
        )
        
        time_limit_mask = (
            intraday_df.date >= pd.Timestamp(timestamp) + pd.Timedelta(minutes=self.time_limit)
        )
        
        if exit_mask.any():
            exit_idx = exit_mask.idxmax()
            exit_reason = 'stop_loss' if intraday_df.loc[exit_idx, 'returns'] <= self.stop_loss_limit else 'take_profit'
        elif time_limit_mask.any():
            exit_idx = time_limit_mask.idxmax()
            exit_reason = 'time_limit'
        else:
            exit_idx = intraday_df.index[-1]
            exit_reason = 'day_end'
                
        exit_time = intraday_df.loc[exit_idx, 'date']
        exit_price = intraday_df.loc[exit_idx, 'close']
        trade_return = (exit_price - entry_price) / entry_price * position_size
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'ticker': self.ticker,
            'action': position_size,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'return': trade_return,
            'similarity_score': analysis_result['similarity_score'],
            'sentiment_class': analysis_result['sentiment_class'],
            'actual_execution': actual_execution,
            'confidence': analysis_result['confidence'],
            'importance': analysis_result['importance']
        }
        
        # Print trade details
        print(f"\nTrade Executed at {timestamp}:")
        print(f"{'Actually Executed: ' if actual_execution else 'Simulated Trade: '}")
        print(f"Position Size: {position_size}")
        print(f"Entry Price: {entry_price:.5f}")
        print(f"Exit Time: {exit_time}")
        print(f"Exit Price: {exit_price:.5f}")
        print(f"Exit Reason: {exit_reason}")
        print(f"Return: {trade_return:.2%}")
        print(f"Similarity Score: {analysis_result['similarity_score']:.2f}")
        print(f"Sentiment Class: {analysis_result['sentiment_class']}")
        print("-" * 50)
        
        # Update trade records DataFrame
        self.trade_records = pd.concat([
            self.trade_records,
            pd.DataFrame([trade_record])
        ], ignore_index=True)
        
        # Save updated records to CSV
        os.makedirs('results', exist_ok=True)
        csv_filename = f"results/backtest_records_{self.ticker}_{self.start_date}_{self.end_date}.csv"
        self.trade_records.to_csv(csv_filename, index=False)
        
        if actual_execution:
            return exit_time
        return last_exit_time

    def get_price(self, timestamp):
        """
        Get market data for a specific timestamp using get_market_data API
        
        Args:
            timestamp (str): The timestamp to get market data for
            
        Returns:
            pd.Series: A series containing OHLCV data for the specified timestamp
            dataframe: A dataframe containing OHLCV data for the specified date of timestamp
            
        Usage:
            price_row = get_price(timestamp)
            close_price = price_row['close']  # Get closing price
            open_price = price_row['open']    # Get opening price
            high_price = price_row['high']    # Get high price
            low_price = price_row['low']      # Get low price
            volume = price_row['volume']      # Get trading volume
        """
        price_data, df = get_market_data(
            symbol=self.ticker,
            timestamp=timestamp
        )
        
        if price_data is not None:
            return price_data, df  # Return the entire price row and the whole day (open, high, low, close, volume)
        else:
            print(f"No market data available for {self.ticker} on {timestamp} (non-trading day or missing data)")
            return None, None

    def calculate_performance(self):
        """
        Calculate trading performance metrics including:
        - Cumulative returns
        - Win rate
        - Profit/Loss ratio
        - Sharpe ratio
        - Maximum drawdown
        """
        if self.trade_records.empty:
            print("No trades executed during backtest period")
            return
        # self.trade_records.set_index('timestamp', inplace=True)
        # Filter actual executed trades
        actual_trades = self.trade_records[self.trade_records['actual_execution']]
        
        # Calculate metrics
        total_trades = len(actual_trades)
        winning_trades = len(actual_trades[actual_trades['return'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate P/L ratio
        avg_win = actual_trades[actual_trades['return'] > 0]['return'].mean()
        avg_loss = abs(actual_trades[actual_trades['return'] < 0]['return'].mean())
        pl_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        # Calculate cumulative returns
        actual_trades['cum_returns'] = (1 + actual_trades['return']).cumprod() - 1
        cum_returns = actual_trades['cum_returns']
        
        # Calculate Sharpe ratio (assuming 0 risk-free rate)
        returns_series = actual_trades['return']
        sharpe = np.sqrt(252) * returns_series.mean() / returns_series.std() if len(returns_series) > 1 else 0
        
        # Calculate max drawdown
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns - rolling_max
        max_drawdown = drawdowns.min()
        
        # Print results
        print("\nPerformance Metrics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Profit/Loss Ratio: {pl_ratio:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Final Return: {cum_returns.iloc[-1]:.2%}")
        
        # Store metrics for plotting
        self.performance_metrics = {
            'cum_returns': cum_returns,
            'trades': actual_trades
        }

    def plot_performance(self):
        """
        Plot performance charts including:
        - Strategy cumulative returns starting from 0 with enhanced visualization
        - Trade entry/exit points differentiated by long/short positions with price movement lines
        """
        if not hasattr(self, 'performance_metrics'):
            return
        
        returns = self.performance_metrics['cum_returns']
        trades = self.performance_metrics['trades']
        dates = trades['timestamp']
        
        plt.figure(figsize=(15, 10))
        
        # Plot cumulative returns with enhanced style
        plt.subplot(2, 1, 1)
        returns_with_start = pd.Series([0] + list(returns.values), 
                                    index=[dates[0] - pd.Timedelta(minutes=1)] + list(dates))
        
        plt.plot(returns_with_start.index, returns_with_start.values, 
                color='blue', linewidth=2, linestyle='-', 
                marker='o', markersize=4, label='Strategy Returns')
        
        plt.title(f'{self.ticker} Strategy Returns', fontsize=12, pad=10)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Cumulative Return', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot price chart with entry/exit points and connecting lines
        plt.subplot(2, 1, 2)
        
        # Plot long trades with connecting lines
        long_trades = trades[trades['action'] > 0]
        if not long_trades.empty:
            for _, trade in long_trades.iterrows():
                plt.plot([trade['timestamp'], trade['exit_time']], 
                        [trade['entry_price'], trade['exit_price']], 
                        'k--', alpha=0.3)
            plt.plot(long_trades['timestamp'], long_trades['entry_price'], 
                    '^', color='green', label='Long Entry', markersize=10)
            plt.plot(long_trades['exit_time'], long_trades['exit_price'], 
                    'v', color='lightgreen', label='Long Exit', markersize=10)
        
        # Plot short trades with connecting lines
        short_trades = trades[trades['action'] < 0]
        if not short_trades.empty:
            for _, trade in short_trades.iterrows():
                plt.plot([trade['timestamp'], trade['exit_time']], 
                        [trade['entry_price'], trade['exit_price']], 
                        'k--', alpha=0.3)
            plt.plot(short_trades['timestamp'], short_trades['entry_price'], 
                    '^', color='red', label='Short Entry', markersize=10)
            plt.plot(short_trades['exit_time'], short_trades['exit_price'], 
                    'v', color='pink', label='Short Exit', markersize=10)
        
        plt.title(f'{self.ticker} Trade Entry/Exit Points', fontsize=12, pad=10)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Price', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/performance_{self.ticker}_{self.start_date}_{self.end_date}.png')
        plt.close()

    def run_backtest(self):
        """
        Run backtest simulation based on news analysis and trading rules
        
        Trading Logic:
        1. Analyze each news item for trading signals
        2. Open positions based on similarity and sentiment scores
        3. Track and exit positions based on stop-loss, take-profit, or time limits
        4. Record all trades including simulated ones that weren't executed
        """
        # Initialize trading parameters
        self.stop_loss_limit = -0.003  # -0.3% stop loss
        self.take_profit_limit = 0.005  # 0.5% take profit
        self.time_limit = 30  # 30 minutes max holding time
        
        # Get news data
        self.news_data = self.get_news_data()
        if self.news_data is None:
            return
                
        # Initialize trade records DataFrame
        self.trade_records = pd.DataFrame(columns=[
            'timestamp', 'ticker', 'action', 'entry_price', 'exit_time', 'exit_price',
            'exit_reason', 'return', 'similarity_score', 'sentiment_class',
            'actual_execution', 'confidence', 'importance'
        ])
        
        last_exit_time = None
        
        print(f"\nStarting backtest for {self.ticker} from {self.start_date} to {self.end_date}")
        print("=" * 80)
        
        # Analyze each news item
        for _, news_row in self.news_data.iterrows():
            required_metrics = ['similarity_score', 'sentiment_score', 'sentiment_class',
                            'confidence', 'relevance', 'impact_length', 'importance']
                            
            if all(metric in news_row.index and pd.notna(news_row[metric]) for metric in required_metrics):
                analysis_result = news_row[required_metrics]
            else:
                analysis_result = self.analyze_news_with_llm(news_row=news_row)
                if analysis_result is None:
                    continue
                        
            if abs(analysis_result['similarity_score']) <= self.similarity_threshold:
                continue
                    
            # Determine position size
            position_size = 0
            if analysis_result['similarity_score'] > 0:
                if analysis_result['sentiment_class'] == 1:
                    position_size = 1
                elif analysis_result['sentiment_class'] == 0:
                    position_size = 0.5
            else:
                if analysis_result['sentiment_class'] == -1:
                    position_size = -1
                elif analysis_result['sentiment_class'] == 0:
                    position_size = -0.5
                        
            if position_size == 0:
                continue
            
            # Print triggering news title
            print(f"\nTrading Signal Detected:")
            print(f"News Title: {news_row['title']}")
            print(f"Time: {news_row['publishedDate']}")
            print("-" * 50)
                    
            # Execute trade
            result = self.execute_trade(
                news_row['publishedDate'],
                position_size,
                analysis_result,
                last_exit_time
            )

            if result is None:
                print(f"No market data available for trade at {news_row['publishedDate']}, skipping this signal")
                print("-" * 50)
                continue
            else:
                last_exit_time = result
        
        # Calculate and display performance metrics
        self.calculate_performance()
        self.plot_performance()




if __name__ == "__main__":
    backtest = BacktestFramework(
        start_date='2025-01-02',
        end_date='2025-02-13',
        ticker='EURUSD',
        event_prompt = "US Dollar weakens as Fed rate cut expectations increase"
    )
    backtest.run_backtest()

    """
    event_prompt examples for EURUSD:
    Bullish EUR:
    "US Dollar weakens as Fed rate cut expectations increase"
    "US economic data disappoints expectations"
    "ECB maintains hawkish stance on rates"
    "German/Eurozone economic data beats expectations"

    Bearish EUR:
    "US PMI/ISM data beats expectations while Eurozone PMIs remain weak"
    "US Core PCE/CPI data comes in hotter than expected"
    "Strong US jobs data pressures EUR"
    "ECB rate cut expectations increase"
    """
    