import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import time
from tools.api import get_forex_news, get_market_data, analyze_news_metrics_with_llm
import logging
from main import run_hedge_fund
os.environ['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '10'


# 移除根 logger 所有 StreamHandler（禁止根 logger 对控制台输出）
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)

# 禁用 api_calls 模块的 logger
api_logger = logging.getLogger("api_calls")
api_logger.setLevel(logging.ERROR)
api_logger.propagate = False
api_logger.disabled = True

import sys
import pandas_market_calendars as mcal
import warnings

from main import run_hedge_fund

# 根据操作系统配置中文字体
if sys.platform.startswith('win'):
    matplotlib.rc('font', family='Microsoft YaHei')
elif sys.platform.startswith('linux'):
    matplotlib.rc('font', family='WenQuanYi Micro Hei')
else:
    matplotlib.rc('font', family='PingFang SC')

# 允许负号显示
matplotlib.rcParams['axes.unicode_minus'] = False

# 禁用 matplotlib 与 pandas.plotting 警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas.plotting')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

class BacktestFramework:
    def __init__(self, agent, start_date, end_date, ticker, event_prompt, similarity_threshold=0.4):
        self.agent = agent
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.event_prompt = event_prompt
        self.similarity_threshold = similarity_threshold
        self.news_data = None
        self.trade_records = []

        # 设置日志
        self.setup_backtest_logging()
        self.logger = self.setup_logging()

        # 初始化 API 调用计数
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0

        # 验证输入有效性
        self.validate_inputs()

    def setup_logging(self):
        logger = logging.getLogger('backtester')
        logger.setLevel(logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def validate_inputs(self):
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("Start date must be earlier than end date")
            if not isinstance(self.ticker, str) or len(self.ticker) == 0:
                raise ValueError("Invalid ticker format")
            if not (self.ticker.isalpha() or (len(self.ticker) == 6 and self.ticker.isdigit())):
                self.backtest_logger.warning(f"Ticker {self.ticker} may be in unusual format")
            self.backtest_logger.info("Input parameters validated")
        except Exception as e:
            self.backtest_logger.error(f"Input parameter validation failed: {str(e)}")
            raise

    def setup_backtest_logging(self):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.backtest_logger = logging.getLogger('backtest')
        self.backtest_logger.setLevel(logging.WARNING)
        if self.backtest_logger.handlers:
            self.backtest_logger.handlers.clear()
        current_date = datetime.now().strftime('%Y%m%d')
        backtest_period = f"{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}"
        log_file = os.path.join(log_dir, f"backtest_{self.ticker}_{current_date}_{backtest_period}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.backtest_logger.addHandler(file_handler)
        self.backtest_logger.info(f"Backtest Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.backtest_logger.info(f"Ticker: {self.ticker}")
        self.backtest_logger.info(f"Backtest Period: {self.start_date} to {self.end_date}")
        self.backtest_logger.info("-" * 100)

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

    def execute_trade(self, timestamp, entry_price_data, intraday_df, position_size, analysis_result, last_exit_time, news_title):
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
        # entry_price_data, intraday_df = self.get_price(timestamp)
        # if entry_price_data is None:
        #     return None
            
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
        
        # Modify trade_record dictionary:
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
            'importance': analysis_result['importance'],
            'news_title': news_title  # Add news title to record
        }
        
        # Print trade details
        print(f"\nTrade Executed at {timestamp}:")
        print(f"News: {news_title}")  # Add news title to print output
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
        
        # Format metrics for saving
        metrics_summary = pd.DataFrame({
            'Metric': ['Total Trades', 'Win Rate', 'Profit/Loss Ratio', 'Sharpe Ratio', 'Maximum Drawdown', 'Final Return'],
            'Value': [
                f"{total_trades}",
                f"{win_rate:.2%}",
                f"{pl_ratio:.2f}",
                f"{sharpe:.2f}",
                f"{max_drawdown:.2%}",
                f"{cum_returns.iloc[-1]:.2%}"
            ]
        })

        # Print metrics
        print("\nPerformance Metrics:")
        for _, row in metrics_summary.iterrows():
            print(f"{row['Metric']}: {row['Value']}")

        # Append metrics to the existing CSV with blank row separation
        csv_filename = f"results/backtest_records_{self.ticker}_{self.start_date}_{self.end_date}.csv"
        with open(csv_filename, 'a', newline='') as f:
            f.write('\n')  # Add blank row
            metrics_summary.to_csv(f, index=False)

        # Store metrics for plotting
        self.performance_metrics = {
            'cum_returns': cum_returns,
            'trades': actual_trades
        }

    def plot_performance(self):
        """
        Plot performance charts including:
        - Strategy cumulative returns vs Buy & Hold returns starting from 0 with enhanced visualization
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
        
        # Strategy returns
        returns_with_start = pd.Series([0] + list(returns.values), 
                                    index=[dates[0] - pd.Timedelta(minutes=1)] + list(dates))
        
        plt.plot(returns_with_start.index, returns_with_start.values * 100, 
                color='blue', linewidth=2, linestyle='-', 
                marker='o', markersize=4, label='Strategy Returns')
        
        # Calculate and plot Buy & Hold returns
        trade_points = pd.concat([
            trades[['timestamp', 'entry_price']].rename(columns={'timestamp': 'time', 'entry_price': 'price'}),
            trades[['exit_time', 'exit_price']].rename(columns={'exit_time': 'time', 'exit_price': 'price'})
        ]).sort_values('time')
        
        initial_price = trade_points.iloc[0]['price']
        bnh_returns = (trade_points['price'] - initial_price) / initial_price * 100
        
        plt.plot(trade_points['time'], bnh_returns.values,
                color='gray', linewidth=2, linestyle='--',
                marker='o', markersize=4, label='Buy & Hold Returns')
        
        plt.title(f'{self.ticker} Strategy Returns vs Buy & Hold', fontsize=12, pad=10)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Cumulative Return (%)', fontsize=10)
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

    def get_agent_decision(self, timestamp):
        max_retries = 5
        current_time = time.time()
        if current_time - self._api_window_start >= 60:
            self._api_call_count = 0
            self._api_window_start = current_time
        if self._api_call_count >= 8:
            wait_time = 60 - (current_time - self._api_window_start)
            if wait_time > 0:
                self.backtest_logger.info(f"API limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self._api_call_count = 0
                self._api_window_start = time.time()
        for attempt in range(max_retries):
            try:
                if self._last_api_call:
                    time_since_last_call = time.time() - self._last_api_call
                    if time_since_last_call < 6:
                        time.sleep(6 - time_since_last_call)
                self._last_api_call = time.time()
                self._api_call_count += 1
                result = self.agent(
                    ticker=self.ticker,
                    time=str(timestamp),
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                try:
                    if isinstance(result, str):
                        result = result.replace('```json\n', '').replace('\n```', '').strip()
                        parsed_result = json.loads(result)
                        formatted_result = {"decision": parsed_result, "analyst_signals": {}}
                        if "agent_signals" in parsed_result:
                            parsed_result["agent_signals"] = self.parse_signals(parsed_result["agent_signals"])
                            formatted_result["analyst_signals"] = {
                                agent_name: {
                                    "signal": signal
                                }
                                for agent_name, signal in parsed_result["agent_signals"].items()
                            }
                        return formatted_result
                    return result
                except json.JSONDecodeError as e:
                    self.backtest_logger.warning(f"JSON parsing error: {str(e)}")
                    self.backtest_logger.warning(f"Raw result: {result}")
                    return {"decision": {"action": "neutral", "quantity": 0}, "analyst_signals": {}}
            except Exception as e:
                if "AFC is enabled" in str(e):
                    self.backtest_logger.warning("AFC limit triggered, waiting 60 seconds...")
                    time.sleep(60)
                    self._api_call_count = 0
                    self._api_window_start = time.time()
                    continue
                self.backtest_logger.warning(f"Failed to get agent decision (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {"decision": {"action": "neutral", "quantity": 0}, "analyst_signals": {}}
                time.sleep(2 ** attempt)

    def parse_signals(self, signal_str):
        # 将字符串按逗号分割
        signals = signal_str.split(',')
        
        # 创建结果字典
        result = {}
        
        # 处理每个信号
        for signal in signals:
            # 去除首尾空格
            signal = signal.strip()
            # 分离 agent 和信号值
            agent, value = signal.split(':')
            # 提取 signal (bearish/bullish)
            signal_type = value.split('(')[0].strip()
            
            # 添加到结果字典
            result[agent] = signal_type

        return result

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
        self.stop_loss_limit = -0.005  # -0.3% stop loss
        self.take_profit_limit = 0.01  # 0.5% take profit
        self.time_limit = 1440  # 30 minutes max holding time
        
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
                        
            if abs(float(analysis_result['similarity_score'])) <= self.similarity_threshold:
                continue
            
            # Get entry price and intraday price data, if not a trading day/time, skip this signal
            entry_price_data, intraday_df = self.get_price(news_row['publishedDate'])
            if entry_price_data is None or news_row['publishedDate'].ceil('min') != entry_price_data['date'].ceil('min'):
                continue

            output = self.get_agent_decision(news_row['publishedDate'])
            if output.get("analyst_signals"):
                print("\nAgent Analysis Results:")
                for agent_name, signal in output["analyst_signals"].items():
                    print(f"\n{agent_name}: - Signal: {signal.get('signal', 'unknown')}, Confidence: {signal.get('confidence', 0)*100:.2f}%")
                    if "analysis" in signal:
                        print("  Analysis: " + str(signal["analysis"]))
                    if "reason" in signal:
                        print("  Reasoning: " + str(signal["reason"]))
            agent_decision = output.get("decision", {"action": "neutral", "quantity": 0})
            action, position_size = agent_decision.get("action", "neutral"), agent_decision.get("quantity", 0)
            print(f"\nFinal Decision: {action.upper()}, Target Quantity: {position_size}")
            if "reasoning" in agent_decision:
                print("Decision Reasoning: " + str(agent_decision["reasoning"]))
                        
            if position_size == 0:
                continue
            
            # Print triggering news title
            print(f"\nTrading Signal Detected:")
            print(f"News Title: {news_row['title']}")
            print(f"Time: {news_row['publishedDate']}")
            print("-" * 50)
            if action == "short":
                position_size = -position_size

            # Execute trade
            result = self.execute_trade(
                news_row['publishedDate'],
                entry_price_data, 
                intraday_df,
                position_size,
                analysis_result,
                last_exit_time,
                news_row['title']  # Add news title as new parameter

            )

            if result is None:
                print(f"No market data available for trade at {news_row['publishedDate']}, skipping this signal")
                print("-" * 50)
                continue
            else:
                last_exit_time = result
        # Read both CSVs
        df1 = pd.read_csv(f"results/Agent_{self.ticker}_{self.start_date}_{self.end_date}.csv")
        df2 = pd.read_csv(f"results/backtest_records_{self.ticker}_{self.start_date}_{self.end_date}.csv")
        # Convert both columns to datetime
        df1['publishedDate'] = pd.to_datetime(df1['publishedDate'])
        df2['timestamp'] = pd.to_datetime(df2['timestamp'])
        # Get columns after 'uniqueness' from df1
        agent_cols = df1.columns[df1.columns.get_loc('uniqueness')+1:]

        # Add these columns to df2 
        for col in agent_cols:
            df2[col] = ""

        # Create a mapping dictionary for each column
        for col in agent_cols:
            value_map = dict(zip(df1['publishedDate'], df1[col]))
            df2[col] = df2['timestamp'].map(value_map).fillna('')

        # Save updated df2
        df2.to_csv(f"results/backtest_records_{self.ticker}_{self.start_date}_{self.end_date}_agent_reasons.csv", index=False)
        # Calculate and display performance metrics
        self.calculate_performance()
        self.plot_performance()




if __name__ == "__main__":
    backtest = BacktestFramework(
        agent=run_hedge_fund,
        start_date='2025-02-01',
        end_date='2025-02-16',
        ticker='EURUSD',
        event_prompt = "Trump announces new tariffs on European goods"
    )
    backtest.run_backtest()

    """
    event_prompt examples for EURUSD:
    Bullish EUR:
    "Trump delays implementation of EU tariffs"


    Bearish EUR:
    "Trump threatens reciprocal tariffs against EU"

    """
    