#!/usr/bin/env python3
# -- coding: utf-8 --
"""
本代码实现以下功能：
1. 根据回测起始日期（例如 "2024-09-25"）与结束日期，生成所有业务日（周一~周五）。
2. 第一个指定的交易日使用前一个（业务日）作为决策依据进行判断和操作，但所有记录都从指定的交易日开始保存。
3. 每个交易日以当天开盘价进行交易，决策依据为前一交易日的数据。实际执行交易调整仓位，并计算每日收益。
4. Buy & Hold 策略假设在回测开始日以当天的开盘价买入，并持有到最后一天，其每日表现以该基准计算。
5. 各项收益指标（每日收益、累计收益、总收益、夏普比率、最大回撤等）的计算均基于组合市值。
6. 图中所有数字以四位小数显示，保证显示精度。

总体说明：
- Daily Return（每日收益） = [(当天组合价值 / 前一交易日组合价值) - 1] × 100%
- Cumulative Return（累计收益） = [(当天组合价值 / 初始资本) - 1] × 100%
- Total Return（总收益）同累计收益，采用最后一天数据计算
- Buy & Hold 策略：假设在回测开始日以当天开盘价买入，持仓数量 = 初始资本 / 当天开盘价；之后每天以当日开盘价计算持仓价值，
  Buy & Hold Value = 持仓数量 × 当日开盘价，Buy & Hold Return = [(Buy & Hold Value / 初始资本) - 1] × 100%

请根据实际情况检查 get_price_data 返回的“open”价格是否准确。
"""

from datetime import datetime, timedelta
import json
import time
import logging
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

import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import matplotlib
import pandas_market_calendars as mcal
import warnings

from main import run_hedge_fund
from tools.api import get_price_data

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


class Backtester:
    def __init__(self, agent, ticker, start_date, end_date, initial_capital, direction, event_prompt):
        self.agent = agent
        self.ticker = ticker
        self.start_date = start_date     # 格式 "YYYY-MM-DD", 如 "2024-09-25"
        self.end_date = end_date         # 回测结束日期
        self.initial_capital = initial_capital
        self.portfolio = {"cash": initial_capital, "stock": 0}  # 当前持仓和现金
        self.portfolio_values = []       # 记录每天 (日期、组合价值、每日收益、当天开盘价)
        self.event_prompt = event_prompt
        self.direction = direction
        self.trade_records = []          # 记录每笔交易收益（百分比）

        # 设置日志
        self.setup_backtest_logging()
        self.logger = self.setup_logging()

        # 初始化 API 调用计数
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0

        # 初始化日历（备用，仅用于生成交易日）
        self.nyse = mcal.get_calendar('NYSE')

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
            if self.initial_capital <= 0:
                raise ValueError("Initial capital must be greater than 0")
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
        self.backtest_logger.info(f"Initial Capital: {self.initial_capital:,.2f}\n")
        self.backtest_logger.info("-" * 100)

    def get_previous_trading_day(self, date_str):
        dt = pd.to_datetime(date_str)
        prev = dt - pd.tseries.offsets.BDay(1)
        return prev.strftime('%Y-%m-%d')

    def get_agent_decision(self, current_date, lookback_start, portfolio, direction, event_prompt):
        max_retries = 3
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
                    start_date=lookback_start,
                    end_date=current_date,
                    portfolio=portfolio,
                    direction=direction,
                    event_prompt=event_prompt
                )
                try:
                    if isinstance(result, str):
                        result = result.replace('```json\n', '').replace('\n```', '').strip()
                        parsed_result = json.loads(result)
                        formatted_result = {"decision": parsed_result, "analyst_signals": {}}
                        if "agent_signals" in parsed_result:
                            formatted_result["analyst_signals"] = {
                                agent_name: {
                                    "signal": signal_data.get("signal", "unknown"),
                                    "confidence": signal_data.get("confidence", 0)
                                }
                                for agent_name, signal_data in parsed_result["agent_signals"].items()
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

    def execute_trade(self, action, target_quantity, current_price):
        """
        根据策略执行交易：
         - "long": 目标仓位为正，差额部分以当天开盘价买入；
         - "short": 目标仓位为负，差额部分以当天开盘价卖出；
         - "neutral": 平仓。
        返回实际成交数量。
        """
        trade = 0.0
        current_pos = self.portfolio["stock"]
        if action == "long":
            target = target_quantity  # 正仓
            delta = target - current_pos
            if delta > 0:
                possible = int(self.portfolio["cash"] // current_price)
                trade = min(delta, possible)
                self.portfolio["cash"] -= trade * current_price
                self.portfolio["stock"] += trade
            elif delta < 0:
                trade = abs(delta)
                self.portfolio["cash"] += trade * current_price
                self.portfolio["stock"] -= trade
            return trade
        elif action == "short":
            target = -target_quantity  # 负仓
            delta = target - current_pos
            if delta < 0:
                trade = abs(delta)
                self.portfolio["cash"] += trade * current_price
                self.portfolio["stock"] -= trade
            elif delta > 0:
                possible = int(self.portfolio["cash"] // current_price)
                trade = min(delta, possible)
                self.portfolio["cash"] -= trade * current_price
                self.portfolio["stock"] += trade
            return trade
        else:  # neutral
            target = 0
            delta = target - current_pos
            if delta > 0:
                possible = int(self.portfolio["cash"] // current_price)
                trade = min(delta, possible)
                self.portfolio["cash"] -= trade * current_price
                self.portfolio["stock"] += trade
            elif delta < 0:
                trade = abs(delta)
                self.portfolio["cash"] += trade * current_price
                self.portfolio["stock"] -= trade
            return trade

    def run_backtest(self):
        # 生成交易日期：起点为 (start_date - 1个业务日)；
        # 这样第一个【实际交易日】即为 self.start_date，其决策依据为前一交易日的数据。
        start_dt = pd.to_datetime(self.start_date) - pd.tseries.offsets.BDay(1)
        end_dt = pd.to_datetime(self.end_date)
        dates = pd.date_range(start=start_dt, end=end_dt, freq='B')

        self.trade_records = []
        last_trade_value = self.initial_capital
        self.backtest_logger.info("\nStarting backtest...")
        print(f"{'Date':<12} {'Ticker':<6} {'Action':<6} {'Qty':>8} {'Price':>12} {'Cash':>15} {'Stock':>8} {'Total':>15} {'Bull':>8} {'Bear':>8} {'Neutral':>8}")
        print("-" * 110)
        # 注意：循环从 i=1 开始，i=0 仅作为决策参考日期，不记录实际交易数据
        for i in range(1, len(dates)):
            current_date_str = dates[i].strftime("%Y-%m-%d")
            decision_date_str = dates[i-1].strftime("%Y-%m-%d")
            try:
                df = get_price_data(self.ticker, current_date_str, current_date_str)
                if df is None or df.empty:
                    self.backtest_logger.warning(f"No price data for {current_date_str}, skipping...")
                    continue
                current_price = df.iloc[0]['open']
            except Exception as e:
                self.backtest_logger.error(f"Error getting price data for {current_date_str}: {str(e)}")
                continue

            output = self.get_agent_decision(decision_date_str, (dates[i-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), self.portfolio, self.direction, self.event_prompt)
            if output.get("analyst_signals"):
                self.backtest_logger.info("\nAgent Analysis Results:")
                for agent_name, signal in output["analyst_signals"].items():
                    self.backtest_logger.info(f"\n{agent_name}: - Signal: {signal.get('signal', 'unknown')}, Confidence: {signal.get('confidence', 0)*100:.2f}%")
                    if "analysis" in signal:
                        self.backtest_logger.info("  Analysis: " + str(signal["analysis"]))
                    if "reason" in signal:
                        self.backtest_logger.info("  Reasoning: " + str(signal["reason"]))
            agent_decision = output.get("decision", {"action": "neutral", "quantity": 0})
            action, target_qty = agent_decision.get("action", "neutral"), agent_decision.get("quantity", 0)
            self.backtest_logger.info(f"\nFinal Decision: {action.upper()}, Target Quantity: {target_qty}")
            if "reasoning" in agent_decision:
                self.backtest_logger.info("Decision Reasoning: " + str(agent_decision["reasoning"]))
            executed_quantity = self.execute_trade(action, target_qty, current_price)
            total_value = self.portfolio["cash"] + self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value
            previous_value = self.portfolio_values[-1]["Portfolio Value"] if self.portfolio_values else self.initial_capital
            daily_return = ((total_value / previous_value) - 1) * 100
            self.portfolio_values.append({
                "Date": current_date_str,
                "Portfolio Value": total_value,
                "Daily Return": daily_return,
                "Price": current_price
            })
            if executed_quantity > 0:
                trade_return = ((total_value / last_trade_value) - 1) * 100
                self.trade_records.append({"Date": current_date_str, "Return": trade_return})
                last_trade_value = total_value
            bull_count = sum(1 for signal in output.get("analyst_signals", {}).values() if signal.get("signal") in ["long", "neutral"])
            bear_count = sum(1 for signal in output.get("analyst_signals", {}).values() if signal.get("signal") == "short")
            neutral_count = sum(1 for signal in output.get("analyst_signals", {}).values() if signal.get("signal") == "neutral")
            self.backtest_logger.info(f"\nFinal Decision (Test - Neutral treated as Long): {action.upper()}, Qty: {target_qty}")
            print(f"{current_date_str:<12} {self.ticker:<6} {action:<6} {executed_quantity:>8} {current_price:>12.4f} {self.portfolio['cash']:>15.4f} {self.portfolio['stock']:>8} {total_value:>15.4f} {bull_count:>8} {bear_count:>8} {neutral_count:>8}")
        self.analyze_performance()

    def analyze_performance(self):
        """输出各项绩效指标，并分别保存两个图：
        1. 总横向资本变化图；
        2. 累计收益图（同时显示 Buy & Hold Return 曲线）。
        此外，将所有 performance metrics（含 Buy & Hold 对应指标）打印并保存为 Excel 文件。
        说明：
        - Daily Return: 当天组合价值相对于前一交易日的比例变动 ×100%
        - Cumulative Return: (当前组合价值/初始资本 - 1) ×100%
        - Buy & Hold: 假设在回测开始日以开盘价买入，持仓数 = 初始资本 / 当日开盘价；
            每日持仓价值 = 持仓数 × 当日开盘价，Buy & Hold Return = [(持仓价值/初始资本) - 1] ×100%
        """
        if not self.portfolio_values:
            self.backtest_logger.warning("No portfolio values to analyze")
            return
        try:
            # 创建 results 文件夹
            results_folder = "results"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                
            # 使用内置 "ggplot" 风格，避免使用 seaborn 风格
            plt.style.use('ggplot')

            performance_df = pd.DataFrame(self.portfolio_values)
            performance_df['Date'] = pd.to_datetime(performance_df['Date'])
            # 此处数据仅来自交易日 (>= self.start_date)
            performance_df = performance_df.sort_values("Date").set_index("Date")
            performance_df["Cumulative Return"] = (performance_df["Portfolio Value"] / self.initial_capital - 1) * 100
            performance_df["Portfolio Value (K)"] = performance_df["Portfolio Value"] / 1000

            # Buy & Hold：以回测开始日的开盘价作为基准
            trade_start_date = pd.to_datetime(self.start_date)
            if trade_start_date in performance_df.index:
                initial_price_for_bh = performance_df.loc[trade_start_date, "Price"]
            else:
                initial_price_for_bh = performance_df.iloc[0]["Price"]
            bh_df = performance_df[performance_df.index >= trade_start_date].copy()
            buy_and_hold_value = (self.initial_capital / initial_price_for_bh) * bh_df["Price"]
            buy_and_hold_return = (buy_and_hold_value / self.initial_capital - 1) * 100

            # 图1：展示组合资本变化 (Portfolio Value)
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            title1 = f"Portfolio Value Change - {self.ticker} ({self.start_date} to {self.end_date})"
            ax1.set_title(title1, fontsize=16)
            ax1.plot(performance_df.index, performance_df["Portfolio Value (K)"], label="Portfolio Value", 
                    marker='o', markersize=6, color='#1f77b4')
            ax1.set_ylabel("Portfolio Value (K)", fontsize=14)
            ax1.set_xlabel("Date", fontsize=14)
            for x, y in zip(performance_df.index, performance_df["Portfolio Value (K)"]):
                ax1.annotate(f"{y:.4f}K", (x, y), textcoords="offset points", xytext=(0, 10), 
                            ha='center', fontsize=8)
            ax1.legend(fontsize=12)
            fig1.autofmt_xdate()
            fig1.tight_layout()
            capital_filename = os.path.join(results_folder, f"backtest_capital_{self.ticker}_{self.start_date}_{self.end_date}.png")
            fig1.savefig(capital_filename, bbox_inches='tight', dpi=300)
            plt.close(fig1)

            # 图2：展示累计收益（Cumulative Return）及 Buy & Hold Return
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            title2 = f"Cumulative Return Change - {self.ticker} ({self.start_date} to {self.end_date})"
            ax2.set_title(title2, fontsize=16)
            ax2.plot(performance_df.index, performance_df["Cumulative Return"], label="Portfolio Return", 
                    marker='o', markersize=6, color='#2ca02c')
            ax2.plot(bh_df.index, buy_and_hold_return, label="Buy & Hold Return", 
                    marker='o', markersize=6, color='#d62728', linestyle='--')
            ax2.set_ylabel("Cumulative Return (%)", fontsize=14)
            ax2.set_xlabel("Date", fontsize=14)
            for x, y in zip(performance_df.index, performance_df["Cumulative Return"]):
                ax2.annotate(f"{y:.4f}%", (x, y), textcoords="offset points", xytext=(0, 10), 
                            ha='center', fontsize=8)
            ax2.legend(fontsize=12)
            fig2.autofmt_xdate()
            fig2.tight_layout()
            return_filename = os.path.join(results_folder, f"backtest_return_{self.ticker}_{self.start_date}_{self.end_date}.png")
            fig2.savefig(return_filename, bbox_inches='tight', dpi=300)
            plt.close(fig2)

            # 计算总体指标
            total_return = (self.portfolio["portfolio_value"] - self.initial_capital) / self.initial_capital
            metrics = []
            metrics.append(f"Initial Capital: {self.initial_capital:.4f}")
            metrics.append(f"Final Portfolio Value: {self.portfolio['portfolio_value']:.4f}")
            metrics.append(f"Portfolio Total Return: {(total_return * 100):.4f}%")
            daily_returns = performance_df["Daily Return"] / 100
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe_ratio = (mean_daily_return / std_daily_return) * (252 ** 0.5) if std_daily_return != 0 else 0
            metrics.append(f"Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")
            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (performance_df["Portfolio Value"] / rolling_max - 1) * 100
            max_drawdown = drawdown.min()
            metrics.append(f"Portfolio Maximum Drawdown: {max_drawdown:.4f}%")
            
            # Buy & Hold Metrics
            bh_final_value = buy_and_hold_value.iloc[-1]
            bh_total_return = (bh_final_value / self.initial_capital - 1) * 100
            metrics.append(f"Buy & Hold Final Value: {bh_final_value:.4f}")
            metrics.append(f"Buy & Hold Total Return: {bh_total_return:.4f}%")

            # 打印所有指标
            self.backtest_logger.info("\n" + "=" * 50)
            self.backtest_logger.info("Backtest Summary")
            self.backtest_logger.info("=" * 50)
            for m in metrics:
                self.backtest_logger.info(m)
                print(m)
            
            # 将指标保存为 Excel 文件
            metrics_df = pd.DataFrame(metrics, columns=["Metric"])
            excel_filename = os.path.join(results_folder, f"backtest_metrics_{self.ticker}_{self.start_date}_{self.end_date}.xlsx")
            metrics_df.to_excel(excel_filename, index=False)

            return performance_df
        except Exception as e:
            self.backtest_logger.error(f"Error in performance analysis: {str(e)}")
            print(f"Error in performance analysis: {str(e)}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run backtest simulation')
    parser.add_argument('--ticker', type=str, required=False, default='CNY', help='FX code (e.g., JPY)')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--direction', type=str, default='long', help='Action direction (long/short)')
    parser.add_argument('--end-date', type=str, default='2025-02-04', help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, default='2025-01-02', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--event-prompt', type=str, default='PBOC sets stronger yuan fixing than expected', help='Event that we would like to trade')

    args = parser.parse_args()

    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        direction=args.direction,
        event_prompt=args.event_prompt
    )

    backtester.run_backtest()