#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze backtest results - Yearly performance metrics
Simplified version - JSON format only

Usage:
    1. Modify CONFIGURATION section below
    2. Run: python plot_account_equity.py
    
Example:
    BACKTEST_FILE = 'backtest_v8_21.json'
"""

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import os
import json
import sys
from pathlib import Path

# Add project path for importing BacktestMarketClient
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'future_v_0_1'))

from market.backtest_client import BacktestMarketClient

# ============================================================================
# CONFIGURATION - 修改这里的参数
# ============================================================================

# 回测结果文件路径（仅支持.json格式）
BACKTEST_FILE = 'backtest_v8_26.json'

# 初始资金（如果从JSON读取则会自动使用JSON中的值）
INITIAL_CAPITAL = 1000000

# 无风险利率（年化）
RISK_FREE_RATE = 0.03

# SPY和QQQ历史数据文件（使用数据库目录下的文件）
SPY_DATA_FILE = 'future_v_0_1/database/spy_historical_data.csv'
QQQ_DATA_FILE = 'future_v_0_1/database/qqq_historical_data.csv'

# ============================================================================

# Helper functions
def to_date(dt):
    """Convert pd.Timestamp or datetime to date object"""
    return dt.date() if hasattr(dt, 'date') else dt

def calculate_sharpe_ratio(daily_returns, annual_rf):
    """
    Calculate annualized Sharpe ratio
    
    Args:
        daily_returns: pd.Series of daily simple returns (decimal format, e.g., 0.001 for 0.1%)
        annual_rf: Annual risk-free rate (decimal, e.g., 0.03 for 3%)
    
    Returns:
        float: Annualized Sharpe ratio
    """
    if len(daily_returns) == 0 or daily_returns.std(ddof=1) == 0:
        return 0.0
    
    # Convert annual risk-free rate to daily (compound method)
    daily_rf = (1 + annual_rf)**(1/252) - 1
    
    # Calculate daily excess returns
    excess = daily_returns - daily_rf
    
    # Annualized Sharpe (sample std with ddof=1, annualize with sqrt(252))
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    
    return sharpe

def load_backtest_data(filename):
    """
    Load backtest data from JSON file
    
    Returns:
        tuple: (trades_df, initial_capital, report_dict)
    """
    if not os.path.exists(filename):
        print(f"Error: Cannot find {filename}")
        return None, None, None
    
    print(f"Loading backtest data from JSON: {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read JSON file: {e}")
        return None, None, None
    
    initial_capital = data.get('initial_cash', INITIAL_CAPITAL)
    trades_list = data.get('trades', [])
    report = data.get('report', {})
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades_list)
    
    if len(trades_df) == 0:
        print("Warning: No trades found in backtest file")
        return trades_df, initial_capital, report
    
    # Parse time - handle mixed timezone formats
    # JSON times have mixed formats: -04:00 (EDT), -05:00 (EST), etc.
    parsed_times = []
    for time_str in trades_df['time']:
        try:
            # Parse time and strip timezone info (all times are Eastern Time)
            dt = pd.to_datetime(time_str)
            dt_naive = dt.tz_localize(None) if dt.tz is not None else dt
            parsed_times.append(dt_naive)
        except Exception as e:
            print(f"Warning: Failed to parse time '{time_str}': {e}")
            parsed_times.append(pd.NaT)
    
    trades_df['time'] = pd.Series(parsed_times)
    
    print(f"Loaded backtest: {len(trades_df)} trades, capital=${initial_capital:,.0f}")
    if len(trades_df) > 0:
        print(f"  Date range: {trades_df['time'].iloc[0]} to {trades_df['time'].iloc[-1]}")
    
    return trades_df, initial_capital, report

def load_benchmark_data(filename, symbol_name):
    """Load SPY or QQQ data from CSV file"""
    if not os.path.exists(filename):
        print(f"Warning: Cannot find {filename}")
        return None
    
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data.set_index('Date', inplace=True)
    
    print(f"Loaded {symbol_name}: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    return data['Close']

def calculate_equity_curve(trades_df, initial_capital, cache_dir='future_v_0_1/database/data_cache'):
    """
    Calculate daily equity curve by marking positions to market using real closing prices
    
    Args:
        trades_df: DataFrame with trade records (from JSON)
        initial_capital: Initial account balance
        cache_dir: Directory for price data cache (parquet files)
        
    Returns:
        DataFrame with columns: Date, Equity
    """
    # Initialize tracking
    cash = initial_capital
    # positions structure: {symbol: {'shares': int}}
    positions = {}
    
    # Sort trades by time
    trades_df = trades_df.sort_values('time').reset_index(drop=True)
    
    # Get date range
    start_date = to_date(trades_df['time'].iloc[0])
    end_date = to_date(trades_df['time'].iloc[-1])
    
    print(f"Calculating equity curve from {start_date} to {end_date}...")
    
    # Get NYSE trading calendar
    nyse = mcal.get_calendar('NYSE')
    trading_days_index = nyse.valid_days(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    trading_dates = [d.date() for d in pd.to_datetime(trading_days_index)]
    
    print(f"NYSE trading days: {len(trading_dates)} days")
    
    # Extract all symbols from trades
    all_symbols = sorted(trades_df['symbol'].unique())
    print(f"Symbols traded: {len(all_symbols)}")
    
    # Create market client for price queries (uses local cache)
    print(f"Initializing market client with cache: {cache_dir}")
    # Note: initial_cash doesn't matter here, we only use client for price queries
    client = BacktestMarketClient(cache_dir=cache_dir, initial_cash=100000)
    client.connect()
    
    print("Note: Price data will be loaded on-demand (cache-first, API as fallback)")
    print("  - Cached data: instant")
    print("  - Missing data: fetched from API and cached for future use")
    
    # Process trades and build equity curve
    equity_curve = []
    dates = []
    trade_idx = 0
    
    # Track last known prices for each symbol (handles delisting/halts)
    last_known_prices = {}
    
    total_days = len(trading_dates)
    print(f"\nProcessing {total_days} trading days...")
    
    for day_idx, current_date in enumerate(trading_dates, 1):
        # Progress indicator (every 50 days or at key milestones)
        if day_idx == 1 or day_idx % 50 == 0 or day_idx == total_days:
            progress_pct = (day_idx / total_days) * 100
            print(f"  Progress: {day_idx}/{total_days} days ({progress_pct:.1f}%) - {current_date}")
        
        # Process all trades for this date
        while trade_idx < len(trades_df):
            trade = trades_df.iloc[trade_idx]
            trade_date = to_date(trade['time'])
            
            if trade_date > current_date:
                break
            
            symbol = trade['symbol']
            shares = trade['shares']
            
            if trade['type'] == 'BUY':
                if symbol not in positions:
                    positions[symbol] = {'shares': 0}
                positions[symbol]['shares'] += shares
                cash -= trade['amount']
            else:  # SELL
                if symbol in positions:
                    positions[symbol]['shares'] -= shares
                    if positions[symbol]['shares'] <= 0:
                        del positions[symbol]
                cash += trade['amount']
            
            trade_idx += 1
        
        # Mark to market using real closing prices (on-demand loading with cache-first)
        position_value = 0
        for symbol, pos in positions.items():
            # get_day_close_price will:
            # 1. Check local parquet cache first (instant)
            # 2. If missing, fetch from API and save to cache
            # 3. Return closing price in ET timezone, or None if delisted/halted
            closing_price = client.get_day_close_price(symbol, current_date, max_lookback_days=5)
            
            if closing_price:
                # Valid price: use it and update last known price
                last_known_prices[symbol] = closing_price
                position_value += pos['shares'] * closing_price
            elif symbol in last_known_prices:
                # No price today (delisted/halted), use last known price
                closing_price = last_known_prices[symbol]
                position_value += pos['shares'] * closing_price
                if day_idx == 1 or day_idx % 50 == 0:  # Only log occasionally to avoid spam
                    print(f"  Note: Using last known price for {symbol} (${closing_price:.2f})")
            else:
                # No price ever (shouldn't happen in normal cases)
                print(f"Warning: No price for {symbol} on {current_date} and no last known price")
        
        total_equity = cash + position_value
        equity_curve.append(total_equity)
        dates.append(pd.Timestamp(current_date))
    
    client.disconnect()
    
    # Create DataFrame
    equity_df = pd.DataFrame({
        'Date': dates,
        'Equity': equity_curve
    })
    
    # Debug: Print final state
    print(f"Final: cash=${cash:,.0f}, positions={len(positions)}", end="")
    if positions:
        print(f", position_value=${position_value:,.0f}")
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos['shares']} shares")
    else:
        print()
    
    return equity_df

def calculate_metrics(equity_df, initial_capital, risk_free_rate=0.03):
    """Calculate performance metrics"""
    
    # Basic metrics
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # Max drawdown
    peak = equity_df['Equity'].iloc[0]
    max_drawdown = 0
    for equity in equity_df['Equity']:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate daily returns
    daily_returns = equity_df['Equity'].pct_change().dropna()
    
    # Sharpe Ratio (annualized)
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def calculate_yearly_metrics(equity_df, initial_capital, spy_prices=None, qqq_prices=None, risk_free_rate=0.03):
    """Calculate yearly performance metrics including benchmark Sharpe ratios"""
    
    equity_df = equity_df.copy()
    equity_df['Year'] = equity_df['Date'].dt.year
    
    yearly_stats = []
    previous_end_equity = initial_capital
    
    for year in sorted(equity_df['Year'].unique()):
        year_data = equity_df[equity_df['Year'] == year]
        
        if len(year_data) == 0:
            continue
        
        # Get year date range
        year_start = year_data['Date'].iloc[0]
        year_end = year_data['Date'].iloc[-1]
        
        # Get equity at start and end
        start_equity = previous_end_equity
        end_equity = year_data['Equity'].iloc[-1]
        previous_end_equity = end_equity
        
        # Calculate return
        year_return = (end_equity - start_equity) / start_equity * 100
        
        # Calculate max drawdown for this year
        peak = start_equity
        max_drawdown = 0
        for equity in year_data['Equity']:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio for this year
        daily_returns = year_data['Equity'].pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
        
        # Calculate SPY metrics for this year
        spy_sharpe = 0
        spy_return = 0
        if spy_prices is not None:
            spy_year = spy_prices[(spy_prices.index >= year_start) & (spy_prices.index <= year_end)]
            if len(spy_year) > 1:
                # Calculate SPY return
                spy_return = (spy_year.iloc[-1] - spy_year.iloc[0]) / spy_year.iloc[0] * 100
                # Calculate SPY Sharpe
                spy_returns = spy_year.pct_change().dropna()
                spy_sharpe = calculate_sharpe_ratio(spy_returns, risk_free_rate)
        
        # Calculate QQQ metrics for this year
        qqq_sharpe = 0
        qqq_return = 0
        if qqq_prices is not None:
            qqq_year = qqq_prices[(qqq_prices.index >= year_start) & (qqq_prices.index <= year_end)]
            if len(qqq_year) > 1:
                # Calculate QQQ return
                qqq_return = (qqq_year.iloc[-1] - qqq_year.iloc[0]) / qqq_year.iloc[0] * 100
                # Calculate QQQ Sharpe
                qqq_returns = qqq_year.pct_change().dropna()
                qqq_sharpe = calculate_sharpe_ratio(qqq_returns, risk_free_rate)
        
        yearly_stats.append({
            'year': year,
            'start_equity': start_equity,
            'end_equity': end_equity,
            'return': year_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'spy_return': spy_return,
            'spy_sharpe': spy_sharpe,
            'qqq_return': qqq_return,
            'qqq_sharpe': qqq_sharpe
        })
    
    return pd.DataFrame(yearly_stats)

def analyze_backtest():
    """Main analysis function"""
    
    print("="*70)
    print("Backtest Performance Analysis")
    print(f"  File: {BACKTEST_FILE}")
    print(f"  Risk-Free Rate: {RISK_FREE_RATE*100:.1f}%")
    print("="*70)
    
    # Load backtest data
    trades_df, initial_capital, report = load_backtest_data(BACKTEST_FILE)
    
    if trades_df is None or len(trades_df) == 0:
        print("Error: No trades found")
        return
    
    # Load benchmark data
    spy_prices = load_benchmark_data(SPY_DATA_FILE, 'SPY')
    qqq_prices = load_benchmark_data(QQQ_DATA_FILE, 'QQQ')
    
    if spy_prices is None or qqq_prices is None:
        print("Warning: Cannot load benchmark data, Sharpe comparison will be skipped")
    
    # Calculate equity curve
    equity_df = calculate_equity_curve(trades_df, initial_capital)
    
    print(f"Equity curve: ${equity_df['Equity'].iloc[0]:,.0f} → ${equity_df['Equity'].iloc[-1]:,.0f}")
    print()
    
    # Calculate yearly metrics
    yearly_stats = calculate_yearly_metrics(equity_df, initial_capital, spy_prices, qqq_prices, RISK_FREE_RATE)
    
    # Print yearly performance table
    print("="*150)
    print("YEARLY PERFORMANCE")
    print("="*150)
    print(f"{'Year':<6} {'Start Equity':<18} {'End Equity':<18} {'Return':<12} {'Max DD':<12} {'Sharpe':<10} "
          f"{'SPY Ret':<12} {'SPY Sharpe':<12} {'QQQ Ret':<12} {'QQQ Sharpe':<12}")
    print("-"*150)
    
    for _, row in yearly_stats.iterrows():
        print(f"{row['year']:<6} ${row['start_equity']:>16,.0f}  ${row['end_equity']:>16,.0f}  "
              f"{row['return']:>+10.2f}%  {row['max_drawdown']:>10.2f}%  {row['sharpe_ratio']:>8.2f}  "
              f"{row['spy_return']:>+10.2f}%  {row['spy_sharpe']:>10.2f}  "
              f"{row['qqq_return']:>+10.2f}%  {row['qqq_sharpe']:>10.2f}")
    
    print("="*150)

def main():
    """Main function"""
    analyze_backtest()

if __name__ == "__main__":
    main()
