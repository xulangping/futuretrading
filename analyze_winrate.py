#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析回测结果胜率 - 多维度统计

分析维度：
1. DTE（到期天数）
2. OTM比例
3. 交易时间段
4. 期权share_eqv
5. 买入当日成交量

Usage:
    python analyze_winrate.py
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
import os
from collections import defaultdict

# Polygon API配置
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY')

def load_backtest_results(json_file: str) -> Dict:
    """加载回测结果JSON"""
    print(f"📖 加载回测结果: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_signal_csv(csv_file: str) -> pd.DataFrame:
    """加载信号CSV数据"""
    print(f"📖 加载信号数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 解析日期时间（CSV是上海时间，需要转成纽约时间）
    df['datetime_shanghai'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['datetime_shanghai'] = df['datetime_shanghai'].dt.tz_localize('Asia/Shanghai')
    
    # 转换为纽约时间
    df['datetime'] = df['datetime_shanghai'].dt.tz_convert('America/New_York')
    
    # 解析date为纽约时区的日期
    df['date_ny'] = df['datetime'].dt.date
    
    # 计算正确的DTE：expiry - date（纽约时间）
    df['expiry_date'] = pd.to_datetime(df['expiry'])
    df['dte_days'] = (df['expiry_date'] - pd.to_datetime(df['date_ny'])).dt.days
    
    # 解析OTM百分比
    df['otm_percent'] = df['otm_pct'].str.replace('%', '').astype(float)
    
    # 解析share_eqv（去掉逗号）
    df['share_eqv_num'] = df['share_eqv'].str.replace(',', '').astype(float)
    
    # 提取交易时间的小时（纽约时间）
    df['trade_hour'] = df['datetime'].dt.hour
    
    # 股价（spot列已经是美元格式）
    df['stock_price'] = df['spot'].str.replace('$', '').astype(float)
    
    print(f"✅ 加载了 {len(df)} 条信号")
    print(f"   时区转换: 上海时间 → 纽约时间")
    print(f"   DTE范围: {df['dte_days'].min()}-{df['dte_days'].max()}天")
    
    return df

def parse_trades(backtest_data: Dict) -> pd.DataFrame:
    """解析交易记录，匹配买入和卖出"""
    trades = backtest_data['trades']
    
    # 分离买入和卖出
    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']
    
    print(f"📊 买入: {len(buys)} 笔, 卖出: {len(sells)} 笔")
    
    # 构建交易对
    trade_pairs = []
    sell_dict = defaultdict(list)
    
    # 按symbol组织卖出记录
    for sell in sells:
        sell_dict[sell['symbol']].append(sell)
    
    # 匹配买入和卖出
    for buy in buys:
        symbol = buy['symbol']
        buy_time = pd.to_datetime(buy['time'])
        
        # 查找对应的卖出
        matched_sell = None
        if symbol in sell_dict:
            for sell in sell_dict[symbol]:
                sell_time = pd.to_datetime(sell['time'])
                # 卖出时间应该晚于买入时间
                if sell_time > buy_time:
                    matched_sell = sell
                    sell_dict[symbol].remove(sell)  # 移除已匹配的
                    break
        
        # 计算盈亏
        if matched_sell:
            pnl = matched_sell.get('profit', 0)
            pnl_pct = (matched_sell['price'] - buy['price']) / buy['price']
            win = pnl > 0
        else:
            # 未匹配到卖出（可能还持仓）
            pnl = 0
            pnl_pct = 0
            win = None
        
        trade_pairs.append({
            'symbol': symbol,
            'buy_time': buy_time,
            'buy_price': buy['price'],
            'sell_time': pd.to_datetime(matched_sell['time']) if matched_sell else None,
            'sell_price': matched_sell['price'] if matched_sell else None,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'win': win,
            'strike': buy.get('strike'),
            'expiry': buy.get('expiry'),
            'shares': buy.get('shares'),
            'position_ratio': buy.get('position_ratio')
        })
    
    df_trades = pd.DataFrame(trade_pairs)
    # 只保留已完成的交易（有卖出的）
    df_trades = df_trades[df_trades['win'].notna()].copy()
    
    print(f"✅ 匹配了 {len(df_trades)} 笔完整交易")
    print(f"   胜率: {df_trades['win'].sum()}/{len(df_trades)} = {df_trades['win'].mean():.1%}")
    
    return df_trades

def match_trades_with_signals(df_trades: pd.DataFrame, df_signals: pd.DataFrame) -> pd.DataFrame:
    """将交易记录与信号数据匹配
    
    匹配策略：
    1. symbol + strike + expiry 精确匹配
    2. 时间验证：交易时间 - 10分钟 ≈ 信号时间（考虑延迟）
    3. 如果有多个匹配，选择时间最接近的
    """
    print("🔗 匹配交易记录与信号数据...")
    print("   匹配逻辑: 交易时间 - 10分钟 ≈ CSV信号时间")
    
    matched_trades = []
    
    for idx, trade in df_trades.iterrows():
        trade_expiry = pd.to_datetime(trade['expiry'])
        
        # 计算预期的信号时间（交易时间 - 10分钟）
        expected_signal_time = trade['buy_time'] - timedelta(minutes=10)
        
        # 方法1：精确匹配 symbol + strike + expiry
        matched_signals = df_signals[
            (df_signals['ticker'] == trade['symbol']) &
            (df_signals['strike'] == trade['strike']) &
            (df_signals['expiry_date'] == trade_expiry)
        ]
        
        if len(matched_signals) > 0:
            # 方法2：在匹配结果中，找时间最接近的（±2分钟容差）
            matched_signals_copy = matched_signals.copy()
            matched_signals_copy['time_diff'] = (matched_signals_copy['datetime'] - expected_signal_time).abs()
            matched_signals_sorted = matched_signals_copy.sort_values('time_diff')
            
            # 取时间最近的信号（通常应该只有1个）
            best_match = matched_signals_sorted.iloc[0]
            time_diff_minutes = best_match['time_diff'].total_seconds() / 60
            
            matched_trades.append({
                **trade.to_dict(),
                'dte': best_match['dte_days'],
                'otm_pct': best_match['otm_percent'],
                'trade_hour': best_match['trade_hour'],
                'share_eqv': best_match['share_eqv_num'],
                'stock_price': best_match['stock_price'],
                'date': best_match['date'],
                'signal_time': best_match['datetime'],
                'time_diff_min': time_diff_minutes
            })
        else:
            # 没匹配到信号
            matched_trades.append({
                **trade.to_dict(),
                'dte': None,
                'otm_pct': None,
                'trade_hour': None,
                'share_eqv': None,
                'stock_price': None,
                'date': None,
                'signal_time': None,
                'time_diff_min': None
            })
    
    df_matched = pd.DataFrame(matched_trades)
    matched_count = df_matched['dte'].notna().sum()
    total_count = len(df_matched)
    print(f"✅ 成功匹配 {matched_count}/{total_count} 笔交易 ({matched_count/total_count:.1%})")
    
    # 显示时间差统计
    if matched_count > 0:
        avg_time_diff = df_matched['time_diff_min'].mean()
        max_time_diff = df_matched['time_diff_min'].max()
        print(f"   时间差: 平均{avg_time_diff:.1f}分钟, 最大{max_time_diff:.1f}分钟")
    
    if matched_count < total_count * 0.5:
        print(f"⚠️  警告：匹配率低于50%，可能CSV数据不完整")
        print(f"   CSV日期范围: {df_signals['date'].min()} 到 {df_signals['date'].max()}")
        print(f"   交易日期范围: {df_matched['buy_time'].min()} 到 {df_matched['buy_time'].max()}")
    
    return df_matched

def get_volume_from_polygon(symbol: str, date: str) -> Optional[float]:
    """从Polygon API获取指定日期的成交量
    
    Args:
        symbol: 股票代码
        date: 日期 (YYYY-MM-DD)
    
    Returns:
        成交量，如果获取失败返回None
    """
    try:
        import requests
        
        url = f"https://api.polygon.io/v1/open-close/{symbol}/{date}"
        params = {'apiKey': POLYGON_API_KEY, 'adjusted': 'true'}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('volume')
        else:
            return None
    except Exception as e:
        print(f"⚠️ 获取 {symbol} {date} 成交量失败: {e}")
        return None

def add_volume_data(df: pd.DataFrame, use_polygon: bool = False) -> pd.DataFrame:
    """添加成交量数据
    
    Args:
        df: 交易数据DataFrame
        use_polygon: 是否使用Polygon API获取成交量（需要API key）
    
    Returns:
        添加了volume列的DataFrame
    """
    if not use_polygon:
        print("⏭️ 跳过成交量数据获取（use_polygon=False）")
        df['volume'] = None
        return df
    
    print("📡 从Polygon获取成交量数据...")
    
    volumes = []
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"   进度: {idx}/{total} ({idx/total:.1%})")
        
        if pd.notna(row['date']) and pd.notna(row['symbol']):
            volume = get_volume_from_polygon(row['symbol'], row['date'])
            volumes.append(volume)
        else:
            volumes.append(None)
    
    df['volume'] = volumes
    print(f"✅ 获取了 {pd.Series(volumes).notna().sum()}/{total} 个成交量数据")
    
    return df

def categorize_dte(dte):
    """DTE分类"""
    if pd.isna(dte):
        return '未知'
    elif dte <= 7:
        return '0-7天'
    elif dte <= 30:
        return '8-30天'
    elif dte <= 60:
        return '31-60天'
    elif dte <= 90:
        return '61-90天'
    else:
        return '>90天'

def categorize_otm(otm_pct):
    """OTM分类"""
    if pd.isna(otm_pct):
        return '未知'
    elif otm_pct < 0:
        return 'ITM(<0%)'
    elif otm_pct <= 5:
        return 'ATM(0-5%)'
    elif otm_pct <= 10:
        return 'OTM(5-10%)'
    elif otm_pct <= 20:
        return 'OTM(10-20%)'
    else:
        return 'OTM(>20%)'

def categorize_time(hour):
    """交易时间分类"""
    if pd.isna(hour):
        return '未知'
    elif hour < 10:
        return '开盘(9:30-10:00)'
    elif hour < 12:
        return '上午(10:00-12:00)'
    elif hour < 14:
        return '午后(12:00-14:00)'
    elif hour < 15:
        return '尾盘前(14:00-15:00)'
    else:
        return '尾盘(15:00-16:00)'

def categorize_share_eqv(share_eqv):
    """Share_eqv分类（流动性）"""
    if pd.isna(share_eqv):
        return '未知'
    elif share_eqv < 10000:
        return '<1万'
    elif share_eqv < 50000:
        return '1-5万'
    elif share_eqv < 100000:
        return '5-10万'
    elif share_eqv < 500000:
        return '10-50万'
    else:
        return '>50万'

def categorize_volume(volume):
    """成交量分类"""
    if pd.isna(volume):
        return '未知'
    elif volume < 1000000:
        return '<100万'
    elif volume < 5000000:
        return '100-500万'
    elif volume < 10000000:
        return '500-1000万'
    else:
        return '>1000万'

def analyze_by_dimension(df: pd.DataFrame, dimension: str, categorize_func) -> pd.DataFrame:
    """按指定维度分析胜率"""
    df[f'{dimension}_cat'] = df[dimension].apply(categorize_func)
    
    stats = df.groupby(f'{dimension}_cat').agg({
        'win': ['count', 'sum', 'mean'],
        'pnl': 'sum',
        'pnl_pct': 'mean'
    }).round(4)
    
    stats.columns = ['交易数', '胜场', '胜率', '总盈亏', '平均盈亏%']
    stats['胜率'] = stats['胜率'].apply(lambda x: f"{x:.1%}")
    stats['平均盈亏%'] = stats['平均盈亏%'].apply(lambda x: f"{x:+.2%}")
    stats['总盈亏'] = stats['总盈亏'].apply(lambda x: f"${x:+,.0f}")
    
    return stats.sort_values('交易数', ascending=False)

def main():
    """主函数"""
    print("\n" + "="*80)
    print("📊 回测结果多维度胜率分析")
    print("="*80 + "\n")
    
    # 配置
    BACKTEST_FILE = 'backtest_v8_all.json'
    CSV_FILE = 'future_v_0_1/database/merged_strategy_v1_calls_bell_2023M3_2025M10.csv'
    USE_POLYGON = False  # 是否使用Polygon获取成交量（需要API key）
    
    # 1. 加载数据
    backtest_data = load_backtest_results(BACKTEST_FILE)
    df_signals = load_signal_csv(CSV_FILE)
    
    # 2. 解析交易记录
    df_trades = parse_trades(backtest_data)
    
    # 3. 匹配交易与信号
    df_matched = match_trades_with_signals(df_trades, df_signals)
    
    # 4. 添加成交量数据（可选）
    df_matched = add_volume_data(df_matched, use_polygon=USE_POLYGON)
    
    # 5. 多维度分析
    print("\n" + "="*80)
    print("📈 多维度胜率分析")
    print("="*80 + "\n")
    
    dimensions = [
        ('dte', '1️⃣  DTE（到期天数）分析', categorize_dte),
        ('otm_pct', '2️⃣  OTM比例分析', categorize_otm),
        ('trade_hour', '3️⃣  交易时间段分析', categorize_time),
        ('share_eqv', '4️⃣  期权流动性（Share_eqv）分析', categorize_share_eqv),
    ]
    
    if USE_POLYGON:
        dimensions.append(('volume', '5️⃣  买入当日成交量分析', categorize_volume))
    
    results = {}
    
    for dim, title, categorize_func in dimensions:
        print(f"\n{title}")
        print("-" * 80)
        stats = analyze_by_dimension(df_matched, dim, categorize_func)
        print(stats)
        results[dim] = stats
    
    # 6. 综合摘要
    print("\n" + "="*80)
    print("📋 综合摘要")
    print("="*80)
    
    total_trades = len(df_matched)
    total_wins = df_matched['win'].sum()
    overall_winrate = df_matched['win'].mean()
    total_pnl = df_matched['pnl'].sum()
    avg_pnl_pct = df_matched['pnl_pct'].mean()
    
    print(f"""
总交易数: {total_trades}
总胜场: {total_wins}
总体胜率: {overall_winrate:.1%}
总盈亏: ${total_pnl:+,.2f}
平均盈亏%: {avg_pnl_pct:+.2%}

数据完整性:
  - DTE数据: {df_matched['dte'].notna().sum()}/{total_trades} ({df_matched['dte'].notna().mean():.1%})
  - OTM数据: {df_matched['otm_pct'].notna().sum()}/{total_trades} ({df_matched['otm_pct'].notna().mean():.1%})
  - 时间数据: {df_matched['trade_hour'].notna().sum()}/{total_trades} ({df_matched['trade_hour'].notna().mean():.1%})
  - Share_eqv数据: {df_matched['share_eqv'].notna().sum()}/{total_trades} ({df_matched['share_eqv'].notna().mean():.1%})
  - 成交量数据: {df_matched['volume'].notna().sum()}/{total_trades} ({df_matched['volume'].notna().mean():.1%})
    """)
    
    # 7. 保存详细数据
    output_file = 'winrate_analysis_detail.csv'
    df_matched.to_csv(output_file, index=False)
    print(f"\n✅ 详细数据已保存到: {output_file}")
    
    # 8. 保存统计摘要
    summary_file = 'winrate_analysis_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("回测结果多维度胜率分析摘要\n")
        f.write("="*80 + "\n\n")
        
        for dim, title, _ in dimensions:
            f.write(f"\n{title}\n")
            f.write("-" * 80 + "\n")
            f.write(results[dim].to_string())
            f.write("\n")
    
    print(f"✅ 统计摘要已保存到: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

