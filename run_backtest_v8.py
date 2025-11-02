#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8策略回测脚本 - 从CSV直接读取，固定仓位版本
与V7架构一致的分钟循环，但数据源改为merged_strategy_v1_calls_bell_*.csv
"""

import sys
import yaml
import csv
import json
import logging
import pandas as pd
import pandas_market_calendars as mcal
import pytz
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'future_v_0_1'))

from strategy.v8 import StrategyV8
from strategy.strategy import StrategyContext, SignalEvent
from market.backtest_client import BacktestMarketClient


class BacktestRunnerV8:
    """V8策略回测运行器"""
    
    def __init__(self, csv_file: str, stock_data_dir: str, config: Dict, initial_cash: float = 100000.0):
        """
        初始化回测运行器
        
        Args:
            csv_file: merged_strategy_v1_calls_bell CSV文件路径
            stock_data_dir: 股价数据目录（用于backtest_client）
            config: 配置字典
            initial_cash: 初始资金
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.csv_file = Path(csv_file)
        self.stock_data_dir = stock_data_dir
        self.config = config
        self.initial_cash = initial_cash
        
        # 从config读取交易成本参数
        cost_cfg = {
            'slippage': 0.0005,
            'commission_per_share': 0.005,
            'min_commission': 1.0
        }
        
        # 创建回测客户端
        self.market_client = BacktestMarketClient(
            stock_data_dir=stock_data_dir,
            initial_cash=initial_cash,
            slippage=cost_cfg.get('slippage', 0.0005),
            commission_per_share=cost_cfg.get('commission_per_share', 0.005),
            min_commission=cost_cfg.get('min_commission', 1.0)
        )
        
        # 创建策略
        strategy_context = StrategyContext(
            cfg=config,
            logger=logging.getLogger('StrategyV8')
        )
        self.strategy = StrategyV8(strategy_context)
        
        # 记录
        self.trade_records = []
        self.signal_records = []
        self.position_entry_times = {}  # {symbol: entry_time}
        
        self.logger.info(
            f"V8回测运行器初始化完成: "
            f"CSV={csv_file}, 资金=${initial_cash:,.2f}"
        )
    
    def load_signals_from_csv(self) -> List[Dict]:
        """
        从merged_strategy CSV加载信号
        
        返回：
            信号列表（按时间排序），每个元素包含：
            {
                'signal': SignalEvent,
                'ticker': str,
                'strike': float,
                'expiry': date,
                'file': Path
            }
        """
        self.logger.info(f"从CSV加载期权信号: {self.csv_file}")
        signals = []
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, 1):
                    try:
                        signal = self._parse_signal_row(row)
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        self.logger.debug(f"行{idx}解析失败: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"加载CSV失败: {e}")
            return []
        
        # 按时间排序
        signals.sort(key=lambda x: x['signal'].event_time_et)
        
        self.logger.info(f"共加载 {len(signals)} 个信号")
        return signals
    
    def _parse_signal_row(self, row: Dict) -> Optional[Dict]:
        """
        解析CSV行为SignalEvent
        
        CSV字段：date, time, ticker, strike, option_type, expiry, premium, spot, ...
        时间是北京时间，需要转换为ET时间
        """
        try:
            ticker = row['ticker']
            date_str = row['date']
            time_str = row['time']
            strike = float(row['strike'])
            expiry_str = row['expiry']
            
            # 解析premium: 格式 "$592K" -> 592000
            premium_str = row['premium'].strip()
            premium = self._parse_usd_value(premium_str)
            
            # 解析price: CSV中price是股票价格（用于计算OTM）
            # 格式示例: "$145.48" -> 145.48
            stock_price_str = row['price'].strip().replace('$', '').replace(',', '')
            stock_price = float(stock_price_str)
            
            # 解析spot: CSV中spot是期权的bid-ask价格，这里取的是ask价格
            # 格式示例: "$0.65" -> 0.65
            spot_str = row['spot'].strip().replace('$', '').replace(',', '')
            spot = float(spot_str)
            
            # 北京时间 → ET时间
            datetime_str = f"{date_str} {time_str}"
            signal_time_cn = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            cn_tz = pytz.timezone('Asia/Shanghai')
            signal_time_cn = cn_tz.localize(signal_time_cn)
            et_tz = pytz.timezone('America/New_York')
            signal_time_et = signal_time_cn.astimezone(et_tz)
            
            # 添加10分钟延迟（在这里直接加到信号时间上）
            signal_time_et = signal_time_et + timedelta(minutes=10)
            
            # 如果延迟后超过16:00，使用15:59:30
            if signal_time_et.hour >= 16:
                signal_time_et = signal_time_et.replace(hour=15, minute=59, second=30)
            
            # 解析expiry日期
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            
            # 解析earnings: 距离earnings日期的天数（整数）
            try:
                earnings = int(row.get('earnings', '')) if row.get('earnings') else None
            except (ValueError, TypeError):
                earnings = None
            
            # 创建SignalEvent (加入strike, expiry, spot, stock_price)
            signal_event = SignalEvent(
                event_id=f"{ticker}_{signal_time_et.strftime('%Y%m%d%H%M%S')}",
                symbol=ticker,
                premium_usd=premium,
                ask=spot,
                chain_id=f"{ticker}_{expiry_str}",
                event_time_cn=signal_time_cn,
                event_time_et=signal_time_et,
                strike=strike,
                expiry=expiry_date,
                spot=spot,  # option price from signal
                stock_price=stock_price,  # stock price from CSV (for OTM calculation)
                iv_pct=float(row.get('iv_pct', '0').rstrip('%')) if row.get('iv_pct') else None,  # IV percentage from CSV
                earnings=earnings # 添加earnings字段
            )
            
            return {
                'signal': signal_event,
                'ticker': ticker,
                'strike': strike,
                'expiry': expiry_date,
                'file': self.csv_file
            }
            
        except Exception as e:
            self.logger.debug(f"解析信号失败: {e}")
            return None
    
    def _parse_usd_value(self, value_str: str) -> float:
        """
        解析美元值（支持K和M后缀）
        
        格式示例：
        - "$592K" -> 592000
        - "$1.5M" -> 1500000
        - "$100" -> 100
        """
        try:
            # 移除 $ 符号
            value_str = value_str.strip().replace('$', '').strip()
            
            # 处理 K 和 M 后缀
            if value_str.endswith('K'):
                return float(value_str[:-1]) * 1000
            elif value_str.endswith('M'):
                return float(value_str[:-1]) * 1000000
            else:
                return float(value_str)
        except Exception as e:
            self.logger.warning(f"解析USD值失败: {value_str}, 错误: {e}")
            return 0.0
    
    def run(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """运行回测 - 分钟循环架构"""
        
        # 加载信号
        all_signals = self.load_signals_from_csv()
        
        if len(all_signals) == 0:
            self.logger.warning("无信号数据")
            return
        
        # Group signals by 20-second intervals
        # 注意：信号时间已在load_signals_from_csv中延迟10分钟
        signals_by_time = {}
        for sig_data in all_signals:
            sig_time = sig_data['signal'].event_time_et  # 已包含10分钟延迟
            # 向下取整到20秒间隔（简化版本）
            seconds = sig_time.second
            interval_second = (seconds // 20) * 20
            sig_time_interval = sig_time.replace(second=interval_second, microsecond=0)
            
            if sig_time_interval not in signals_by_time:
                signals_by_time[sig_time_interval] = []
            signals_by_time[sig_time_interval].append(sig_data)
        
        # 获取回测日期范围
        all_times = sorted(signals_by_time.keys())
        default_start_date = all_times[0].date()
        default_end_date = all_times[-1].date()
        
        # 使用方法参数中的日期，或使用默认值
        actual_start_date = start_date.date() if start_date and hasattr(start_date, 'date') else (start_date if start_date else default_start_date)
        actual_end_date = end_date.date() if end_date and hasattr(end_date, 'date') else (end_date if end_date else default_end_date)
        
        self.logger.info(f"回测时间: {actual_start_date} 至 {actual_end_date}")
        self.logger.info(f"共 {len(all_signals)} 个信号")
        
        
        # 开始回测
        self.strategy.on_start()
        et_tz = pytz.timezone('America/New_York')
        
        # Get NYSE trading calendar
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.valid_days(
            start_date=actual_start_date.strftime('%Y-%m-%d'),
            end_date=actual_end_date.strftime('%Y-%m-%d')
        )
        trading_days_set = set(pd.to_datetime(trading_days).date)
        
        self.logger.info(f"NYSE 交易日: {len(trading_days_set)} 天")
        
        # Generate all 20-second intervals (9:30-16:00 each trading day only)
        start_date = actual_start_date
        end_date = actual_end_date
        
        all_intervals = []
        current_date = start_date
        while current_date <= end_date:
            # Only generate intervals for NYSE trading days
            if current_date in trading_days_set:
                day_start = et_tz.localize(datetime.combine(current_date, datetime.min.time()).replace(hour=9, minute=30))
                day_end = et_tz.localize(datetime.combine(current_date, datetime.min.time()).replace(hour=16, minute=0))
                
                current_time = day_start
                while current_time <= day_end:
                    all_intervals.append(current_time)
                    current_time += timedelta(seconds=20)
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"Generated {len(all_intervals)} intervals (20-second)")
         
        # Track prefetched symbols to avoid duplicate API calls
        prefetched_symbols = set()
        
        # Loop through each 20-second interval
        current_date = None
        for idx, current_time in enumerate(all_intervals, 1):
            if idx % 500 == 0:
                self.logger.info(f"Progress: {idx}/{len(all_intervals)} ({idx/len(all_intervals):.1%})")
            
            # 检查换日
            minute_date = current_time.date()
            if current_date is None:
                current_date = minute_date
                self.strategy.on_day_open(current_date)
            elif minute_date != current_date:
                self.strategy.on_day_close(current_date)
                self.strategy.daily_trade_count = 0
                current_date = minute_date
                self.strategy.on_day_open(current_date)
            
            # 设置当前时间
            self.market_client.set_current_time(current_time)
            
            # 1. 每分钟检查所有持仓（检查止损、止盈、strike、expiry等）
            # 但如果当前没有价格数据，就跳过（比如股票停盘）
            # 这样逻辑更合理：没数据就不处理，等到下一个有数据的交易日
            exit_decisions = self.strategy.on_minute_check(
                self.market_client,
                entry_time_map={k: v.isoformat() for k, v in self.position_entry_times.items()}
            )
            
            if exit_decisions:
                for exit_dec in exit_decisions:
                    self._execute_sell(exit_dec, current_time)
            
            # 2. 收集当前时间需要预加载的新符号（动态加载）
            if current_time in signals_by_time:
                for sig_data in signals_by_time[current_time]:
                    symbol = sig_data['signal'].symbol
                    # 检查是否需要预加载：不在prefetch_ranges中，或虽然在prefetched_symbols中但缓存已被清空
                    if symbol not in self.market_client.prefetch_ranges:
                        self.logger.info(f"📦 预加载 {symbol} 的6天数据...")
                        self.market_client.prefetch_multiple_days(symbol, current_time.date(), days=6)
                        prefetched_symbols.add(symbol)
            
            # 3. 处理所有信号
            if current_time in signals_by_time:
                for sig_data in signals_by_time[current_time]:
                    signal = sig_data['signal']
                    strike = sig_data['strike']
                    expiry = sig_data['expiry']
                    symbol = signal.symbol
                    
                    # 调试：打印前5个信号的时间
                    if not hasattr(self, '_signal_debug_count'):
                        self._signal_debug_count = 0
                    if self._signal_debug_count < 5:
                        self.logger.info(f"[DEBUG] 处理{symbol}信号: current_time={current_time.strftime('%Y-%m-%d %H:%M:%S')}, signal.event_time_et={signal.event_time_et.strftime('%Y-%m-%d %H:%M:%S')}")
                        self._signal_debug_count += 1
                    
                    # 策略判断（V8无复杂过滤）
                    decision = self.strategy.on_signal(signal, self.market_client)
                    
                    if decision:
                        self.signal_records.append({
                            'symbol': signal.symbol,
                            'time': signal.event_time_et,
                            'decision': 'BUY',
                            'shares': decision.shares,
                            'position_ratio': decision.pos_ratio,
                            'strike': strike,
                            'expiry': expiry.isoformat()
                        })
                        
                        # 执行买入
                        self._execute_buy(decision, signal, strike, expiry)
                        
                        # 记录进入时间
                        self.position_entry_times[signal.symbol] = signal.event_time_et
                        
                        # 在策略中存储strike、expiry和期权价格信息
                        option_price = signal.spot if signal.spot else 0.0
                        self.strategy.store_position_metadata(signal.symbol, strike, expiry, option_price, signal.event_time_et)
                        
                        # 更新策略状态
                        self.strategy.daily_trade_count += 1
                    else:
                        self.signal_records.append({
                            'symbol': signal.symbol,
                            'time': signal.event_time_et,
                            'decision': 'FILTERED',
                            'strike': strike,
                            'expiry': expiry.isoformat()
                        })
        
        # 回测结束
        self._close_all_positions()
        self.strategy.on_shutdown()
        
        self.logger.info("回测完成！")
    
    def _execute_buy(self, decision, signal, strike: float, expiry: date):
        """执行买入"""
        order = self.market_client.buy_stock(
            symbol=decision.symbol,
            quantity=decision.shares,
            price=decision.price_limit,
            order_type='LIMIT'
        )
        
        if order:
            self.trade_records.append({
                'type': 'BUY',
                'symbol': decision.symbol,
                'time': signal.event_time_et,
                'shares': order['quantity'],
                'price': order['price'],
                'amount': order['cost'],
                'position_ratio': decision.pos_ratio,
                'strike': strike,
                'expiry': expiry.isoformat()
            })
    
    def _execute_sell(self, decision, current_time):
        """执行卖出"""
        order = self.market_client.sell_stock(
            symbol=decision.symbol,
            quantity=decision.shares,
            price=decision.price_limit,
            order_type='LIMIT'
        )
        
        if order:
            self.trade_records.append({
                'type': 'SELL',
                'symbol': decision.symbol,
                'time': current_time,
                'shares': order['quantity'],
                'price': order['price'],
                'amount': order['proceeds'],
                'reason': decision.reason
            })
            
            # 移除持仓记录
            if decision.symbol in self.position_entry_times:
                del self.position_entry_times[decision.symbol]
            
            # 清除该股票的缓存数据以节省内存（关键！）
            self._clear_symbol_cache(decision.symbol)
    
    def _clear_symbol_cache(self, symbol: str):
        """
        清除某个股票的所有缓存数据（平仓时调用）
        
        这样内存占用会很低，因为只需要缓存当前持仓的股票数据
        """
        # 找出该符号的所有缓存 key
        keys_to_delete = [
            key for key in self.market_client.price_cache.keys()
            if key.startswith(f"{symbol}_")
        ]
        
        # 删除缓存
        for key in keys_to_delete:
            del self.market_client.price_cache[key]
        
        # 同时删除prefetch_ranges记录，这样下次遇到该符号时会重新预加载
        if symbol in self.market_client.prefetch_ranges:
            del self.market_client.prefetch_ranges[symbol]
        
        if keys_to_delete:
            self.logger.info(f"🗑️  清空 {symbol} 的缓存（{len(keys_to_delete)} 天数据，节省内存）")
    
    def _close_all_positions(self):
        """回测结束时平掉所有持仓"""
        positions = self.market_client.get_positions()
        if not positions:
            return
        
        self.logger.info(f"回测结束，平掉 {len(positions)} 个剩余持仓...")
        
        for pos in positions:
            order = self.market_client.sell_stock(
                symbol=pos['symbol'],
                quantity=pos['position'],
                price=pos['market_price'],
                order_type='LIMIT'
            )
            
            if order:
                self.trade_records.append({
                    'type': 'SELL',
                    'symbol': pos['symbol'],
                    'time': datetime.now(pytz.timezone('America/New_York')),
                    'shares': order['quantity'],
                    'price': order['price'],
                    'amount': order['proceeds'],
                    'reason': '回测结束'
                })
    
    def generate_report(self) -> Dict:
        """生成回测报告"""
        summary = self.market_client.get_summary()
        
        total_signals = len(self.signal_records)
        buy_signals = len([s for s in self.signal_records if s['decision'] == 'BUY'])
        filtered = total_signals - buy_signals
        
        return {
            '=== 账户概况 ===': {
                '初始资金': f"${summary['initial_cash']:,.2f}",
                '最终资金': f"${summary['cash']:,.2f}",
                '持仓市值': f"${summary['position_value']:,.2f}",
                '总资产': f"${summary['total_assets']:,.2f}",
                '总盈亏': f"${summary['total_pnl']:+,.2f}",
                '收益率': f"{summary['total_pnl_ratio']:+.2%}"
            },
            '=== 交易统计 ===': {
                '总信号数': total_signals,
                '买入信号': buy_signals,
                '过滤信号': filtered,
                '过滤率': f"{filtered/total_signals:.1%}" if total_signals > 0 else 'N/A',
                '实际交易数': len([t for t in self.trade_records if t['type'] == 'BUY']),
                '持仓数': summary['num_positions']
            },
            '=== 盈亏分析 ===': {
                '已实现盈亏': f"${summary['realized_pnl']:+,.2f}",
                '未实现盈亏': f"${summary['unrealized_pnl']:+,.2f}"
            }
        }
    
    def print_report(self):
        """打印报告"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("V8策略回测报告")
        print("="*60)
        
        for section, data in report.items():
            print(f"\n{section}")
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*60)
    
    def save_report(self, filename: str):
        """保存报告到JSON"""
        report = self.generate_report()
        
        # Add profit info to SELL trades
        self._add_profit_to_trades()
        
        output = {
            'backtest_time': datetime.now().isoformat(),
            'config_file': 'config_v8.yaml',
            'csv_file': str(self.csv_file),
            'initial_cash': self.initial_cash,
            'report': report,
            'trades': self.trade_records
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"报告已保存: {filename}")
    
    def _add_profit_to_trades(self):
        """为SELL交易添加收益信息"""
        symbol_buys = {}  # {symbol: [buy_records]}
        
        # First pass: collect all BUY records
        for trade in self.trade_records:
            if trade['type'] == 'BUY':
                symbol = trade['symbol']
                if symbol not in symbol_buys:
                    symbol_buys[symbol] = []
                symbol_buys[symbol].append(trade)
        
        # Second pass: add profit to SELL records
        for trade in self.trade_records:
            if trade['type'] == 'SELL' and 'profit' not in trade:
                symbol = trade['symbol']
                if symbol in symbol_buys and symbol_buys[symbol]:
                    buy = symbol_buys[symbol].pop(0)
                    buy_amount = buy['amount']
                    sell_amount = trade['amount']
                    profit = sell_amount - buy_amount
                    profit_rate = (profit / buy_amount * 100) if buy_amount > 0 else 0
                    
                    trade['profit'] = round(profit, 2)
                    trade['profit_rate'] = round(profit_rate, 2)
                    trade['buy_price'] = round(buy['price'], 4)
                    trade['buy_time'] = buy['time']


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V8策略回测（固定仓位版本）')
    parser.add_argument('--config', '-c', default='config_v8.yaml', help='配置文件')
    parser.add_argument('--csv', default='future_v_0_1/database/merged_strategy_v1_calls_bell_2023M3_2025M10.csv', help='CSV数据文件')
    parser.add_argument('--stock-dir', '-s', default='future_v_0_1/database/stock_data_csv_min', help='股价目录（backtest_client用）')
    parser.add_argument('--cash', type=float, default=1000000.0, help='初始资金')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期（YYYY-MM-DD）')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期（YYYY-MM-DD）')
    parser.add_argument('--output', '-o', default='backtest_v8_final.json', help='输出文件')
    parser.add_argument('--log-file', default='backtest_v8_log.txt', help='日志文件')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # 加载配置
    logger.info(f"加载配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 添加回测成本配置
    if 'backtest' not in config:
        config['backtest'] = {}
    config['backtest']['cost'] = {
        'slippage': 0.001,
        'commission_per_share': 0.005,
        'min_commission': 1.0
    }
    
    print("\n" + "="*60)
    print("V8策略回测 - 固定仓位版本（直接买入）")
    print("="*60)
    print(f"配置: {args.config}")
    print(f"CSV文件: {args.csv}")
    print(f"初始资金: ${args.cash:,.2f}")
    print("="*60 + "\n")
    
    # 创建运行器
    runner = BacktestRunnerV8(
        csv_file=args.csv,
        stock_data_dir=args.stock_dir,
        config=config,
        initial_cash=args.cash
    )
    
    # 解析日期参数
    start_date_obj = None
    end_date_obj = None
    
    if args.start_date:
        try:
            start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"开始日期 '{args.start_date}' 格式不正确，使用默认值")
    
    if args.end_date:
        try:
            end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"结束日期 '{args.end_date}' 格式不正确，使用默认值")
    
    # 运行回测
    start_time = datetime.now()
    logger.info(f"回测开始: {start_time}")
    
    runner.run(start_date=start_date_obj, end_date=end_date_obj)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"回测结束: {end_time}, 用时{duration:.1f}秒")
    
    # 打印报告
    runner.print_report()
    
    # 保存报告
    runner.save_report(args.output)
    
    print(f"\n✓ 回测完成！报告已保存到: {args.output}")


if __name__ == '__main__':
    main()
