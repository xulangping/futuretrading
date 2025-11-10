#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权卖出回测脚本 - 支持完整配置文件
Option Selling Backtest Runner with Full Configuration Support
"""

import sys
import csv
import json
import yaml
import logging
import pandas as pd
import pytz
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'future_v_0_1'))

from market.option_backtest_client import OptionBacktestClient


class OptionBacktestRunner:
    """期权卖出回测运行器 - 支持完整配置"""
    
    def __init__(self, config: Dict):
        """
        初始化期权回测运行器
        
        Args:
            config: 配置字典（从 YAML 加载）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # 回测基础设置
        backtest_cfg = config['backtest']
        self.initial_cash = backtest_cfg['initial_cash']
        self.hold_days = backtest_cfg['hold_days']
        self.max_positions = backtest_cfg['max_positions']
        self.signal_delay_minutes = backtest_cfg['signal_delay_minutes']
        
        # 交易成本
        costs_cfg = config['costs']
        
        # 数据配置
        data_cfg = config['data']
        self.csv_file = Path(data_cfg['csv_file'])
        self.timezone = pytz.timezone(data_cfg['timezone'])
        
        # 创建期权回测客户端
        self.option_client = OptionBacktestClient(
            initial_cash=self.initial_cash,
            slippage=costs_cfg['slippage'],
            commission_per_contract=costs_cfg['commission_per_contract'],
            min_commission=costs_cfg['min_commission'],
            cache_dir=data_cfg['cache_dir']
        )
        
        # 记录
        self.trade_records = []
        self.signal_records = []
        self.daily_stats = []
        
        # 持仓管理
        self.position_entry_times = {}  # {option_ticker: entry_time}
        self.position_target_close_times = {}  # {option_ticker: target_close_time}
        self.position_entry_premium = {}  # {option_ticker: entry_premium}
        
        self.logger.info(
            f"期权回测运行器初始化完成: CSV={self.csv_file}, "
            f"资金=${self.initial_cash:,.2f}, 持仓{self.hold_days}天"
        )
    
    def load_signals_from_csv(self) -> List[Dict]:
        """
        从CSV加载期权信号并应用筛选条件
        
        Returns:
            信号列表（按时间排序）
        
        DTE计算说明：
        - DTE从CSV中读取但被丢弃（CSV中的DTE计算方式不正确）
        - 系统根据交易时间（signal_time_et）和到期日期（expiry）重新计算DTE
        - 计算公式：dte = (expiry_date - signal_date).days
        - 这与股票回测中的DTE计算方式一致
        """
        self.logger.info(f"📂 从CSV加载期权信号: {self.csv_file}")
        signals = []
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, 1):
                    try:
                        signal = self._parse_and_filter_signal(row)
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        self.logger.debug(f"行{idx}解析失败: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"加载CSV失败: {e}")
            return []
        
        # 按时间排序
        signals.sort(key=lambda x: x['time_et'])
        
        self.logger.info(f"✅ 共加载 {len(signals)} 个期权信号（已应用筛选条件）")
        
        if signals:
            first_signal_time = signals[0]['time_et']
            last_signal_time = signals[-1]['time_et']
            self.logger.info(
                f"📅 信号时间范围: {first_signal_time.date()} 至 {last_signal_time.date()}"
            )
        
        return signals
    
    def _parse_and_filter_signal(self, row: Dict) -> Optional[Dict]:
        """
        解析CSV行并应用筛选条件
        
        Args:
            row: CSV行字典
        
        Returns:
            期权信号字典（符合筛选条件），或 None（不符合）
        """
        try:
            # 基础字段解析
            ticker = row['ticker']
            date_str = row['date']
            time_str = row['time']
            strike = float(row['strike'])
            option_type = row['option_type'].lower()
            expiry_str = row['expiry']
            side = row['side']
            
            # 解析premium (权利金总额)
            premium_str = row['premium'].strip()
            premium_usd = self._parse_usd_value(premium_str)
            
            # 解析spot (期权价格)
            spot_str = row['spot'].strip().replace('$', '').replace(',', '')
            spot = float(spot_str)
            
            # 解析price (股票价格)
            stock_price_str = row['price'].strip().replace('$', '').replace(',', '')
            stock_price = float(stock_price_str)
            
            # 解析otm_pct (价外百分比)
            otm_pct_str = row.get('otm_pct', '0%').strip().rstrip('%')
            otm_pct = float(otm_pct_str) if otm_pct_str else 0.0
            
            # 计算DTE（需要先转换时间）
            # 时间转换：中国时间 → ET时间
            datetime_str = f"{date_str} {time_str}"
            signal_time_cn = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            signal_time_cn = self.timezone.localize(signal_time_cn)
            et_tz = pytz.timezone('America/New_York')
            signal_time_et = signal_time_cn.astimezone(et_tz)
            
            # 添加信号延迟
            signal_time_et = signal_time_et + timedelta(minutes=self.signal_delay_minutes)
            
            # 如果延迟后超过16:00，使用15:59:00
            if signal_time_et.hour >= 16:
                signal_time_et = signal_time_et.replace(hour=15, minute=59, second=0)
            
            # 解析expiry日期并计算DTE
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            signal_date = signal_time_et.date()
            dte = (expiry_date - signal_date).days
            
            # 解析交易量
            volume_str = row.get('volume', '0').strip().replace(',', '')
            volume = int(volume_str) if volume_str and volume_str != '-' else 0
            
            # 解析持仓量
            oi_str = row.get('oi', '0').strip().replace(',', '')
            oi = int(oi_str) if oi_str and oi_str != '-' else 0
            
            # 解析IV
            iv_pct_str = row.get('iv_pct', '0').strip().rstrip('%')
            iv_pct = float(iv_pct_str) if iv_pct_str else 0.0
            
            # 解析Greeks
            delta = float(row.get('delta', 0)) if row.get('delta') else None
            theta = float(row.get('theta', 0)) if row.get('theta') else None
            gamma = float(row.get('gamma', 0)) if row.get('gamma') else None
            vega = float(row.get('vega', 0)) if row.get('vega') else None
            
            # 解析earnings
            earnings_str = row.get('earnings', '')
            earnings = int(earnings_str) if earnings_str and earnings_str.isdigit() else None
            
            # 解析sector和industry
            sector = row.get('sector', '')
            industry = row.get('industry', '')
            rule = row.get('rule', '')
            
            # 应用筛选条件
            if not self._passes_filters(
                option_type=option_type,
                side=side,
                ticker=ticker,
                premium_usd=premium_usd,
                spot=spot,
                stock_price=stock_price,
                strike=strike,
                otm_pct=otm_pct,
                dte=dte,
                volume=volume,
                oi=oi,
                iv_pct=iv_pct,
                delta=delta,
                theta=theta,
                gamma=gamma,
                vega=vega,
                earnings=earnings,
                sector=sector,
                rule=rule
            ):
                return None
            
            return {
                'ticker': ticker,
                'time_et': signal_time_et,
                'strike': strike,
                'option_type': option_type,
                'expiry': expiry_date,
                'premium_usd': premium_usd,
                'spot': spot,
                'stock_price': stock_price,
                'dte': dte,
                'otm_pct': otm_pct,
                'volume': volume,
                'oi': oi,
                'iv_pct': iv_pct,
                'delta': delta,
                'theta': theta,
                'gamma': gamma,
                'vega': vega,
                'earnings': earnings,
                'sector': sector,
                'industry': industry,
                'rule': rule,
            }
            
        except Exception as e:
            self.logger.debug(f"解析信号失败: {e}")
            return None
    
    def _passes_filters(self, **kwargs) -> bool:
        """
        检查信号是否通过所有筛选条件
        
        Returns:
            True = 通过筛选，False = 不通过
        """
        filters = self.config['filters']
        
        # Option type filter
        if kwargs['option_type'] not in filters['option_type']:
            return False
        
        # Side filter
        if kwargs['side'] not in filters['side']:
            return False
        
        # Premium filter
        if not (filters['premium']['min'] <= kwargs['premium_usd'] <= filters['premium']['max']):
            return False
        
        # Spot filter
        if not (filters['spot']['min'] <= kwargs['spot'] <= filters['spot']['max']):
            return False
        
        # Stock price filter
        if not (filters['stock_price']['min'] <= kwargs['stock_price'] <= filters['stock_price']['max']):
            return False
        
        # Strike filter
        if not (filters['strike']['min'] <= kwargs['strike'] <= filters['strike']['max']):
            return False
        
        # OTM percentage filter
        if not (filters['otm_pct']['min'] <= kwargs['otm_pct'] <= filters['otm_pct']['max']):
            return False
        
        # DTE filter
        if not (filters['dte']['min'] <= kwargs['dte'] <= filters['dte']['max']):
            return False
        
        # Volume filter
        min_vol = filters['volume']['min']
        max_vol = filters['volume']['max']
        if kwargs['volume'] < min_vol:
            return False
        if max_vol and kwargs['volume'] > max_vol:
            return False
        
        # Open interest filter
        min_oi = filters['open_interest']['min']
        max_oi = filters['open_interest']['max']
        if kwargs['oi'] < min_oi:
            return False
        if max_oi and kwargs['oi'] > max_oi:
            return False
        
        # IV filter
        if not (filters['iv_pct']['min'] <= kwargs['iv_pct'] <= filters['iv_pct']['max']):
            return False
        
        # Greeks filters
        if kwargs['delta'] is not None:
            if not (filters['delta']['min'] <= kwargs['delta'] <= filters['delta']['max']):
                return False
        
        if kwargs['theta'] is not None:
            if not (filters['theta']['min'] <= kwargs['theta'] <= filters['theta']['max']):
                return False
        
        if kwargs['gamma'] is not None:
            if not (filters['gamma']['min'] <= kwargs['gamma'] <= filters['gamma']['max']):
                return False
        
        if kwargs['vega'] is not None:
            if not (filters['vega']['min'] <= kwargs['vega'] <= filters['vega']['max']):
                return False
        
        # Earnings filter
        if kwargs['earnings'] is not None:
            min_days = filters['earnings']['min_days_to_earnings']
            max_days = filters['earnings']['max_days_to_earnings']
            if min_days and kwargs['earnings'] < min_days:
                return False
            if max_days and kwargs['earnings'] > max_days:
                return False
        
        # Sector filter
        sectors = filters.get('sectors')
        if sectors and kwargs['sector'] not in sectors:
            return False
        
        # Symbol whitelist/blacklist
        whitelist = filters['symbols'].get('whitelist')
        blacklist = filters['symbols'].get('blacklist')
        
        if whitelist and kwargs['ticker'] not in whitelist:
            return False
        if blacklist and kwargs['ticker'] in blacklist:
            return False
        
        # Rule filter
        rules = filters.get('rules')
        if rules and kwargs['rule'] not in rules:
            return False
        
        return True
    
    def _parse_usd_value(self, value_str: str) -> float:
        """解析美元值（支持K和M后缀）"""
        try:
            value_str = value_str.strip().replace('$', '').strip()
            
            if value_str.endswith('K'):
                return float(value_str[:-1]) * 1000
            elif value_str.endswith('M'):
                return float(value_str[:-1]) * 1000000
            else:
                return float(value_str)
        except Exception as e:
            self.logger.warning(f"解析USD值失败: {value_str}, 错误: {e}")
            return 0.0
    
    def run(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        """
        运行期权卖出回测
        
        Args:
            start_date: 回测开始日期（覆盖配置文件）
            end_date: 回测结束日期（覆盖配置文件）
        """
        # 加载信号
        all_signals = self.load_signals_from_csv()
        
        if len(all_signals) == 0:
            self.logger.warning("⚠️  无符合筛选条件的信号数据")
            return
        
        # 确定回测日期范围
        config_start = self.config['backtest'].get('start_date')
        config_end = self.config['backtest'].get('end_date')
        
        default_start_date = all_signals[0]['time_et'].date()
        default_end_date = all_signals[-1]['time_et'].date()
        
        actual_start_date = (
            start_date if start_date 
            else (datetime.strptime(config_start, '%Y-%m-%d').date() if config_start else default_start_date)
        )
        actual_end_date = (
            end_date if end_date 
            else (datetime.strptime(config_end, '%Y-%m-%d').date() if config_end else default_end_date)
        )
        
        self.logger.info(f"📅 回测时间: {actual_start_date} 至 {actual_end_date}")
        self.logger.info(f"📊 共 {len(all_signals)} 个符合条件的期权信号")
        
        # 过滤信号（只处理回测日期范围内的信号）
        signals_in_range = [
            sig for sig in all_signals
            if actual_start_date <= sig['time_et'].date() <= actual_end_date
        ]
        
        self.logger.info(f"✅ 回测范围内信号数: {len(signals_in_range)}")
        
        if len(signals_in_range) == 0:
            self.logger.warning("⚠️  回测范围内无信号")
            return
        
        # 按时间顺序处理信号
        all_events = []
        
        # 添加信号事件
        for sig in signals_in_range:
            all_events.append({
                'type': 'SIGNAL',
                'time': sig['time_et'],
                'data': sig
            })
        
        # 排序
        all_events.sort(key=lambda x: x['time'])
        
        self.logger.info(f"🚀 开始回测，共 {len(all_events)} 个事件...")
        
        # 处理事件
        processed_count = 0
        for event in all_events:
            processed_count += 1
            
            if processed_count % 500 == 0:
                self.logger.info(
                    f"⏳ Progress: {processed_count}/{len(all_events)} "
                    f"({processed_count/len(all_events):.1%})"
                )
            
            event_time = event['time']
            self.option_client.set_current_time(event_time)
            
            if event['type'] == 'SIGNAL':
                # 处理信号：卖出期权
                self._handle_sell_signal(event['data'])
            
            # 每次事件后，检查是否需要平仓
            self._check_positions_for_close(event_time)
        
        # 回测结束，平掉所有剩余持仓
        self._close_all_positions()
        
        self.logger.info("✅ 回测完成！")
    
    def _handle_sell_signal(self, signal: Dict):
        """处理期权卖出信号"""
        ticker = signal['ticker']
        strike = signal['strike']
        option_type = signal['option_type']
        expiry = signal['expiry']
        premium = signal['spot']
        time_et = signal['time_et']
        
        # 检查最大持仓数限制
        current_positions = len(self.option_client.get_option_positions())
        if current_positions >= self.max_positions:
            self.signal_records.append({
                'ticker': ticker,
                'time': time_et,
                'decision': 'SKIP_MAX_POSITIONS',
                'reason': f'已达最大持仓数 {self.max_positions}'
            })
            return
        
        # 构建期权代码
        option_ticker = self.option_client.construct_option_ticker(
            ticker, expiry, option_type, strike
        )
        
        # 计算合约数（根据仓位管理配置）
        contracts = self._calculate_position_size(signal)
        
        if contracts == 0:
            self.signal_records.append({
                'ticker': ticker,
                'time': time_et,
                'decision': 'SKIP_POSITION_SIZE',
            })
            return
        
        # 卖出期权（开仓）
        order = self.option_client.sell_option(
            underlying=ticker,
            expiry=expiry,
            option_type=option_type,
            strike=strike,
            contracts=contracts,
            premium=premium
        )
        
        if order:
            self.trade_records.append({
                'type': 'SELL_TO_OPEN',
                'option_ticker': option_ticker,
                'underlying': ticker,
                'time': time_et.isoformat(),
                'contracts': contracts,
                'premium': order['premium'],
                'credit_received': order['credit_received'],
                'strike': strike,
                'expiry': expiry.isoformat(),
                'option_type': option_type,
                'dte': signal['dte'],
                'iv_pct': signal['iv_pct'],
                'delta': signal['delta'],
            })
            
            # 记录持仓信息
            self.position_entry_times[option_ticker] = time_et
            self.position_entry_premium[option_ticker] = order['premium']
            
            # 计算目标平仓时间
            target_close_time = time_et + timedelta(days=self.hold_days)
            
            # 确保不晚于到期日
            expiry_time = datetime.combine(expiry, datetime.min.time()).replace(hour=15, minute=0)
            expiry_time = pytz.timezone('America/New_York').localize(expiry_time)
            
            # 到期前N天平仓
            exit_cfg = self.config['exit_rules']['expiry_based']
            if exit_cfg['enabled']:
                days_before = exit_cfg['days_before_expiry']
                expiry_time = expiry_time - timedelta(days=days_before)
            
            if target_close_time > expiry_time:
                target_close_time = expiry_time
            
            self.position_target_close_times[option_ticker] = target_close_time
            
            self.signal_records.append({
                'ticker': ticker,
                'time': time_et,
                'decision': 'SELL',
                'option_ticker': option_ticker,
                'contracts': contracts,
            })
        else:
            self.signal_records.append({
                'ticker': ticker,
                'time': time_et,
                'decision': 'SKIP_FAILED',
            })
    
    def _calculate_position_size(self, signal: Dict) -> int:
        """
        根据仓位管理配置计算合约数
        
        Returns:
            合约数量
        """
        sizing_cfg = self.config['position_sizing']
        method = sizing_cfg['method']
        
        if method == 'fixed':
            return sizing_cfg['fixed']['contracts_per_trade']
        
        elif method == 'risk_based':
            risk_cfg = sizing_cfg['risk_based']
            account_value = self.option_client.get_account_info()['total_assets']
            risk_amount = min(
                account_value * risk_cfg['risk_per_trade_pct'] / 100,
                risk_cfg['max_risk_per_trade']
            )
            # 简化：假设最大风险 = 权利金 * 2（期权价格翻倍）
            premium = signal['spot']
            if premium > 0:
                contracts = int(risk_amount / (premium * 100 * 2))
                return max(1, contracts)
            return 1
        
        elif method == 'kelly':
            kelly_cfg = sizing_cfg['kelly']
            win_rate = kelly_cfg['win_rate']
            avg_win = kelly_cfg['avg_win']
            avg_loss = kelly_cfg['avg_loss']
            kelly_fraction = kelly_cfg['kelly_fraction']
            
            # Kelly formula: f = (p*b - q) / b
            # where p = win rate, q = 1-p, b = avg_win/avg_loss
            b = avg_win / avg_loss if avg_loss > 0 else 1
            kelly_pct = (win_rate * b - (1 - win_rate)) / b
            kelly_pct = max(0, min(kelly_pct * kelly_fraction, 0.25))  # Cap at 25%
            
            account_value = self.option_client.get_account_info()['total_assets']
            position_value = account_value * kelly_pct
            premium = signal['spot']
            if premium > 0:
                contracts = int(position_value / (premium * 100))
                return max(1, contracts)
            return 1
        
        return 1
    
    def _check_positions_for_close(self, current_time: datetime):
        """检查是否有持仓需要平仓"""
        positions_to_close = []
        exit_cfg = self.config['exit_rules']
        
        for option_ticker in list(self.position_target_close_times.keys()):
            # 时间止损
            if exit_cfg['time_based']['enabled']:
                target_close_time = self.position_target_close_times[option_ticker]
                if current_time >= target_close_time:
                    positions_to_close.append((option_ticker, '到达目标持仓时间'))
                    continue
            
            # 获取当前期权价格
            current_premium = self.option_client.get_option_price_at_time(option_ticker, current_time)
            
            if current_premium is None:
                continue
            
            entry_premium = self.position_entry_premium.get(option_ticker)
            if entry_premium is None:
                continue
            
            # 计算收益率
            pnl_pct = (entry_premium - current_premium) / entry_premium * 100
            
            # 收益止盈
            if exit_cfg['profit_target']['enabled']:
                target_pct = exit_cfg['profit_target']['target_pct']
                if pnl_pct >= target_pct:
                    positions_to_close.append((option_ticker, f'止盈 ({pnl_pct:.1f}%)'))
                    continue
            
            # 止损
            if exit_cfg['stop_loss']['enabled']:
                loss_pct = exit_cfg['stop_loss']['loss_pct']
                if pnl_pct <= -loss_pct:
                    positions_to_close.append((option_ticker, f'止损 ({pnl_pct:.1f}%)'))
                    continue
        
        # 执行平仓
        for option_ticker, reason in positions_to_close:
            self._close_position(option_ticker, current_time, reason)
    
    def _close_position(self, option_ticker: str, close_time: datetime, reason: str):
        """平仓期权持仓"""
        # 获取当前期权价格
        current_premium = self.option_client.get_option_price_at_time(option_ticker, close_time)
        
        if current_premium is None:
            self.logger.warning(f"⚠️  无法获取 {option_ticker} 在 {close_time} 的价格，使用$0.01平仓")
            current_premium = 0.01
        
        # 获取持仓信息
        positions = self.option_client.get_option_positions()
        position = next((p for p in positions if p['option_ticker'] == option_ticker), None)
        
        if not position:
            return
        
        contracts = abs(position['contracts'])
        
        # 买入平仓
        order = self.option_client.buy_to_close_option(
            option_ticker=option_ticker,
            contracts=contracts,
            premium=current_premium
        )
        
        if order:
            self.trade_records.append({
                'type': 'BUY_TO_CLOSE',
                'option_ticker': option_ticker,
                'underlying': position['underlying'],
                'time': close_time.isoformat(),
                'contracts': contracts,
                'premium': order['premium'],
                'debit_paid': order['debit_paid'],
                'pnl': order['pnl'],
                'pnl_ratio': order['pnl_ratio'],
                'reason': reason,
                'strike': position['strike'],
                'expiry': position['expiry'].isoformat(),
                'option_type': position['option_type'],
            })
            
            # 移除持仓记录
            if option_ticker in self.position_entry_times:
                del self.position_entry_times[option_ticker]
            if option_ticker in self.position_target_close_times:
                del self.position_target_close_times[option_ticker]
            if option_ticker in self.position_entry_premium:
                del self.position_entry_premium[option_ticker]
    
    def _close_all_positions(self):
        """回测结束时平掉所有剩余持仓"""
        positions = self.option_client.get_option_positions()
        if not positions:
            return
        
        self.logger.info(f"📦 回测结束，平掉 {len(positions)} 个剩余持仓...")
        
        for pos in positions:
            option_ticker = pos['option_ticker']
            current_time = self.option_client.current_time
            current_premium = pos.get('current_premium', 0.01)
            
            if current_premium is None:
                current_premium = 0.01
            
            contracts = abs(pos['contracts'])
            
            order = self.option_client.buy_to_close_option(
                option_ticker=option_ticker,
                contracts=contracts,
                premium=current_premium
            )
            
            if order:
                self.trade_records.append({
                    'type': 'BUY_TO_CLOSE',
                    'option_ticker': option_ticker,
                    'underlying': pos['underlying'],
                    'time': current_time.isoformat(),
                    'contracts': contracts,
                    'premium': order['premium'],
                    'debit_paid': order['debit_paid'],
                    'pnl': order['pnl'],
                    'pnl_ratio': order['pnl_ratio'],
                    'reason': '回测结束',
                    'strike': pos['strike'],
                    'expiry': pos['expiry'].isoformat(),
                    'option_type': pos['option_type'],
                })
    
    def generate_report(self) -> Dict:
        """生成回测报告"""
        summary = self.option_client.get_summary()
        
        total_signals = len(self.signal_records)
        sell_signals = len([s for s in self.signal_records if s['decision'] == 'SELL'])
        skip_signals = total_signals - sell_signals
        
        # 统计交易结果
        close_trades = [t for t in self.trade_records if t['type'] == 'BUY_TO_CLOSE']
        winning_trades = [t for t in close_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in close_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(close_trades) if close_trades else 0
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return {
            '=== 账户概况 ===': {
                '初始资金': f"${summary['initial_cash']:,.2f}",
                '最终现金': f"${summary['cash']:,.2f}",
                '未实现盈亏': f"${summary['unrealized_pnl']:+,.2f}",
                '总资产': f"${summary['total_assets']:,.2f}",
                '总盈亏': f"${summary['total_pnl']:+,.2f}",
                '收益率': f"{summary['total_pnl_ratio']:+.2%}"
            },
            '=== 交易统计 ===': {
                '总信号数': total_signals,
                '卖出期权数': sell_signals,
                '跳过信号数': skip_signals,
                '平仓交易数': len(close_trades),
                '当前持仓数': summary['num_positions'],
                '收取权利金总额': f"${summary['total_premium_received']:,.2f}"
            },
            '=== 盈亏分析 ===': {
                '已实现盈亏': f"${summary['realized_pnl']:+,.2f}",
                '未实现盈亏': f"${summary['unrealized_pnl']:+,.2f}",
                '胜率': f"{win_rate:.1%}",
                '盈利交易数': len(winning_trades),
                '亏损交易数': len(losing_trades),
                '平均盈利': f"${avg_win:+,.2f}",
                '平均亏损': f"${avg_loss:+,.2f}",
            }
        }
    
    def print_report(self):
        """打印报告"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("期权卖出回测报告")
        print("="*60)
        
        for section, data in report.items():
            print(f"\n{section}")
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*60)
    
    def save_report(self, filename: str):
        """保存报告到JSON"""
        report = self.generate_report()
        
        output = {
            'backtest_time': datetime.now().isoformat(),
            'config': self.config,
            'csv_file': str(self.csv_file),
            'initial_cash': self.initial_cash,
            'report': report,
            'trades': self.trade_records if self.config['output']['save_trades'] else [],
            'api_stats': {
                'api_calls': self.option_client.api_calls,
                'cache_hits': self.option_client.cache_hits
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📄 报告已保存: {filename}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='期权卖出回测 - 支持完整配置文件')
    parser.add_argument('--config', '-c', default='config_option.yaml', help='配置文件路径')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期（覆盖配置文件）')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期（覆盖配置文件）')
    
    args = parser.parse_args()
    
    # 加载配置
    print("\n" + "="*60)
    print("期权卖出回测系统")
    print("="*60)
    print(f"📂 配置文件: {args.config}\n")
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        return
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return
    
    # 配置日志
    log_cfg = config['logging']
    logging.basicConfig(
        level=getattr(logging, log_cfg['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_cfg['log_file']),
            logging.StreamHandler() if log_cfg['console_output'] else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # 创建运行器
    runner = OptionBacktestRunner(config)
    
    # 解析日期参数
    start_date_obj = None
    end_date_obj = None
    
    if args.start_date:
        try:
            start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"开始日期 '{args.start_date}' 格式不正确")
    
    if args.end_date:
        try:
            end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"结束日期 '{args.end_date}' 格式不正确")
    
    # 运行回测
    start_time = datetime.now()
    logger.info(f"🚀 回测开始: {start_time}")
    
    runner.run(start_date=start_date_obj, end_date=end_date_obj)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"✅ 回测结束: {end_time}, 用时{duration:.1f}秒")
    
    # 打印报告
    runner.print_report()
    
    # 保存报告
    output_file = config['output']['result_file']
    runner.save_report(output_file)
    
    print(f"\n✅ 回测完成！报告已保存到: {output_file}\n")


if __name__ == '__main__':
    main()
