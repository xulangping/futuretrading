"""
V8 事件驱动期权流动量策略 - 简化版本（直接买入 + 固定仓位）

核心特性：
- 数据源：CSV历史数据（merged_strategy_v1_calls_bell_2023M3_2025M9.csv）
- 入场：直接买入，无复杂过滤
- 仓位：固定20%
- 出场优先级：
  1. 达到strike价格 卖出
  2. 定时出场（如配置max_holding_days）
  3. 达到expiry日期 + 10:00 AM 卖出
  4. 追踪止损（从最高价下跌）
  5. 止损-10% 卖出
  6. 止盈+20% 卖出
"""

import logging
from datetime import date, datetime, time, timedelta
from re import S
from typing import Dict
from zoneinfo import ZoneInfo
from pathlib import Path

try:
    from .strategy import StrategyBase, StrategyContext, EntryDecision, ExitDecision
except ImportError:
    from strategy import StrategyBase, StrategyContext, EntryDecision, ExitDecision

from numpy.ma import minimum_fill_value
import pandas_market_calendars as mcal


class StrategyV8(StrategyBase):
    """V8 事件驱动期权流动量策略 - 简化版本"""
    
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        
        # 读取 strategy 配置
        strategy_cfg = self.cfg.get('strategy', {})
        
        # === 市场日历 ===
        self.market_calendar = mcal.get_calendar('NYSE')
        
        # === 入场配置 ===
        filter_cfg = strategy_cfg.get('filter', {})
        
        # 交易时间窗口（支持多个时间段）
        self.trade_time_ranges = filter_cfg.get('trade_time_ranges', [])
        # 向后兼容：如果没有配置ranges，使用trade_start_time
        if not self.trade_time_ranges and 'trade_start_time' in filter_cfg:
            self.trade_time_ranges = [[filter_cfg.get('trade_start_time', '09:30:00'), '16:00:00']]
        
        
        # DTE过滤
        self.dte_min = filter_cfg.get('dte_min', 0)
        self.dte_max = filter_cfg.get('dte_max', 999999)
        
        # OTM率过滤
        self.otm_max = filter_cfg.get('otm_max', 100.0)
        self.otm_min = filter_cfg.get('otm_min', 0.0)
        
        # Premium过滤
        self.premium_max = filter_cfg.get('premium_max', 999999999)
        self.premium_min = filter_cfg.get('premium_min', 0)
        
        # MACD过滤（技术指标，从Polygon API获取）
        self.macd_enabled = filter_cfg.get('macd_enabled', False)
        self.macd_threshold = filter_cfg.get('macd_threshold', 0)  # MACD < 0表示看跌
        self.macd_short_window = filter_cfg.get('macd_short_window', 12)
        self.macd_long_window = filter_cfg.get('macd_long_window', 26)
        self.macd_signal_window = filter_cfg.get('macd_signal_window', 9)
        self.macd_timespan = filter_cfg.get('macd_timespan', 'day')
        
        # Earnings过滤（距离earnings日期的天数）
        self.earnings_enabled = filter_cfg.get('earnings_enabled', False)
        self.earnings_max = filter_cfg.get('earnings_max', 5)  # 最小距离earnings天数，小于这个值就过滤
        
        # 价格趋势过滤（比较-1天 vs -21天的收盘价）
        self.price_trend_enabled = filter_cfg.get('price_trend_enabled', False)
        self.price_trend_lookback = filter_cfg.get('price_trend_lookback', 21)
        
        # === 仓位配置（固定20%）===
        position_cfg = strategy_cfg.get('position_compute', {})
        self.fixed_position_ratio = position_cfg.get('fixed_position_ratio', 0.20)  # 固定20%
        self.max_daily_position = position_cfg.get('max_daily_position', 0.99)  # 每日总仓位上限
        
        # === 出场配置 ===
        self.stop_loss = strategy_cfg.get('stop_loss', 0.10)  # 止损 -10%
        self.take_profit = strategy_cfg.get('take_profit', 0.20)  # 止盈 +20%
        self.exit_time = strategy_cfg.get('exit_time', '10:00:00')  # 定时退出时间（expiry日10:00）
        self.trailing_stop_loss = strategy_cfg.get('trailing_stop_loss', 0.05) # 追踪止损 -5%
        self.max_holding_days = strategy_cfg.get('max_holding_days', None)  # 最大持有天数，None表示不启用
        
        # === 运行时状态 ===
        self.daily_trade_count = 0
        self.position_metadata: Dict[str, Dict] = {}  # {symbol: {strike, expiry, ...}}
        self.highest_price_map: Dict[str, float] = {}  # 追踪最高价
        
        # 打印配置信息
        time_ranges_str = ', '.join([f"{r[0]}-{r[1]}" for r in self.trade_time_ranges]) if self.trade_time_ranges else '全天'
        premium_str = f"${self.premium_min:,.0f}-${self.premium_max:,.0f}" if self.premium_max < 999999999 else f">${self.premium_min:,.0f}"
        macd_str = f"MACD过滤启用 (threshold={self.macd_threshold})" if self.macd_enabled else "MACD过滤禁用"
        holding_days_str = f"{self.max_holding_days}天" if self.max_holding_days else "禁用"
        
        self.logger.info(
            f"StrategyV8 初始化完成:\n"
            f"  入场: 时间={time_ranges_str}, DTE={self.dte_min}-{self.dte_max}天, OTM={self.otm_min:.1f}-{self.otm_max:.1f}%, Premium={premium_str}\n"
            f"  MACD: {macd_str}\n"
            f"  仓位: 固定{self.fixed_position_ratio:.0%}, 日限<={self.max_daily_position:.0%}\n"
            f"  出场: strike/定时({holding_days_str})/expiry(10:00)/止盈{self.take_profit:+.0%}/止损{self.stop_loss:+.0%}"
        )

    def on_start(self):
        """策略启动"""
        self.logger.info("StrategyV8 启动")

    def on_shutdown(self):
        """策略关闭"""
        self.logger.info("StrategyV8 关闭")

    def on_day_open(self, trading_date_et: date):
        """交易日开盘"""
        self.logger.info(f"交易日开盘: {trading_date_et}")

    def on_day_close(self, trading_date_et: date):
        """交易日收盘"""
        self.logger.info(f"交易日收盘: {trading_date_et}")

    def _quick_filter(self, ev) -> bool:
        """
        快速过滤：只做不需要实时价格的检查
        这样可以避免为明显不合格的信号加载价格数据
        
        Args:
            ev: SignalEvent
            
        Returns:
            True: 通过快速过滤，需要进一步检查
            False: 被过滤，无需加载价格
        """
        # ===== 1. 时间窗口过滤 =====
        current_time = ev.event_time_et.time()
        
        # 检查是否在配置的时间窗口内
        if self.trade_time_ranges:
            in_time_range = False
            for time_range in self.trade_time_ranges:
                start_time = datetime.strptime(time_range[0], '%H:%M:%S').time()
                end_time = datetime.strptime(time_range[1], '%H:%M:%S').time()
                if start_time <= current_time <= end_time:
                    in_time_range = True
                    break
            
            if not in_time_range:
                self.logger.info(f"过滤[时间窗口]: {ev.symbol} 不在交易时段 {current_time}")
                return False
        
        # 检查距离收盘时间
        market_close = time(15, 59, 0)
        if current_time >= market_close:
            self.logger.info(f"过滤[接近收盘]: {ev.symbol} 时间={current_time}")
            return False
        
        # ===== 2. DTE过滤 =====
        if hasattr(ev, 'expiry') and ev.expiry:
            current_date = ev.event_time_et.date()
            expiry_date = ev.expiry
            dte = (expiry_date - current_date).days
            
            if dte < self.dte_min:
                self.logger.info(f"过滤[DTE过短]: {ev.symbol} DTE={dte}天 < {self.dte_min}天")
                return False
            
            if dte > self.dte_max:
                self.logger.info(f"过滤[DTE过长]: {ev.symbol} DTE={dte}天 > {self.dte_max}天")
                return False
        
        # ===== 3. OTM过滤（使用signal中的stock_price）=====
        if hasattr(ev, 'strike') and hasattr(ev, 'stock_price') and ev.stock_price > 0:
            strike_price = ev.strike
            stock_price = ev.stock_price
            otm_pct = ((strike_price - stock_price) / stock_price) * 100
            
            if otm_pct > self.otm_max:
                self.logger.info(
                    f"过滤[OTM过高]: {ev.symbol} OTM={otm_pct:.2f}% > {self.otm_max}% "
                    f"(Strike=${strike_price:.2f}, SignalPrice=${stock_price:.2f})"
                )
                return False
            
            if otm_pct < self.otm_min:
                self.logger.info(
                    f"过滤[OTM过低]: {ev.symbol} OTM={otm_pct:.2f}% < {self.otm_min}%"
                )
                return False
        
        # ===== 4. Premium过滤 =====
        if hasattr(ev, 'premium_usd') and ev.premium_usd > 0:
            if ev.premium_usd > self.premium_max:
                self.logger.info(
                    f"过滤[Premium过大]: {ev.symbol} Premium=${ev.premium_usd:,.0f} > ${self.premium_max:,.0f}"
                )
                return False
            
            if ev.premium_usd < self.premium_min:
                self.logger.info(
                    f"过滤[Premium过小]: {ev.symbol} Premium=${ev.premium_usd:,.0f} < ${self.premium_min:,.0f}"
                )
                return False
        
        # ===== 5. Earnings过滤 =====
        if self.earnings_enabled and hasattr(ev, 'earnings') and ev.earnings is not None:
            if ev.earnings > self.earnings_max:
                self.logger.info(
                    f"过滤[Earnings]: {ev.symbol} 距离Earnings={ev.earnings}天 > {self.earnings_max}天"
                )
                return False
        
        # 通过所有快速过滤
        return True
    
    def _get_trading_day_before(self, target_date: date, num_trading_days: int) -> date:
        """
        计算target_date之前第num_trading_days个交易日
        
        Args:
            target_date: 参考日期
            num_trading_days: 往前回溯的交易日数（例如21表示21个交易日前）
        
        Returns:
            date: 往前第num_trading_days个交易日的日期
        
        Raises:
            ValueError: 如果无法找到足够多的交易日
        """
        # 获取从target_date向前200天的所有交易日（足够了）
        search_start = target_date - timedelta(days=300)  # 向前查找300天，足以找到21个交易日
        
        valid_days = self.market_calendar.valid_days(start_date=search_start, end_date=target_date)
        
        # valid_days是升序排列的，我们需要找倒数第num_trading_days个
        if len(valid_days) < num_trading_days + 1:  # +1因为我们需要至少比target_date早num_trading_days个交易日
            raise ValueError(
                f"无法找到足够的交易日。要求从 {target_date} 回溯 {num_trading_days} 个交易日，"
                f"但在过去300天内只找到 {len(valid_days)} 个交易日。"
            )
        
        # 找到target_date对应的索引
        target_idx = None
        for i, day in enumerate(valid_days):
            if day.date() == target_date:
                target_idx = i
                break
        
        if target_idx is None:
            raise ValueError(
                f"目标日期 {target_date} 不是交易日。"
            )
        
        # 返回从target_date向前num_trading_days个交易日
        lookback_idx = target_idx - num_trading_days
        if lookback_idx < 0:
            raise ValueError(
                f"无法回溯 {num_trading_days} 个交易日。从 {target_date} 往前只有 {target_idx} 个交易日。"
            )
        
        result_date = valid_days[lookback_idx].date()
        self.logger.debug(
            f"交易日计算: 从 {target_date} 回溯 {num_trading_days} 个交易日 → {result_date}"
        )
        return result_date

    def on_signal(self, ev, market_client=None):
        """
        处理信号事件，生成开仓决策
        
        V8优化版本：加入DTE、OTM、时间窗口过滤
        
        Args:
            ev: SignalEvent 信号事件
            market_client: 市场数据客户端实例
            
        Returns:
            EntryDecision 或 None
        """
        if not market_client:
            self.logger.error("市场数据客户端未提供，无法处理信号")
            return None
        
        # ===== 1. 时间窗口过滤 =====
        current_time = ev.event_time_et.time()
        
        # 检查是否在配置的时间窗口内
        if self.trade_time_ranges:
            in_time_range = False
            for time_range in self.trade_time_ranges:
                start_time = datetime.strptime(time_range[0], '%H:%M:%S').time()
                end_time = datetime.strptime(time_range[1], '%H:%M:%S').time()
                if start_time <= current_time <= end_time:
                    in_time_range = True
                    break
            
            if not in_time_range:
                self.logger.info(
                    f"过滤[时间窗口]: {ev.symbol} 不在交易时段 {current_time}"
                )
                return None
        
        # 检查距离收盘时间
        market_close = time(15, 59, 0)
        if current_time >= market_close:
            self.logger.info(
                f"过滤[接近收盘]: {ev.symbol} 时间={current_time}"
            )
            return None
        
        # ===== 2. DTE过滤 =====
        if hasattr(ev, 'expiry') and ev.expiry:
            expiry_date = ev.expiry if isinstance(ev.expiry, date) else datetime.strptime(str(ev.expiry), '%Y-%m-%d').date()
            current_date = ev.event_time_et.date()
            dte = (expiry_date - current_date).days
            
            if dte < self.dte_min:
                self.logger.info(
                    f"过滤[DTE过短]: {ev.symbol} DTE={dte}天 < {self.dte_min}天"
                )
                return None
            
            if dte > self.dte_max:
                self.logger.info(
                    f"过滤[DTE过长]: {ev.symbol} DTE={dte}天 > {self.dte_max}天"
                )
                return None
        else:
            dte = None
            self.logger.warning(f"{ev.symbol} 缺少expiry信息，无法计算DTE")
        
        # ===== 3. OTM率过滤（使用CSV中的stock_price，避免合股等问题）=====
        if hasattr(ev, 'strike') and ev.strike and hasattr(ev, 'stock_price') and ev.stock_price:
            strike = float(ev.strike)
            signal_stock_price = float(ev.stock_price)  # 使用CSV中的价格
            
            # 计算OTM率 (对于call期权)
            otm_rate = (strike - signal_stock_price) / signal_stock_price * 100
            
            if otm_rate > self.otm_max:
                self.logger.info(
                    f"过滤[OTM过高]: {ev.symbol} OTM={otm_rate:.2f}% > {self.otm_max:.1f}% "
                    f"(Strike=${strike:.2f}, SignalPrice=${signal_stock_price:.2f})"
                )
                return None
            
            if otm_rate < self.otm_min:
                self.logger.info(
                    f"过滤[OTM过低]: {ev.symbol} OTM={otm_rate:.2f}% < {self.otm_min:.1f}%"
                )
                return None
        else:
            otm_rate = None
            if not (hasattr(ev, 'strike') and ev.strike):
                self.logger.warning(f"{ev.symbol} 缺少strike信息，无法计算OTM率")
            elif not (hasattr(ev, 'stock_price') and ev.stock_price):
                self.logger.warning(f"{ev.symbol} 缺少stock_price信息，无法计算OTM率")
        
        # ===== 3.5. Premium过滤 =====
        if hasattr(ev, 'premium_usd') and ev.premium_usd:
            premium = ev.premium_usd
            
            if premium > self.premium_max:
                self.logger.info(
                    f"过滤[Premium过大]: {ev.symbol} Premium=${premium:,.0f} > ${self.premium_max:,.0f}"
                )
                return None
            
            if premium < self.premium_min:
                self.logger.info(
                    f"过滤[Premium过小]: {ev.symbol} Premium=${premium:,.0f} < ${self.premium_min:,.0f}"
                )
                return None
        
        # ===== 4. MACD过滤（技术指标，从Polygon API获取）=====
        if self.macd_enabled:
            macd_value = market_client.get_stock_macd(
                ev.symbol,
                timestamp=ev.event_time_et,
                short_window=self.macd_short_window,
                long_window=self.macd_long_window,
                signal_window=self.macd_signal_window,
                timespan=self.macd_timespan
            )
            
            if macd_value is None:
                self.logger.debug(
                    f"过滤[MACD查询失败]: {ev.symbol} 无法获取MACD数据，跳过MACD过滤"
                )
            else:
                if macd_value < self.macd_threshold:
                    self.logger.info(
                        f"过滤[MACD看跌]: {ev.symbol} MACD={macd_value:.4f} < {self.macd_threshold}"
                    )
                    return None
        
        # ===== 5. Earnings过滤（距离earnings日期天数）=====
        if self.earnings_enabled:
            earnings = ev.earnings
            
            if earnings is None:
                self.logger.debug(
                    f"过滤[Earnings缺失]: {ev.symbol} 缺少earnings信息，跳过earnings过滤"
                )
            else:
                if earnings < self.earnings_max:
                    self.logger.info(
                        f"过滤[Earnings太近]: {ev.symbol} earnings={earnings}天 < {self.earnings_max}天（太接近earnings日，不交易）"
                    )
                    return None
        
        # ===== 6. 价格趋势过滤（对比-1天 vs -21天收盘价）=====
        if self.price_trend_enabled:
            signal_date = ev.event_time_et.date()
            
            # 获取-1交易日收盘价
            try:
                price_minus_1_date = self._get_trading_day_before(signal_date, 1)
                price_minus_1 = market_client.get_day_close_price(ev.symbol, price_minus_1_date, max_lookback_days=5)
                
                if price_minus_1 is None:
                    self.logger.debug(
                        f"过滤[价格数据缺失]: {ev.symbol} 无法获取-1交易日收盘价，跳过价格趋势过滤"
                    )
                    price_minus_1 = None
            except (ValueError, KeyError) as e:
                self.logger.debug(f"交易日计算失败（-1天）: {e}，跳过价格趋势过滤")
                price_minus_1 = None
            
            # 获取-lookback交易日收盘价
            try:
                date_minus_lookback = self._get_trading_day_before(signal_date, self.price_trend_lookback)
                price_minus_lookback = market_client.get_day_close_price(ev.symbol, date_minus_lookback, max_lookback_days=5)
                
                if price_minus_lookback is None:
                    self.logger.debug(
                        f"过滤[价格数据缺失]: {ev.symbol} 无法获取-{self.price_trend_lookback}交易日收盘价，跳过价格趋势过滤"
                    )
                    price_minus_lookback = None
            except (ValueError, KeyError) as e:
                self.logger.debug(f"交易日计算失败(-{self.price_trend_lookback}天): {e}，跳过价格趋势过滤")
                price_minus_lookback = None
            
            # 如果两个价格都能获取，检查趋势
            if price_minus_1 is not None and price_minus_lookback is not None:
                if price_minus_1 < price_minus_lookback:
                    self.logger.info(
                        f"过滤[价格下跌]: {ev.symbol} -1交易日收盘价${price_minus_1:.2f} < -{self.price_trend_lookback}交易日${price_minus_lookback:.2f}（下跌趋势，不交易）"
                    )
                    return None
        
        # ===== 7. 获取股票价格（用于实际买入）=====
        # 注意：market_client.current_time已在主循环中设置，这里无需再设置
        
        price_info = market_client.get_stock_price(ev.symbol)
        if not price_info:
            self.logger.error(f"获取 {ev.symbol} 价格失败")
            return None
        
        current_price = price_info['last_price']
        
        # 调试：打印获取价格的时间（仅前10笔）
        if not hasattr(self, '_price_debug_count'):
            self._price_debug_count = 0
        if self._price_debug_count < 10:
            self.logger.info(f"[DEBUG] {ev.symbol} market_client.current_time={market_client.current_time.strftime('%Y-%m-%d %H:%M:%S')}, 价格=${current_price:.4f}")
            self._price_debug_count += 1
        
        # ===== 6. 获取账户信息 =====
        acc_info = market_client.get_account_info()
        if not acc_info:
            self.logger.error("获取账户信息失败")
            return None
        
        total_assets = acc_info['total_assets']
        cash = acc_info['cash']
        
        # ===== 7. 计算仓位（固定比例）=====
        target_value = total_assets * self.fixed_position_ratio
        qty = int(target_value / current_price)
        
        if qty <= 0:
            self.logger.debug(f"过滤: {ev.symbol} 计算股数为0")
            return None
        
        target_cost = current_price * qty
        
        self.logger.debug(
            f"仓位计算: 固定比例{self.fixed_position_ratio:.0%} → "
            f"{qty}股 × ${current_price:.2f} = ${target_cost:,.2f}"
        )
        
        # ===== 8. 检查总仓位限制 =====
        positions = market_client.get_positions()
        current_position_value = 0
        
        if positions:
            # 检查是否已持有该股票
            for pos in positions:
                if pos['symbol'] == ev.symbol and pos['position'] > 0:
                    self.logger.info(f"过滤: {ev.symbol} 已持有仓位，避免重复开仓")
                    return None
                
                current_position_value += pos.get('market_value', 0)
        
        # 检查总仓位限制
        new_total_position_value = current_position_value + target_cost
        new_total_position_ratio = new_total_position_value / total_assets
        
        if new_total_position_ratio > self.max_daily_position:
            self.logger.info(
                f"过滤: {ev.symbol} 总仓位将超限 {new_total_position_ratio:.1%} > "
                f"{self.max_daily_position:.0%}"
            )
            return None
        
        # ===== 9. 检查现金充足性 =====
        if cash < target_cost:
            self.logger.info(
                f"过滤: {ev.symbol} 现金不足 需要${target_cost:,.2f}, 剩余${cash:,.2f}"
            )
            return None
        
        # ===== 10. 生成开仓决策 =====
        # 注意：信号时间已在run_backtest_v8.py聚合阶段延迟10分钟
        # 这里的event_time_et已经是延迟后的时间，无需再计算
        signal_time = ev.event_time_et
        exec_time = signal_time
        
        # ===== 11. 生成开仓决策 =====
        client_id = f"{ev.symbol}_{ev.event_time_et.strftime('%Y%m%d%H%M%S')}"
        
        # 构建详细日志
        log_parts = [
            f"✓ 开仓决策: {ev.symbol} {qty}股 @${current_price:.2f}",
            f"仓位{self.fixed_position_ratio:.0%}"
        ]
        if dte is not None:
            log_parts.append(f"DTE={dte}天")
        if otm_rate is not None:
            log_parts.append(f"OTM={otm_rate:.2f}%")
        
        self.logger.info(", ".join(log_parts))
        
        # 准备元数据
        meta_data = {
            'event_id': ev.event_id,
            'signal_time': signal_time.isoformat(),
            'exec_time': exec_time.isoformat(),
        }
        if dte is not None:
            meta_data['dte'] = dte
        if otm_rate is not None:
            meta_data['otm_rate'] = otm_rate
        
        return EntryDecision(
            symbol=ev.symbol,
            shares=qty,
            price_limit=current_price,
            t_exec_et=exec_time,  # 使用延迟后的执行时间
            pos_ratio=self.fixed_position_ratio,
            client_id=client_id,
            meta=meta_data
        )

    def on_minute_check(self, market_client=None, entry_time_map=None):
        """每分钟检查持仓"""
        return self.on_position_check(market_client, entry_time_map)
    
    def on_position_check(self, market_client=None, entry_time_map=None):
        """
        检查持仓，生成平仓决策
        
        出场优先级：
        1. 达到strike价格 卖出
        2. 定时出场（如配置max_holding_days，替代expiry出场）或 expiry日期 + 10:00 AM 卖出
        3. 追踪止损（从最高价下跌）
        4. 止损-10% 卖出
        5. 止盈+20% 卖出
        
        注意：max_holding_days和expiry是互斥的，配置了max_holding_days就不再使用expiry
        """
        if not market_client:
            self.logger.error("市场数据客户端未提供，无法检查持仓")
            return []
        
        positions = market_client.get_positions()
        if not positions:
            return []
        
        if entry_time_map is None:
            entry_time_map = {}
        
        exit_decisions = []
        
        # 获取当前时间
        if hasattr(market_client, 'current_time') and market_client.current_time:
            current_et = market_client.current_time
        else:
            current_et = datetime.now(ZoneInfo('America/New_York'))
        
        exit_time_today = datetime.strptime(self.exit_time, '%H:%M:%S').time()
        
        for pos in positions:
            symbol = pos['symbol']
            cost_price = pos['cost_price']
            current_price = pos['market_price']
            can_sell_qty = pos['can_sell_qty']
            
            # 跳过可卖数量为0的持仓，以及当前没有价格数据的持仓
            if can_sell_qty <= 0:
                continue
            
            # 如果没有价格数据（比如股票停盘），跳过该持仓的检查
            # 等待下一个有数据的交易日再处理
            if current_price is None or current_price <= 0:
                self.logger.debug(f"跳过 {symbol}：无价格数据（可能停盘）")
                continue
            
            # 计算盈亏比例
            pnl_ratio = (current_price - cost_price) / cost_price
            
            # 更新历史最高价
            if symbol not in self.highest_price_map:
                self.highest_price_map[symbol] = max(cost_price, current_price)
            else:
                self.highest_price_map[symbol] = max(self.highest_price_map[symbol], current_price)
            
            # ===== 1. 检查strike价格出场 =====
            # 目标价 = strike (不加option_price)
            if symbol in self.position_metadata:
                meta = self.position_metadata[symbol]
                if 'strike' in meta:
                    strike_price = meta['strike']
                    out_price = strike_price
                    if current_price >= out_price:
                        self.logger.info(
                            f"✓ 平仓决策[达到strike]: {symbol} {can_sell_qty}股 @${current_price:.2f} "
                            f"(成本${cost_price:.2f}, Strike=${strike_price:.2f}, 盈亏{pnl_ratio:+.1%})"
                        )
                        exit_decisions.append(ExitDecision(
                            symbol=symbol,
                            shares=can_sell_qty,
                            price_limit=current_price,
                            reason='strike_price',
                            client_id=f"{symbol}_ST_{current_et.strftime('%Y%m%d%H%M%S')}",
                            meta={'pnl_ratio': pnl_ratio, 'strike': strike_price}
                        ))
                        # 清除元数据
                        if symbol in self.position_metadata:
                            del self.position_metadata[symbol]
                        if symbol in self.highest_price_map:
                            del self.highest_price_map[symbol]
                        continue
            
            # ===== 2. 检查定时/expiry日期出场 =====
            # 如果配置了max_holding_days，使用定时出场；否则使用expiry出场
            if symbol in self.position_metadata:
                meta = self.position_metadata[symbol]
                
                # 优先使用max_holding_days定时出场
                if self.max_holding_days and 'entry_time' in meta and meta['entry_time']:
                    entry_time = meta['entry_time']
                    entry_date = entry_time.date()
                    current_date = current_et.date()
                    
                    # 计算持有天数（自然日）
                    holding_days = (current_date - entry_date).days
                    
                    # 达到指定天数 + 10:00 AM 才卖出
                    if holding_days >= self.max_holding_days and current_et.time() >= exit_time_today:
                        self.logger.info(
                            f"✓ 平仓决策[定时出场]: {symbol} {can_sell_qty}股 @${current_price:.2f} "
                            f"(成本${cost_price:.2f}, 持有{holding_days}天≥{self.max_holding_days}天, 盈亏{pnl_ratio:+.1%})"
                        )
                        exit_decisions.append(ExitDecision(
                            symbol=symbol,
                            shares=can_sell_qty,
                            price_limit=current_price,
                            reason='max_holding_days',
                            client_id=f"{symbol}_HD_{current_et.strftime('%Y%m%d%H%M%S')}",
                            meta={'pnl_ratio': pnl_ratio, 'holding_days': holding_days, 'entry_date': entry_date.isoformat()}
                        ))
                        # 清除元数据
                        if symbol in self.position_metadata:
                            del self.position_metadata[symbol]
                        if symbol in self.highest_price_map:
                            del self.highest_price_map[symbol]
                        continue
                
                # 如果没有配置max_holding_days，使用expiry日期出场
                if 'expiry' in meta:
                    expiry_date = meta['expiry']
                    current_date = current_et.date()
                    
                    if current_date >= expiry_date and current_et.time() >= exit_time_today:
                        self.logger.info(
                            f"✓ 平仓决策[expiry到期]: {symbol} {can_sell_qty}股 @${current_price:.2f} "
                            f"(成本${cost_price:.2f}, 到期日{expiry_date}, 盈亏{pnl_ratio:+.1%})"
                        )
                        exit_decisions.append(ExitDecision(
                            symbol=symbol,
                            shares=can_sell_qty,
                            price_limit=current_price,
                            reason='expiry_date',
                            client_id=f"{symbol}_EX_{current_et.strftime('%Y%m%d%H%M%S')}",
                            meta={'pnl_ratio': pnl_ratio, 'expiry': expiry_date.isoformat()}
                        ))
                        # 清除元数据
                        if symbol in self.position_metadata:
                            del self.position_metadata[symbol]
                        if symbol in self.highest_price_map:
                            del self.highest_price_map[symbol]
                        continue
            
            # ===== 4. 检查动态止损（从最高价下跌） =====
            if self.trailing_stop_loss > 0 and symbol in self.highest_price_map:
                highest_price = self.highest_price_map[symbol]
                trailing_stop_price = highest_price * (1 - self.trailing_stop_loss)
                
                if current_price <= trailing_stop_price:
                    trailing_loss_ratio = (highest_price - current_price) / highest_price
                    self.logger.info(
                        f"✓ 平仓决策[动态止损]: {symbol} {can_sell_qty}股 @${current_price:.2f} "
                        f"(最高${highest_price:.2f}, 触发阈值${trailing_stop_price:.2f}, 从高点下跌{trailing_loss_ratio:.1%})"
                    )
                    exit_decisions.append(ExitDecision(
                        symbol=symbol,
                        shares=can_sell_qty,
                        price_limit=current_price,
                        reason='trailing_stop_loss',
                        client_id=f"{symbol}_TS_{current_et.strftime('%Y%m%d%H%M%S')}",
                        meta={'pnl_ratio': pnl_ratio, 'highest_price': highest_price, 'trailing_loss_ratio': trailing_loss_ratio}
                    ))
                    # 清除元数据和最高价
                    if symbol in self.position_metadata:
                        del self.position_metadata[symbol]
                    if symbol in self.highest_price_map:
                        del self.highest_price_map[symbol]
                    continue
            
            # ===== 5. 检查止损 =====
            stop_loss_price = cost_price * (1 - self.stop_loss)
            if current_price <= stop_loss_price:
                self.logger.info(
                    f"✓ 平仓决策[止损]: {symbol} {can_sell_qty}股 @${current_price:.2f} "
                    f"(成本${cost_price:.2f}, 目标价${strike_price:.2f}, 止损价${stop_loss_price:.2f}, 亏损{pnl_ratio:.1%})"
                )
                exit_decisions.append(ExitDecision(
                    symbol=symbol,
                    shares=can_sell_qty,
                    price_limit=current_price,
                    reason='stop_loss',
                    client_id=f"{symbol}_SL_{current_et.strftime('%Y%m%d%H%M%S')}",
                    meta={'pnl_ratio': pnl_ratio, 'stop_loss_price': stop_loss_price}
                ))
                # 清除元数据和最高价
                if symbol in self.position_metadata:
                    del self.position_metadata[symbol]
                if symbol in self.highest_price_map:
                    del self.highest_price_map[symbol]
                continue
            # ===== 6. 检查止盈 =====
            take_profit_price = cost_price * (1 + self.take_profit)
            if current_price >= take_profit_price:
                self.logger.info(
                    f"✓ 平仓决策[止盈]: {symbol} {can_sell_qty}股 @${current_price:.2f} "
                    f"(成本${cost_price:.2f}, 止盈价${take_profit_price:.2f}, 盈利{pnl_ratio:.1%})"
                )
                exit_decisions.append(ExitDecision(
                    symbol=symbol,
                    shares=can_sell_qty,
                    price_limit=current_price,
                    reason='take_profit',
                    client_id=f"{symbol}_TP_{current_et.strftime('%Y%m%d%H%M%S')}",
                    meta={'pnl_ratio': pnl_ratio, 'take_profit_price': take_profit_price}
                ))
                # 清除元数据和最高价
                if symbol in self.position_metadata:
                    del self.position_metadata[symbol]
                if symbol in self.highest_price_map:
                    del self.highest_price_map[symbol]
                continue
            
        
        return exit_decisions

    def store_position_metadata(self, symbol: str, strike: float, expiry: date, option_price: float = None, entry_time: datetime = None):
        """存储持仓的strike、expiry和期权价格信息（由回测器调用）"""
        self.position_metadata[symbol] = {
            'strike': strike,
            'expiry': expiry,
            'option_price': option_price,  # spot from signal (option ask price)
            'entry_time': entry_time  # 买入时间，用于定时出场
        }

    def on_order_filled(self, res):
        """订单成交回调"""
        self.logger.info(
            f"订单成交: {res.client_id}, 成交价: ${res.avg_price:.2f}, "
            f"成交量: {res.filled_shares}"
        )

    def on_order_rejected(self, res, reason: str):
        """订单拒绝回调"""
        self.logger.warning(
            f"订单拒绝: {res.client_id}, 原因: {reason}"
        )


if __name__ == '__main__':
    """测试脚本"""
    import yaml
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 读取配置文件
    config_path = Path(__file__).parent.parent.parent / 'config_v8.yaml'
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建策略上下文
    context = StrategyContext(
        cfg=config,
        logger=logging.getLogger('StrategyV8')
    )
    
    # 创建策略实例
    strategy = StrategyV8(context)
    
    print("\n✓ StrategyV8 测试成功")
    print(f"  固定仓位: {strategy.fixed_position_ratio:.0%}")
    print(f"  每日总仓位上限: {strategy.max_daily_position:.0%}")
    print(f"  止损/止盈: {strategy.stop_loss:.0%} / {strategy.take_profit:.0%}")
    print(f"  expiry出场时间: {strategy.exit_time}")
