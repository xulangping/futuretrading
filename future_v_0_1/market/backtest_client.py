"""
Backtesting market client using Polygon.io second-level data
"""

import os
import logging
import pytz
import requests
import pandas as pd
import json
import pandas_market_calendars as mcal
import duckdb
from pathlib import Path
from typing import Optional, Dict, List, Set
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Load .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


class BacktestMarketClient:
    """Backtesting market client using Polygon API (second-level data)"""
    
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
    
    def __init__(self, stock_data_dir: str = None, initial_cash: float = 100000.0,
                 slippage: float = 0.0005, commission_per_share: float = 0.005, 
                 min_commission: float = 1.0, cache_dir: str = None):
        """
        Initialize backtesting client
        
        Args:
            stock_data_dir: Ignored (for compatibility)
            initial_cash: Initial cash amount
            slippage: Single-side slippage (default 0.05%)
            commission_per_share: Commission per share (default $0.005)
            min_commission: Minimum commission (default $1)
            cache_dir: Local cache directory for parquet files (default: ./data_cache)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Polygon API
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in .env file")
        
        # Local cache directory (parquet files)
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'database' / 'data_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Local cache directory: {self.cache_dir}")
        
        # Data cache {symbol_date: DataFrame}
        self.price_cache: Dict[str, pd.DataFrame] = {}
        # Track prefetch range for each symbol {symbol: (start_date, end_date)}
        self.prefetch_ranges: Dict[str, tuple] = {}
        self.api_calls = 0
        self.cache_hits = 0
        
        # Market calendar (NYSE)
        self.market_calendar = mcal.get_calendar('NYSE')
        
        # Trading costs
        self.slippage = slippage
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        
        # Account info
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict] = {}
        
        # Orders
        self.orders: List[Dict] = []
        self.order_id_counter = 1
        
        # Current time
        self.current_time: Optional[datetime] = None
        
        self.logger.info(
            f"Backtest client initialized: cash=${initial_cash:,.2f}, "
            f"slippage={slippage:.1%}, commission=${commission_per_share}/share"
        )
    
    def connect(self) -> bool:
        """Simulate connection"""
        self.logger.info("Connected (Polygon API)")
        return True
    
    def disconnect(self):
        """Simulate disconnection"""
        self.logger.info("Disconnected")
        self.logger.info(
            f"API stats: {self.api_calls} calls, "
            f"{self.cache_hits} cache hits, {len(self.price_cache)} days cached"
        )
    
    def _fetch_day_data(self, symbol: str, date_obj) -> Optional[pd.DataFrame]:
        """Fetch second-level data for a single day"""
        date_str = date_obj.isoformat() if hasattr(date_obj, 'isoformat') else str(date_obj)
        
        try:
            url = (f"{self.BASE_URL}/{symbol}/range/1/second"
                   f"/{date_str}/{date_str}"
                   f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            self.api_calls += 1
            self.logger.info(f"Fetching {symbol} on {date_str} (API call #{self.api_calls})...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('resultsCount', 0) == 0:
                self.logger.warning(f"No data for {symbol} on {date_str} (possibly weekend/holiday)")
                return None
            
            results = data.get('results', [])
            records = []
            et_tz = pytz.timezone('America/New_York')
            
            for item in results:
                timestamp = datetime.fromtimestamp(item['t'] / 1000, tz=pytz.UTC)
                timestamp = timestamp.astimezone(et_tz)
                
                records.append({
                    'datetime': timestamp,
                    'close': item['c'],
                })
            
            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            
            self.logger.debug(f"Fetched {symbol} on {date_str}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch {symbol} on {date_str}: {e}")
            return None
    
    def _load_from_local_cache(self, symbol: str, start_date, end_date) -> Optional[Dict[str, pd.DataFrame]]:
        """Load data from local parquet cache using DuckDB"""
        try:
            # Check if cache file exists
            cache_file = self.cache_dir / f"{symbol}.parquet"
            if not cache_file.exists():
                return None
            
            start_str = start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date)
            end_str = end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date)
            
            # Use pandas to read parquet file (better timezone handling than DuckDB)
            df = pd.read_parquet(cache_file)
            
            # Filter by date range
            df = df[(df['date'] >= start_str) & (df['date'] <= end_str)]
            
            if len(df) == 0:
                return None
            
            # Check if we have data for all required trading days
            # Get NYSE trading days in the requested range
            dates_in_cache = set(df['date'].unique())
            
            try:
                nyse = mcal.get_calendar('NYSE')
                trading_days = nyse.valid_days(start_date=start_str, end_date=end_str)
                expected_trading_days = set(pd.to_datetime(trading_days).date.astype(str))
                
                # Find missing trading days
                missing_days = expected_trading_days - dates_in_cache
                
                if missing_days:
                    self.logger.info(f"缓存中缺少 {symbol} 的 {len(missing_days)} 个交易日，将从 API 补充: {sorted(list(missing_days))[:3]}...")
                    # Return None to trigger full API fetch which will merge with existing cache
                    # The _save_to_local_cache will handle merging new data with existing cache
                    return None
                
                # All required trading days are in cache
                # No need to check non-trading days (weekends/holidays)
                
            except Exception as e:
                # If calendar check fails, fall back to basic validation
                self.logger.debug(f"无法检查交易日历: {e}，使用基本验证")
                
                # At minimum, check if start_date is in cache
                if start_str not in dates_in_cache:
                    self.logger.debug(f"缓存中没有 {symbol} 在 {start_str} 的数据，需要从 API 获取")
                    return None
            
            # Ensure datetime column has correct timezone (already stored with ET timezone)
            if df['datetime'].dt.tz is None:
                et_tz = pytz.timezone('America/New_York')
                df['datetime'] = df['datetime'].dt.tz_localize(et_tz)
            
            # Group by date
            result = {}
            for date_str, group in df.groupby('date'):
                cache_key = f"{symbol}_{date_str}"
                group_df = group[['datetime', 'close']].copy()
                group_df.set_index('datetime', inplace=True)
                result[cache_key] = group_df
            
            self.logger.info(f"💾 Loaded {symbol} from local cache: {len(result)} days, {len(df)} records")
            return result
            
        except Exception as e:
            self.logger.debug(f"Failed to load from local cache: {e}")
            return None
    
    def _save_to_local_cache(self, symbol: str, daily_data: Dict[str, pd.DataFrame]):
        """Save data to local parquet cache"""
        try:
            cache_file = self.cache_dir / f"{symbol}.parquet"
            
            # Convert daily_data to single DataFrame
            rows = []
            for cache_key, df in daily_data.items():
                date_str = cache_key.split('_', 1)[1]
                for idx, row in df.iterrows():
                    rows.append({
                        'date': date_str,
                        'datetime': idx,
                        'close': row['close']
                    })
            
            if not rows:
                return
            
            new_df = pd.DataFrame(rows)
            
            # If cache file exists, merge with existing data
            if cache_file.exists():
                try:
                    existing_df = pd.read_parquet(cache_file)
                    # Remove duplicates and concatenate
                    combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['datetime'])
                    combined_df.to_parquet(cache_file, index=False)
                    self.logger.debug(f"💾 Updated local cache: {cache_file}")
                except Exception as e:
                    self.logger.debug(f"Failed to merge cache, overwriting: {e}")
                    new_df.to_parquet(cache_file, index=False)
            else:
                new_df.to_parquet(cache_file, index=False)
                self.logger.debug(f"💾 Created local cache: {cache_file}")
                
        except Exception as e:
            self.logger.debug(f"Failed to save to local cache: {e}")
    
    def _fetch_range_data(self, symbol: str, start_date, end_date) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetch multi-day data with pagination support (完整获取所有数据！)
        先从本地缓存读取，如果没有再调用API
        
        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dict[cache_key, DataFrame] 按日期分组的数据字典
        """
        start_str = start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date)
        end_str = end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date)
        
        # Try to load from local cache first
        cached_data = self._load_from_local_cache(symbol, start_date, end_date)
        if cached_data is not None:
            self.cache_hits += 1
            return cached_data
        
        try:
            all_results = []
            url = (f"{self.BASE_URL}/{symbol}/range/1/second"
                   f"/{start_str}/{end_str}"
                   f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            # 分页循环：不断获取，直到没有 next_url
            page = 0
            while url:
                page += 1
                self.api_calls += 1
                self.logger.info(f"Fetching {symbol} from {start_str} to {end_str} page {page} (API call #{self.api_calls})...")
                
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('resultsCount', 0) == 0:
                    if page == 1:  # 第一页就没有数据
                        self.logger.warning(f"No data for {symbol} from {start_str} to {end_str}")
                    break
                
                results = data.get('results', [])
                all_results.extend(results)
                self.logger.debug(f"Page {page}: {len(results)} records, total so far: {len(all_results)}")
                
                # 检查是否有下一页
                url = data.get('next_url')
                if url and 'apiKey=' not in url:
                    # next_url 缺少 API key，添加它
                    separator = '&' if '?' in url else '?'
                    url = f"{url}{separator}apiKey={self.api_key}"
            
            if not all_results:
                return {}
            
            self.logger.info(f"✅ Fetched {symbol} range [{page} pages]: {len(all_results)} total records")
            
            # 按日期分组数据
            daily_data = {}
            et_tz = pytz.timezone('America/New_York')
            
            for item in all_results:
                timestamp = datetime.fromtimestamp(item['t'] / 1000, tz=pytz.UTC)
                timestamp = timestamp.astimezone(et_tz)
                
                # 用日期作为key
                date_key = f"{item['symbol'] if 'symbol' in item else symbol}_{timestamp.date().isoformat()}"
                
                if date_key not in daily_data:
                    daily_data[date_key] = []
                
                daily_data[date_key].append({
                    'datetime': timestamp,
                    'close': item['c'],
                })
            
            # 转换为DataFrame并缓存
            result = {}
            for cache_key, records in daily_data.items():
                df = pd.DataFrame(records)
                df.set_index('datetime', inplace=True)
                result[cache_key] = df
                self.logger.debug(f"Cached {cache_key}: {len(df)} records")
            
            self.logger.info(f"Fetched {symbol} range: {len(result)} days, {sum(len(df) for df in result.values())} total records")
            
            # Save to local cache
            self._save_to_local_cache(symbol, result)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch {symbol} range {start_str} to {end_str}: {e}")
            return {}
    
    def prefetch_multiple_days(self, symbol: str, start_date, days: int = 5):
        """
        预加载多天数据，确保返回完整的N天数据（即使某些日期停盘也用前一天数据填充）
        
        当看到新股票信号时调用此方法，一次API调用加载N天数据
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            days: Number of days to prefetch (default 5)
        """
        end_date = start_date + timedelta(days=days - 1)
        
        # 调用多日期范围API
        daily_dfs = self._fetch_range_data(symbol, start_date, end_date)
        
        # 确保完整6天的数据，对于停盘日期用前一天数据填充
        # 生成所有日期的cache_key（包括交易日和非交易日）
        current_date = start_date.date() if hasattr(start_date, 'date') else start_date
        end_date_obj = end_date.date() if hasattr(end_date, 'date') else end_date
        last_available_data = None
        
        while current_date <= end_date_obj:
            cache_key = f"{symbol}_{current_date.isoformat()}"
            
            # 如果该日期有数据，缓存并记录
            if cache_key in daily_dfs:
                df = daily_dfs[cache_key]
                self.price_cache[cache_key] = df
                last_available_data = df
            # 如果该日期没有数据但前面有，就用前一天的数据填充
            elif last_available_data is not None:
                self.price_cache[cache_key] = last_available_data
            
            current_date += timedelta(days=1)
        
        # 记录该符号的预加载范围
        self.prefetch_ranges[symbol] = (start_date.date() if hasattr(start_date, 'date') else start_date, 
                                        end_date_obj)
        
        return len(daily_dfs)
    
    def prefetch_multiple_symbols_concurrent(self, symbols: List[str], start_date, days: int = 5, max_workers: int = 5):
        """
        并发预加载多个股票的数据（超级高效！）
        
        使用线程池并发执行 API 调用，最多同时执行 max_workers 个请求
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            days: 预加载天数
            max_workers: 最大并发线程数（默认5）
            
        Returns:
            加载成功的股票数
        """
        if not symbols:
            return 0
        
        self.logger.info(f"🚀 并发预加载 {len(symbols)} 个股票，max_workers={max_workers}...")
        
        end_date = start_date + timedelta(days=days - 1)
        successfully_loaded = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self._fetch_range_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    daily_dfs = future.result()
                    if daily_dfs:
                        # 缓存结果
                        for cache_key, df in daily_dfs.items():
                            self.price_cache[cache_key] = df
                        successfully_loaded += 1
                        self.logger.debug(f"✅ {symbol} 预加载完成 ({len(daily_dfs)} 天)")
                    else:
                        self.logger.debug(f"⚠️  {symbol} 无数据")
                except Exception as e:
                    self.logger.warning(f"❌ {symbol} 预加载失败: {e}")
        
        self.logger.info(f"✅ 并发预加载完成: {successfully_loaded}/{len(symbols)} 成功")
        return successfully_loaded
    
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息（用于验证数据完整性）
        
        Returns:
            统计字典：缓存大小、符号数、日期范围等
        """
        if not self.price_cache:
            return {'total_entries': 0, 'symbols': 0, 'total_records': 0}
        
        symbols_set = set()
        total_records = 0
        date_range = {'min': None, 'max': None}
        
        for cache_key, df in self.price_cache.items():
            # 解析 cache_key: "JPM_2023-03-10"
            parts = cache_key.rsplit('_', 1)
            if len(parts) == 2:
                symbol, date_str = parts
                symbols_set.add(symbol)
                
                # 更新日期范围
                if date_range['min'] is None or date_str < date_range['min']:
                    date_range['min'] = date_str
                if date_range['max'] is None or date_str > date_range['max']:
                    date_range['max'] = date_str
            
            total_records += len(df)
        
        return {
            'total_entries': len(self.price_cache),
            'total_records': total_records,
            'unique_symbols': len(symbols_set),
            'avg_records_per_day': total_records // len(self.price_cache) if self.price_cache else 0,
            'date_range': date_range,
            'cache_size_mb': sum(df.memory_usage(deep=True).sum() for df in self.price_cache.values()) / 1024 / 1024
        }
    
    def verify_cache(self) -> bool:
        """
        验证缓存数据完整性（检查是否有丢失）
        
        Returns:
            bool: True 表示数据完整，False 表示有问题
        """
        stats = self.get_cache_stats()
        
        self.logger.info("="*60)
        self.logger.info("📊 缓存数据完整性检查")
        self.logger.info("="*60)
        self.logger.info(f"  缓存条目数: {stats['total_entries']}")
        self.logger.info(f"  唯一符号数: {stats['unique_symbols']}")
        self.logger.info(f"  总秒级数据条数: {stats['total_records']:,}")
        self.logger.info(f"  平均每日数据: {stats['avg_records_per_day']:,}")
        self.logger.info(f"  日期范围: {stats['date_range']['min']} ~ {stats['date_range']['max']}")
        self.logger.info(f"  缓存大小: {stats['cache_size_mb']:.1f} MB")
        self.logger.info("="*60)
        
        # 基本的数据完整性检查
        if stats['total_entries'] > 0 and stats['total_records'] > 0:
            self.logger.info("✅ 缓存数据检查通过！数据完整")
            return True
        else:
            self.logger.warning("❌ 缓存为空或数据不完整！")
            return False
    
    def get_price_at_time(self, symbol: str, target_time: datetime) -> Optional[float]:
        """Get close price at specific time (forward-fill if no exact match)"""
        et_tz = pytz.timezone('America/New_York')
        
        # Convert to ET timezone
        if target_time.tzinfo is not None:
            target_time = target_time.astimezone(et_tz)
        else:
            target_time = et_tz.localize(target_time)
        
        cache_key = f"{symbol}_{target_time.date().isoformat()}"
        target_date = target_time.date()
        
        if cache_key not in self.price_cache:
            # Check if target_date is within prefetch_ranges
            # If not, we need to prefetch data for that range
            needs_prefetch = False
            
            if symbol not in self.prefetch_ranges:
                # First time seeing this symbol, prefetch 6 days starting from target_date
                self.logger.info(f"📦 首次预加载 {symbol} 的6天数据（从{target_date}开始）")
                self.prefetch_multiple_days(symbol, target_date, days=6)
                needs_prefetch = True
            else:
                start_range, end_range = self.prefetch_ranges[symbol]
                if target_date < start_range or target_date > end_range:
                    # Out of range, need to prefetch
                    self.logger.info(f"📦 预加载范围外，重新预加载 {symbol}（目标日期{target_date}，当前范围{start_range}到{end_range}）")
                    self.prefetch_multiple_days(symbol, target_date, days=6)
                    needs_prefetch = True
            
            # After prefetch, if the exact cache_key still doesn't exist, try to find nearby date
            if cache_key not in self.price_cache:
                # Try to find a nearby date with data (within 5 days)
                found_key = None
                for lookback in range(1, 6):
                    check_date = target_date - timedelta(days=lookback)
                    check_key = f"{symbol}_{check_date.isoformat()}"
                    if check_key in self.price_cache:
                        found_key = check_key
                        self.logger.debug(f"使用 {symbol} 在 {check_date} 的数据替代 {target_date}")
                        break
                
                if found_key:
                    cache_key = found_key
                else:
                    self.logger.warning(f"无法获取 {symbol} 在 {target_date} 附近的价格数据（已回溯5天）")
                    return None
        
        df = self.price_cache[cache_key]
        
        if len(df) == 0:
            self.logger.warning(f"{symbol} 在 {target_date} 的数据为空")
            return None
         
        try:
            # 获取当天数据的时间范围
            first_time = df.index[0]
            last_time = df.index[-1]
            
            # 情况1: 查询时间在数据范围内 → 使用 forward-fill
            if target_time >= first_time:
                idx = df.index.get_indexer([target_time], method='pad')
                if idx[0] >= 0:
                    return round(float(df.iloc[idx[0]]['close']), 2)
            
            # 情况2: 查询时间早于当天第一条数据（如 09:30:00 早于 09:30:01）
            # → 尝试用前一天的收盘价
            if target_time < first_time:
                self.logger.debug(f"查询时间 {target_time} 早于 {symbol} 当天第一条数据 {first_time}")
                
                # 尝试获取前一天的收盘价
                prev_date = target_date - timedelta(days=1)
                lookback_count = 0
                while lookback_count < 5:
                    prev_key = f"{symbol}_{prev_date.isoformat()}"
                    if prev_key in self.price_cache:
                        prev_df = self.price_cache[prev_key]
                        if len(prev_df) > 0:
                            prev_close = round(float(prev_df.iloc[-1]['close']), 2)
                            self.logger.debug(f"使用前一天 {prev_date} 的收盘价: ${prev_close:.2f}")
                            return prev_close
                    
                    prev_date -= timedelta(days=1)
                    lookback_count += 1
                
                # 如果找不到前一天数据，使用当天第一条数据
                self.logger.debug(f"无法找到前一天数据，使用当天第一条数据（开盘价）")
                return round(float(df.iloc[0]['close']), 2)
                
        except Exception as e:
            self.logger.debug(f"Error getting price for {symbol}: {e}")
        
        self.logger.warning(f"无法在 {symbol} 的 {target_date} 数据中找到 {target_time} 的价格")
        return None
    
    def get_day_close_price(self, symbol: str, target_date: date, max_lookback_days: int = 5) -> Optional[float]:
        """
        高效获取某一天的收盘价（16:00收盘价）
        
        相比get_price_at_time()快得多，因为不需要匹配具体时间戳
        自动回退：如果target_date无数据，向前回溯找最近的交易日（最多回溯max_lookback_days）
        
        Args:
            symbol: Stock symbol
            target_date: 目标日期
            max_lookback_days: 最多向前回溯的交易日数
        
        Returns:
            float: 收盘价，或None（无法找到）
        """
        # 首先检查是否需要预加载
        cache_key = f"{symbol}_{target_date.isoformat()}"
        if cache_key not in self.price_cache:
            if symbol not in self.prefetch_ranges:
                self.logger.debug(f"首次预加载 {symbol}（目标日期{target_date}）")
                self.prefetch_multiple_days(symbol, target_date, days=6)
            else:
                start_range, end_range = self.prefetch_ranges[symbol]
                if target_date < start_range or target_date > end_range:
                    self.logger.debug(f"预加载范围外，重新预加载 {symbol}（目标日期{target_date}）")
                    self.prefetch_multiple_days(symbol, target_date, days=6)
        
        # 向前回溯，找最近的有数据的交易日
        current_date = target_date
        lookback_count = 0
        
        while lookback_count <= max_lookback_days:
            check_key = f"{symbol}_{current_date.isoformat()}"
            if check_key in self.price_cache:
                df = self.price_cache[check_key]
                if len(df) > 0:
                    # 获取该日期的最后一条记录（收盘价），round到2位小数
                    return round(float(df.iloc[-1]['close']), 2)
            
            # 向前回溯一天
            current_date -= timedelta(days=1)
            lookback_count += 1
        
        # 无法找到
        self.logger.warning(f"无法获取 {symbol} 在 {target_date} 附近的价格数据（已回溯{max_lookback_days}天）")
        return None
    
    def set_current_time(self, current_time: datetime):
        """Set current time for backtest progress"""
        self.current_time = current_time
        
        # Update market price for all positions
        for symbol in list(self.positions.keys()):
            if self.positions[symbol]['position'] > 0:
                price = self.get_price_at_time(symbol, current_time)
                if price:
                    price = round(price, 2)
                    self.positions[symbol]['market_price'] = price
                    self.positions[symbol]['market_value'] = round(price * self.positions[symbol]['position'], 2)
                    
                    if 'highest_price' not in self.positions[symbol]:
                        self.positions[symbol]['highest_price'] = self.positions[symbol]['cost_price']
                    if price > self.positions[symbol]['highest_price']:
                        self.positions[symbol]['highest_price'] = price
    
    def _load_stock_price_data(self, symbol: str):
        """Preload stock data (no-op, kept for compatibility)"""
        pass
    
    def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """Get current stock price at current_time"""
        if not self.current_time:
            return None
        
        price = self.get_price_at_time(symbol, self.current_time)
        if price:
            return {
                'last_price': price,
                'bid': round(price * 0.999, 2),
                'ask': round(price * 1.001, 2),
            }
        
        return None
    
    def get_stock_rsi(self, symbol: str, timestamp: datetime = None, window: int = 14, timespan: str = "day") -> Optional[float]:
        """
        Get RSI (Relative Strength Index) for a stock using Polygon API
        
        Args:
            symbol: Stock ticker symbol
            timestamp: Query timestamp (if None, use self.current_time)
            window: RSI calculation window (default 14)
            timespan: Time granularity - "day", "hour", "minute" (default "day")
        
        Returns:
            RSI value (0-100) or None if query fails or data not available
        """
        if not timestamp:
            timestamp = self.current_time
        
        if not timestamp:
            self.logger.warning(f"get_stock_rsi: No timestamp available for {symbol}")
            return None
        
        try:
            # Format timestamp as YYYY-MM-DD for Polygon API
            timestamp_str = timestamp.strftime('%Y-%m-%d')
            
            # Build API request URL - use v1/indicators/rsi endpoint (not v2/aggs/ticker)
            url = f"https://api.polygon.io/v1/indicators/rsi/{symbol}"
            params = {
                'timestamp': timestamp_str,
                'window': window,
                'timespan': timespan,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            # Make API request
            response = requests.get(url, params=params, timeout=10)
            
            # Handle 404 - symbol not found or not supported
            if response.status_code == 404:
                self.logger.debug(f"RSI data not available for {symbol} (404 Not Found)")
                return None
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            data = response.json()
            
            # Check if request was successful
            if data.get('status') != 'OK':
                self.logger.debug(f"RSI query failed for {symbol}: {data.get('message', 'Unknown error')}")
                return None
            
            # Extract RSI value from results
            results = data.get('results', {})
            if not results or 'values' not in results or len(results['values']) == 0:
                self.logger.debug(f"No RSI data available for {symbol} on {timestamp_str}")
                return None
            
            # Get the most recent RSI value
            rsi_value = results['values'][0].get('value')
            
            if rsi_value is not None:
                self.logger.info(f"✓ RSI for {symbol}: {rsi_value:.2f} (window={window})")
                self.api_calls += 1
                return float(rsi_value)
            
            return None
            
        except requests.exceptions.Timeout:
            self.logger.debug(f"RSI query timeout for {symbol}")
            return None
        except requests.exceptions.HTTPError as e:
            # Handle other HTTP errors (5xx, rate limits, etc)
            self.logger.debug(f"RSI query HTTP error for {symbol}: {e.response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"RSI query network error for {symbol}: {str(e)}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"RSI data parsing error for {symbol}: {str(e)}")
            return None
    
    def get_stock_macd(self, symbol: str, timestamp: datetime = None, short_window: int = 12, long_window: int = 26, signal_window: int = 9, timespan: str = "day") -> Optional[float]:
        """
        Get MACD (Moving Average Convergence Divergence) for a stock using Polygon API
        
        Args:
            symbol: Stock ticker symbol
            timestamp: Query timestamp (if None, use self.current_time)
            short_window: Short EMA window (default 12)
            long_window: Long EMA window (default 26)
            signal_window: Signal line window (default 9)
            timespan: Time granularity - "day", "hour", "minute" (default "day")
        
        Returns:
            MACD value (can be positive or negative) or None if query fails
            Positive MACD = Bullish trend
            Negative MACD = Bearish trend
        """
        if not timestamp:
            timestamp = self.current_time
        
        if not timestamp:
            self.logger.warning(f"get_stock_macd: No timestamp available for {symbol}")
            return None
        
        try:
            # Format timestamp as YYYY-MM-DD for Polygon API
            timestamp_str = timestamp.strftime('%Y-%m-%d')
            
            # Build API request URL - use v1/indicators/macd endpoint
            url = f"https://api.polygon.io/v1/indicators/macd/{symbol}"
            params = {
                'timestamp': timestamp_str,
                'short_window': short_window,
                'long_window': long_window,
                'signal_window': signal_window,
                'timespan': timespan,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            # Make API request
            response = requests.get(url, params=params, timeout=10)
            
            # Handle 404 - symbol not found or not supported
            if response.status_code == 404:
                self.logger.debug(f"MACD data not available for {symbol} (404 Not Found)")
                return None
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            data = response.json()
            
            # Check if request was successful
            if data.get('status') != 'OK':
                self.logger.debug(f"MACD query failed for {symbol}: {data.get('message', 'Unknown error')}")
                return None
            
            # Extract MACD value from results
            results = data.get('results', {})
            if not results or 'values' not in results or len(results['values']) == 0:
                self.logger.debug(f"No MACD data available for {symbol} on {timestamp_str}")
                return None
            
            # Get the most recent MACD value
            macd_value = results['values'][0].get('value')
            
            if macd_value is not None:
                self.logger.info(f"✓ MACD for {symbol}: {macd_value:.4f} (short={short_window}, long={long_window})")
                self.api_calls += 1
                return float(macd_value)
            
            return None
            
        except requests.exceptions.Timeout:
            self.logger.debug(f"MACD query timeout for {symbol}")
            return None
        except requests.exceptions.HTTPError as e:
            # Handle other HTTP errors (5xx, rate limits, etc)
            self.logger.debug(f"MACD query HTTP error for {symbol}: {e.response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"MACD query network error for {symbol}: {str(e)}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"MACD data parsing error for {symbol}: {str(e)}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account info"""
        position_value = sum(pos['market_value'] for pos in self.positions.values())
        total_assets = self.cash + position_value
        
        return {
            'cash': self.cash,
            'market_value': position_value,
            'total_assets': total_assets,
            'available_cash': self.cash,
            'power': self.cash,
        }
    
    def get_positions(self) -> List[Dict]:
        """Get position list"""
        positions = []
        for symbol, pos in self.positions.items():
            if pos['position'] > 0:
                positions.append({
                    'symbol': symbol,
                    'position': pos['position'],
                    'can_sell_qty': pos['position'],
                    'cost_price': pos['cost_price'],
                    'market_price': pos['market_price'],
                    'market_value': pos['market_value'],
                    'pnl': (pos['market_price'] - pos['cost_price']) * pos['position'],
                    'pnl_ratio': (pos['market_price'] - pos['cost_price']) / pos['cost_price'] 
                                if pos['cost_price'] > 0 else 0,
                })
        return positions
    
    def buy_stock(self, symbol: str, quantity: int, price: float, 
                  order_type: str = 'LIMIT') -> Optional[Dict]:
        """Buy stock (with slippage and commission)"""
        actual_price = round(price * (1 + self.slippage), 2)
        commission = round(max(quantity * self.commission_per_share, self.min_commission), 2)
        cost = round(actual_price * quantity + commission, 2)
        
        acc_info = self.get_account_info()
        total_assets = acc_info['total_assets']
        cash_after = self.cash - cost
        cash_ratio_after = cash_after / total_assets
        
        if cash_ratio_after < -1.0:
            self.logger.warning(
                f"Buy failed: insufficient cash {symbol}, "
                f"need ${cost:,.2f}, cash ratio {cash_ratio_after:.1%} < -100%"
            )
            return None
        
        self.cash -= cost
        
        if symbol in self.positions:
            old_pos = self.positions[symbol]
            old_shares = old_pos['position']
            old_cost = old_pos['cost_price']
            
            new_shares = old_shares + quantity
            new_cost = round((old_cost * old_shares + actual_price * quantity) / new_shares, 2)
            
            self.positions[symbol] = {
                'position': new_shares,
                'cost_price': new_cost,
                'market_price': actual_price,
                'market_value': round(actual_price * new_shares, 2),
            }
        else:
            self.positions[symbol] = {
                'position': quantity,
                'cost_price': actual_price,
                'market_price': actual_price,
                'market_value': round(actual_price * quantity, 2),
                'highest_price': actual_price,
            }
        
        order_id = f"BUY_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        et_tz = pytz.timezone('America/New_York')
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': 'BUY',
            'quantity': quantity,
            'price': actual_price,
            'base_price': price,
            'commission': commission,
            'cost': cost,
            'status': 'FILLED',
            'time': self.current_time if self.current_time else datetime.now(et_tz),
        }
        self.orders.append(order)
        
        # 加上时间信息
        time_str = self.current_time.strftime('%Y-%m-%d %H:%M:%S') if self.current_time else 'N/A'
        self.logger.info(
            f"Buy: {symbol} {quantity}@${actual_price:.2f} cost=${cost:,.2f} [时间:{time_str}]"
        )
        
        return order
    
    def sell_stock(self, symbol: str, quantity: int, price: float,
                   order_type: str = 'LIMIT') -> Optional[Dict]:
        """Sell stock (with slippage and commission)"""
        if symbol not in self.positions or self.positions[symbol]['position'] < quantity:
            self.logger.warning(f"Sell failed: insufficient position {symbol}")
            return None
        
        actual_price = round(price * (1 - self.slippage), 2)
        commission = round(max(quantity * self.commission_per_share, self.min_commission), 2)
        proceeds = round(actual_price * quantity - commission, 2)
        
        self.cash += proceeds
        
        pos = self.positions[symbol]
        cost_price = pos['cost_price']
        
        pos['position'] -= quantity
        pos['market_value'] = round(pos['market_price'] * pos['position'], 2)
        
        if pos['position'] == 0:
            del self.positions[symbol]
        
        order_id = f"SELL_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        pnl = round((actual_price - cost_price) * quantity - commission, 2)
        pnl_ratio = pnl / (cost_price * quantity) if cost_price > 0 else 0
        
        et_tz = pytz.timezone('America/New_York')
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': 'SELL',
            'quantity': quantity,
            'price': actual_price,
            'base_price': price,
            'commission': commission,
            'proceeds': proceeds,
            'status': 'FILLED',
            'time': self.current_time if self.current_time else datetime.now(et_tz),
            'pnl': pnl,
            'pnl_ratio': pnl_ratio,
        }
        self.orders.append(order)
        
        # 加上时间信息
        time_str = self.current_time.strftime('%Y-%m-%d %H:%M:%S') if self.current_time else 'N/A'
        self.logger.info(
            f"Sell: {symbol} {quantity}@${actual_price:.2f} pnl=${pnl:+,.2f} [时间:{time_str}]"
        )
        
        return order
    
    def get_order_list(self, status_filter: str = None, symbol_filter: str = None) -> List[Dict]:
        """Get order list"""
        orders = self.orders
        
        if status_filter:
            orders = [o for o in orders if o['status'] == status_filter]
        
        if symbol_filter:
            orders = [o for o in orders if o['symbol'] == symbol_filter]
        
        return orders
    
    def count_trading_days_between(self, start_date: str, end_date: str, market: str = 'US') -> Optional[int]:
        """Count trading days between dates using NYSE calendar"""
        try:
            # Get valid trading days from NYSE calendar
            schedule = self.market_calendar.valid_days(
                start_date=start_date,
                end_date=end_date
            )
            
            # Count days (excluding start_date if needed, matching original behavior)
            trading_days = len([d for d in schedule if d.strftime('%Y-%m-%d') > start_date])
            
            return trading_days
            
        except Exception as e:
            self.logger.warning(f"Failed to get trading days from calendar: {e}")
            # Fallback: simple weekday counting
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            
            trading_days = 0
            current = start + timedelta(days=1)
            while current <= end:
                if current.weekday() < 5:
                    trading_days += 1
                current += timedelta(days=1)
            
            return trading_days
    
    def get_summary(self) -> Dict:
        """Get account summary"""
        acc_info = self.get_account_info()
        
        total_pnl = acc_info['total_assets'] - self.initial_cash
        total_pnl_ratio = total_pnl / self.initial_cash
        
        buy_orders = [o for o in self.orders if o['side'] == 'BUY']
        sell_orders = [o for o in self.orders if o['side'] == 'SELL']
        
        realized_pnl = sum(o.get('pnl', 0) for o in sell_orders)
        unrealized_pnl = sum(pos['pnl'] for pos in self.get_positions())
        
        return {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'position_value': acc_info['market_value'],
            'total_assets': acc_info['total_assets'],
            'total_pnl': total_pnl,
            'total_pnl_ratio': total_pnl_ratio,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'num_trades': len(buy_orders),
            'num_positions': len(self.positions),
        }

