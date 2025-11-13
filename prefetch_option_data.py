#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权数据预下载脚本
Pre-download all option data from CSV signals to cache
从CSV中读取所有期权信号，预先下载从信号时间到到期日的所有历史数据
"""

import os
import sys
import csv
import pytz
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prefetch_option_data.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptionDataPrefetcher:
    """期权数据预下载器"""
    
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
    
    def __init__(self, csv_file: str, cache_dir: str):
        """
        初始化预下载器
        
        Args:
            csv_file: CSV信号文件路径
            cache_dir: 缓存目录
        """
        self.csv_file = Path(csv_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in .env file")
        
        self.api_calls = 0
        self.cache_hits = 0
        self.failed_tickers = []
        
        # 时区
        self.cn_tz = pytz.timezone('Asia/Shanghai')
        self.et_tz = pytz.timezone('America/New_York')
        
        logger.info(f"📂 CSV文件: {self.csv_file}")
        logger.info(f"📁 缓存目录: {self.cache_dir}")
    
    def load_option_signals(self) -> List[Dict]:
        """
        从CSV加载所有期权信号
        
        Returns:
            期权信号列表
        """
        logger.info(f"📖 读取CSV文件: {self.csv_file}")
        signals = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                try:
                    signal = self._parse_signal(row)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.debug(f"行{idx}解析失败: {e}")
                    continue
        
        logger.info(f"✅ 共读取 {len(signals)} 个期权信号")
        return signals
    
    def _parse_signal(self, row: Dict) -> Optional[Dict]:
        """解析CSV行"""
        try:
            ticker = row['ticker']
            date_str = row['date']
            time_str = row['time']
            strike = float(row['strike'])
            option_type = row['option_type'].lower()
            expiry_str = row['expiry']
            
            # 时间转换：中国时间 → 美东时间
            datetime_str = f"{date_str} {time_str}"
            signal_time_cn = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            signal_time_cn = self.cn_tz.localize(signal_time_cn)
            signal_time_et = signal_time_cn.astimezone(self.et_tz)
            
            # 解析到期日
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
            
            # 构建期权代码
            option_ticker = self._construct_option_ticker(ticker, expiry_date, option_type, strike)
            
            return {
                'option_ticker': option_ticker,
                'ticker': ticker,
                'strike': strike,
                'option_type': option_type,
                'expiry': expiry_date,
                'signal_time_et': signal_time_et,
                'signal_date_et': signal_time_et.date()
            }
            
        except Exception as e:
            logger.debug(f"解析失败: {e}")
            return None
    
    def _construct_option_ticker(self, underlying: str, expiry, option_type: str, strike: float) -> str:
        """构建Polygon期权代码"""
        date_str = expiry.strftime('%y%m%d')
        cp = 'C' if option_type.lower() == 'call' else 'P'
        strike_str = f"{int(strike * 1000):08d}"
        return f"O:{underlying}{date_str}{cp}{strike_str}"
    
    def check_cache_exists(self, option_ticker: str, start_date, end_date) -> Dict:
        """
        检查缓存中已有哪些日期的数据
        
        Returns:
            {'cached_dates': [...], 'missing_dates': [...]}
        """
        all_dates = pd.date_range(start_date, end_date, freq='D')
        cached_dates = []
        missing_dates = []
        
        for date in all_dates:
            cache_file = self.cache_dir / f"{option_ticker.replace(':', '_')}_{date.date().isoformat()}.parquet"
            if cache_file.exists():
                cached_dates.append(date.date())
            else:
                missing_dates.append(date.date())
        
        return {
            'cached_dates': cached_dates,
            'missing_dates': missing_dates
        }
    
    def download_option_data(self, option_ticker: str, start_date, end_date) -> bool:
        """
        下载期权数据并保存到缓存
        
        Args:
            option_ticker: 期权代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            True if successful, False otherwise
        """
        try:
            start_str = start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date)
            end_str = end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date)
            
            # Polygon API endpoint
            url = (f"{self.BASE_URL}/{option_ticker}/range/1/minute"
                   f"/{start_str}/{end_str}"
                   f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            self.api_calls += 1
            logger.info(f"📥 Downloading {option_ticker} ({start_str} to {end_str}) [API #{self.api_calls}]...")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('resultsCount', 0) == 0:
                logger.warning(f"⚠️  No data for {option_ticker} in range {start_str} to {end_str}")
                self.failed_tickers.append(option_ticker)
                return False
            
            results = data.get('results', [])
            records = []
            
            for item in results:
                timestamp = datetime.fromtimestamp(item['t'] / 1000, tz=pytz.UTC)
                timestamp = timestamp.astimezone(self.et_tz)
                
                records.append({
                    'datetime': timestamp,
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item.get('v', 0),
                })
            
            if not records:
                logger.warning(f"⚠️  No records parsed for {option_ticker}")
                return False
            
            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            
            # 按天拆分并保存
            days_saved = 0
            for day in pd.date_range(start_str, end_str, freq='D'):
                day_data = df[df.index.date == day.date()]
                if len(day_data) > 0:
                    self._save_to_cache(option_ticker, day.date(), day_data)
                    days_saved += 1
            
            logger.info(f"✅ Saved {option_ticker}: {len(records)} bars → {days_saved} days")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"⚠️  {option_ticker} not found (404)")
            else:
                logger.warning(f"⚠️  HTTP error: {e}")
            self.failed_tickers.append(option_ticker)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download {option_ticker}: {e}")
            self.failed_tickers.append(option_ticker)
            return False
    
    def _save_to_cache(self, option_ticker: str, date, df: pd.DataFrame):
        """保存数据到缓存"""
        try:
            date_str = date.isoformat() if hasattr(date, 'isoformat') else str(date)
            cache_file = self.cache_dir / f"{option_ticker.replace(':', '_')}_{date_str}.parquet"
            
            # Reset index to save datetime as column
            df_to_save = df.reset_index()
            df_to_save.to_parquet(cache_file, index=False)
            
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
    
    def prefetch_all(self, max_workers: int = 5, start_date_filter: str = "2024-01-01"):
        """
        预下载所有期权数据（支持并发）
        
        Args:
            max_workers: 并发线程数（建议3-10，避免触发限流）
            start_date_filter: 只下载此日期之后的期权（默认2024-01-01）
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 开始预下载所有期权数据")
        logger.info("="*60)
        logger.info(f"📅 日期过滤: 只下载 {start_date_filter} 及之后的期权")
        
        # 转换过滤日期
        filter_date = datetime.strptime(start_date_filter, '%Y-%m-%d').date()
        
        # 加载信号
        signals = self.load_option_signals()
        
        if not signals:
            logger.warning("⚠️  没有找到期权信号")
            return
        
        # 过滤：只保留信号时间在过滤日期之后的
        filtered_signals = [sig for sig in signals if sig['signal_date_et'] >= filter_date]
        logger.info(f"📊 过滤后信号数: {len(signals)} → {len(filtered_signals)}")
        
        # 去重：同一个期权只下载一次
        unique_options = {}
        for sig in filtered_signals:
            option_ticker = sig['option_ticker']
            if option_ticker not in unique_options:
                unique_options[option_ticker] = sig
            else:
                # 取最早的信号时间（但不早于过滤日期）
                if sig['signal_date_et'] < unique_options[option_ticker]['signal_date_et']:
                    unique_options[option_ticker] = sig
        
        logger.info(f"📊 去重后共 {len(unique_options)} 个唯一期权")
        
        # 检查已有缓存
        tasks = []
        for option_ticker, sig in unique_options.items():
            start_date = sig['signal_date_et']
            end_date = sig['expiry']
            
            # 确保开始日期不早于过滤日期
            if start_date < filter_date:
                start_date = filter_date
            
            cache_status = self.check_cache_exists(option_ticker, start_date, end_date)
            
            if len(cache_status['missing_dates']) == 0:
                logger.info(f"✅ {option_ticker} 缓存已完整，跳过")
                self.cache_hits += 1
            else:
                tasks.append({
                    'option_ticker': option_ticker,
                    'start_date': start_date,
                    'end_date': end_date,
                    'cached_days': len(cache_status['cached_dates']),
                    'missing_days': len(cache_status['missing_dates'])
                })
        
        logger.info(f"📋 需要下载的期权数: {len(tasks)}")
        logger.info(f"💾 已有完整缓存: {self.cache_hits} 个")
        
        if not tasks:
            logger.info("🎉 所有期权数据已缓存，无需下载！")
            return
        
        # 并发下载
        logger.info(f"\n🔄 开始并发下载 (并发数: {max_workers})...\n")
        
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.download_option_data,
                    task['option_ticker'],
                    task['start_date'],
                    task['end_date']
                ): task for task in tasks
            }
            
            for idx, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    # 进度显示
                    if idx % 10 == 0 or idx == len(tasks):
                        logger.info(
                            f"⏳ 进度: {idx}/{len(tasks)} "
                            f"({idx/len(tasks)*100:.1f}%) | "
                            f"成功: {success_count} | 失败: {failed_count}"
                        )
                
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Download task failed: {e}")
        
        # 统计结果
        logger.info("\n" + "="*60)
        logger.info("📊 下载完成统计")
        logger.info("="*60)
        logger.info(f"✅ 成功下载: {success_count} 个期权")
        logger.info(f"❌ 下载失败: {failed_count} 个期权")
        logger.info(f"💾 已有缓存: {self.cache_hits} 个期权")
        logger.info(f"📞 API调用数: {self.api_calls}")
        logger.info(f"📁 缓存目录: {self.cache_dir}")
        
        if self.failed_tickers:
            logger.warning(f"\n⚠️  以下 {len(self.failed_tickers)} 个期权下载失败:")
            for ticker in self.failed_tickers[:20]:  # 只显示前20个
                logger.warning(f"  - {ticker}")
            if len(self.failed_tickers) > 20:
                logger.warning(f"  ... 还有 {len(self.failed_tickers) - 20} 个")
        
        logger.info("\n🎉 预下载完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='期权数据预下载 - 批量下载CSV中所有期权的历史数据')
    parser.add_argument(
        '--csv', 
        default='future_v_0_1/database/merged_strategy_v1_calls_bell_2023M3_2025M10.csv',
        help='CSV信号文件路径'
    )
    parser.add_argument(
        '--cache-dir',
        default='future_v_0_1/database/option_cache',
        help='缓存目录'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='并发下载线程数（1-10，建议5）'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='只下载此日期及之后的期权数据（默认2024-01-01，避免403错误）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📥 期权数据预下载系统")
    print("="*60)
    print(f"📂 CSV文件: {args.csv}")
    print(f"📁 缓存目录: {args.cache_dir}")
    print(f"🔄 并发数: {args.workers}")
    print(f"📅 起始日期: {args.start_date} 及之后")
    print("="*60 + "\n")
    
    # 创建预下载器
    prefetcher = OptionDataPrefetcher(
        csv_file=args.csv,
        cache_dir=args.cache_dir
    )
    
    # 开始下载
    start_time = datetime.now()
    logger.info(f"🚀 开始时间: {start_time}")
    
    try:
        prefetcher.prefetch_all(max_workers=args.workers, start_date_filter=args.start_date)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  用户中断下载")
        print("\n⚠️  下载已中断")
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        print(f"\n❌ 下载失败: {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"✅ 结束时间: {end_time}")
    logger.info(f"⏱️  总用时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    
    print("\n" + "="*60)
    print("✅ 预下载完成！")
    print(f"⏱️  用时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    print(f"📁 缓存位置: {args.cache_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

