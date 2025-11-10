"""
Option Backtesting Market Client using Polygon.io Options Data
期权回测市场客户端 - 使用 Polygon.io 期权历史数据
"""

import os
import logging
import pytz
import requests
import pandas as pd
import pandas_market_calendars as mcal
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


class OptionBacktestClient:
    """
    Option backtesting client using Polygon API
    专门用于期权卖出回测
    """
    
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
    
    def __init__(self, initial_cash: float = 100000.0,
                 slippage: float = 0.01, commission_per_contract: float = 0.65, 
                 min_commission: float = 1.0, cache_dir: str = None):
        """
        Initialize option backtesting client
        
        Args:
            initial_cash: Initial cash amount (初始资金)
            slippage: Single-side slippage for options (default 1%, 期权滑点通常较大)
            commission_per_contract: Commission per contract (default $0.65, 每张合约佣金)
            min_commission: Minimum commission (default $1)
            cache_dir: Local cache directory for parquet files (本地缓存目录)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Polygon API
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in .env file")
        
        # Local cache directory (确保在 database 目录下)
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'database' / 'option_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"📂 Option cache directory: {self.cache_dir}")
        
        # Data cache {option_ticker_date: DataFrame}
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.api_calls = 0
        self.cache_hits = 0
        
        # Market calendar (NYSE)
        self.market_calendar = mcal.get_calendar('NYSE')
        
        # Trading costs
        self.slippage = slippage
        self.commission_per_contract = commission_per_contract
        self.min_commission = min_commission
        
        # Account info
        self.initial_cash = initial_cash
        self.cash = initial_cash
        
        # Option positions: {option_ticker: {contracts, avg_premium, ...}}
        self.option_positions: Dict[str, Dict] = {}
        
        # Orders
        self.orders: List[Dict] = []
        self.order_id_counter = 1
        
        # Current time
        self.current_time: Optional[datetime] = None
        
        self.logger.info(
            f"Option backtest client initialized: cash=${initial_cash:,.2f}, "
            f"slippage={slippage:.1%}, commission=${commission_per_contract}/contract"
        )
    
    def connect(self) -> bool:
        """Simulate connection"""
        self.logger.info("Connected (Polygon Options API)")
        return True
    
    def disconnect(self):
        """Simulate disconnection"""
        self.logger.info("Disconnected")
        self.logger.info(
            f"API stats: {self.api_calls} calls, "
            f"{self.cache_hits} cache hits"
        )
    
    def construct_option_ticker(self, underlying: str, expiry: date, 
                               option_type: str, strike: float) -> str:
        """
        Construct Polygon option ticker format
        
        Format: O:{underlying}{YYMMDD}{C/P}{strike*1000}
        Example: O:SPY230317C00400000 (SPY Call $400 expiring 2023-03-17)
        
        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiry: Expiration date
            option_type: 'call' or 'put'
            strike: Strike price (e.g., 400.0)
        
        Returns:
            Option ticker in Polygon format
        """
        # Format date as YYMMDD
        date_str = expiry.strftime('%y%m%d')
        
        # Call/Put indicator
        cp = 'C' if option_type.lower() == 'call' else 'P'
        
        # Strike price with 8 digits (multiply by 1000)
        strike_str = f"{int(strike * 1000):08d}"
        
        option_ticker = f"O:{underlying}{date_str}{cp}{strike_str}"
        
        return option_ticker
    
    def _load_from_local_cache(self, option_ticker: str, target_date) -> Optional[pd.DataFrame]:
        """Load option data from local parquet cache"""
        try:
            date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
            cache_file = self.cache_dir / f"{option_ticker.replace(':', '_')}_{date_str}.parquet"
            
            if not cache_file.exists():
                return None
            
            df = pd.read_parquet(cache_file)
            
            # Ensure datetime has timezone
            if df['datetime'].dt.tz is None:
                et_tz = pytz.timezone('America/New_York')
                df['datetime'] = df['datetime'].dt.tz_localize(et_tz)
            
            df.set_index('datetime', inplace=True)
            
            self.logger.debug(f"💾 Loaded {option_ticker} from local cache: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.debug(f"Failed to load from local cache: {e}")
            return None
    
    def _save_to_local_cache(self, option_ticker: str, target_date, df: pd.DataFrame):
        """Save option data to local parquet cache"""
        try:
            date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
            cache_file = self.cache_dir / f"{option_ticker.replace(':', '_')}_{date_str}.parquet"
            
            # Reset index to save datetime as column
            df_to_save = df.reset_index()
            df_to_save.to_parquet(cache_file, index=False)
            
            self.logger.debug(f"💾 Saved {option_ticker} to local cache: {cache_file}")
            
        except Exception as e:
            self.logger.debug(f"Failed to save to local cache: {e}")
    
    def _fetch_option_day_data(self, option_ticker: str, target_date) -> Optional[pd.DataFrame]:
        """
        Fetch option price data for a single day
        
        Args:
            option_ticker: Polygon option ticker (e.g., 'O:SPY230317C00400000')
            target_date: Target date
        
        Returns:
            DataFrame with datetime index and columns: open, high, low, close, volume
        """
        date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
        
        # Try local cache first
        cached_df = self._load_from_local_cache(option_ticker, target_date)
        if cached_df is not None:
            self.cache_hits += 1
            return cached_df
        
        try:
            # Polygon Options API endpoint
            url = (f"{self.BASE_URL}/{option_ticker}/range/1/minute"
                   f"/{date_str}/{date_str}"
                   f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
            
            self.api_calls += 1
            self.logger.info(f"Fetching option {option_ticker} on {date_str} (API call #{self.api_calls})...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('resultsCount', 0) == 0:
                self.logger.warning(f"No option data for {option_ticker} on {date_str}")
                return None
            
            results = data.get('results', [])
            records = []
            et_tz = pytz.timezone('America/New_York')
            
            for item in results:
                timestamp = datetime.fromtimestamp(item['t'] / 1000, tz=pytz.UTC)
                timestamp = timestamp.astimezone(et_tz)
                
                records.append({
                    'datetime': timestamp,
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item.get('v', 0),
                })
            
            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            
            # Save to cache
            self._save_to_local_cache(option_ticker, target_date, df)
            
            self.logger.debug(f"Fetched {option_ticker} on {date_str}: {len(df)} records")
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.warning(f"Option {option_ticker} not found (404) - may have expired or invalid ticker")
            else:
                self.logger.warning(f"HTTP error fetching {option_ticker}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to fetch {option_ticker} on {date_str}: {e}")
            return None
    
    def get_option_price_at_time(self, option_ticker: str, target_time: datetime) -> Optional[float]:
        """
        Get option price at specific time (forward-fill if no exact match)
        
        Args:
            option_ticker: Polygon option ticker
            target_time: Target datetime (ET timezone)
        
        Returns:
            Option price (close price) or None
        """
        et_tz = pytz.timezone('America/New_York')
        
        # Convert to ET timezone
        if target_time.tzinfo is not None:
            target_time = target_time.astimezone(et_tz)
        else:
            target_time = et_tz.localize(target_time)
        
        target_date = target_time.date()
        cache_key = f"{option_ticker}_{target_date.isoformat()}"
        
        # Check cache
        if cache_key not in self.price_cache:
            df = self._fetch_option_day_data(option_ticker, target_date)
            if df is None or len(df) == 0:
                self.logger.warning(f"No data for {option_ticker} on {target_date}")
                return None
            self.price_cache[cache_key] = df
        
        df = self.price_cache[cache_key]
        
        try:
            # Forward-fill to get price at target_time
            idx = df.index.get_indexer([target_time], method='pad')
            if idx[0] >= 0:
                return round(float(df.iloc[idx[0]]['close']), 2)
            
            # If target_time is before first data point, use first available price
            if target_time < df.index[0]:
                self.logger.debug(f"Target time before first data, using opening price")
                return round(float(df.iloc[0]['close']), 2)
                
        except Exception as e:
            self.logger.debug(f"Error getting option price: {e}")
        
        return None
    
    def set_current_time(self, current_time: datetime):
        """Set current time for backtest progress"""
        self.current_time = current_time
    
    def sell_option(self, underlying: str, expiry: date, option_type: str, 
                   strike: float, contracts: int, premium: float) -> Optional[Dict]:
        """
        Sell (write) option contracts
        卖出（开仓）期权合约
        
        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiry: Expiration date
            option_type: 'call' or 'put'
            strike: Strike price
            contracts: Number of contracts to sell
            premium: Premium per contract (received from buyer)
        
        Returns:
            Order dict or None if failed
        """
        option_ticker = self.construct_option_ticker(underlying, expiry, option_type, strike)
        
        # Calculate actual premium after slippage and commission
        # When selling, we receive LESS due to slippage
        actual_premium = round(premium * (1 - self.slippage), 2)
        commission = round(max(contracts * self.commission_per_contract, self.min_commission), 2)
        
        # Total credit received (premium * 100 shares per contract - commission)
        credit_received = round(actual_premium * contracts * 100 - commission, 2)
        
        # Check margin requirement (simplified: use 20% of underlying value as margin)
        # In real trading, margin requirements are more complex
        underlying_value_estimate = strike * contracts * 100
        margin_required = underlying_value_estimate * 0.2  # Simplified margin calculation
        
        if self.cash < margin_required:
            self.logger.warning(
                f"Sell option failed: insufficient margin. Need ${margin_required:,.2f}, "
                f"available cash ${self.cash:,.2f}"
            )
            return None
        
        # Receive premium
        self.cash += credit_received
        
        # Track position (negative contracts = short position)
        if option_ticker in self.option_positions:
            pos = self.option_positions[option_ticker]
            old_contracts = pos['contracts']
            old_avg_premium = pos['avg_premium']
            
            new_contracts = old_contracts - contracts
            new_avg_premium = ((old_avg_premium * old_contracts + actual_premium * contracts) / 
                              abs(new_contracts)) if new_contracts != 0 else 0
            
            self.option_positions[option_ticker] = {
                'contracts': new_contracts,
                'avg_premium': round(new_avg_premium, 2),
                'underlying': underlying,
                'expiry': expiry,
                'option_type': option_type,
                'strike': strike,
            }
        else:
            self.option_positions[option_ticker] = {
                'contracts': -contracts,  # Negative = short position
                'avg_premium': actual_premium,
                'underlying': underlying,
                'expiry': expiry,
                'option_type': option_type,
                'strike': strike,
            }
        
        order_id = f"SELL_OPT_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        et_tz = pytz.timezone('America/New_York')
        order = {
            'order_id': order_id,
            'option_ticker': option_ticker,
            'underlying': underlying,
            'side': 'SELL',  # Sell to open (short position)
            'contracts': contracts,
            'premium': actual_premium,
            'base_premium': premium,
            'commission': commission,
            'credit_received': credit_received,
            'status': 'FILLED',
            'time': self.current_time if self.current_time else datetime.now(et_tz),
            'strike': strike,
            'expiry': expiry.isoformat(),
            'option_type': option_type,
        }
        self.orders.append(order)
        
        time_str = self.current_time.strftime('%Y-%m-%d %H:%M:%S') if self.current_time else 'N/A'
        self.logger.info(
            f"Sell Option: {option_ticker} {contracts} contracts @${actual_premium:.2f} "
            f"credit=${credit_received:,.2f} [时间:{time_str}]"
        )
        
        return order
    
    def buy_to_close_option(self, option_ticker: str, contracts: int, 
                           premium: float) -> Optional[Dict]:
        """
        Buy to close option position (cover short)
        买入平仓（回补空头仓位）
        
        Args:
            option_ticker: Option ticker
            contracts: Number of contracts to buy back
            premium: Current premium per contract (cost to buy back)
        
        Returns:
            Order dict or None if failed
        """
        if option_ticker not in self.option_positions:
            self.logger.warning(f"Buy to close failed: no position in {option_ticker}")
            return None
        
        pos = self.option_positions[option_ticker]
        
        if pos['contracts'] >= 0 or abs(pos['contracts']) < contracts:
            self.logger.warning(
                f"Buy to close failed: insufficient short position. "
                f"Have {abs(pos['contracts'])} contracts, trying to buy {contracts}"
            )
            return None
        
        # Calculate actual premium after slippage and commission
        # When buying to close, we pay MORE due to slippage
        actual_premium = round(premium * (1 + self.slippage), 2)
        commission = round(max(contracts * self.commission_per_contract, self.min_commission), 2)
        
        # Total debit (cost to buy back)
        debit_paid = round(actual_premium * contracts * 100 + commission, 2)
        
        if self.cash < debit_paid:
            self.logger.warning(f"Buy to close failed: insufficient cash ${self.cash:,.2f} < ${debit_paid:,.2f}")
            return None
        
        # Pay to close position
        self.cash -= debit_paid
        
        # Calculate P&L
        original_premium = pos['avg_premium']
        pnl = round((original_premium - actual_premium) * contracts * 100 - commission, 2)
        pnl_ratio = pnl / (original_premium * contracts * 100) if original_premium > 0 else 0
        
        # Update position
        pos['contracts'] += contracts
        if pos['contracts'] == 0:
            del self.option_positions[option_ticker]
        
        order_id = f"BUY_CLOSE_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        et_tz = pytz.timezone('America/New_York')
        order = {
            'order_id': order_id,
            'option_ticker': option_ticker,
            'underlying': pos['underlying'],
            'side': 'BUY_TO_CLOSE',
            'contracts': contracts,
            'premium': actual_premium,
            'base_premium': premium,
            'commission': commission,
            'debit_paid': debit_paid,
            'status': 'FILLED',
            'time': self.current_time if self.current_time else datetime.now(et_tz),
            'pnl': pnl,
            'pnl_ratio': pnl_ratio,
            'original_premium': original_premium,
            'strike': pos['strike'],
            'expiry': pos['expiry'].isoformat(),
            'option_type': pos['option_type'],
        }
        self.orders.append(order)
        
        time_str = self.current_time.strftime('%Y-%m-%d %H:%M:%S') if self.current_time else 'N/A'
        self.logger.info(
            f"Buy to Close: {option_ticker} {contracts} contracts @${actual_premium:.2f} "
            f"pnl=${pnl:+,.2f} ({pnl_ratio:+.1%}) [时间:{time_str}]"
        )
        
        return order
    
    def get_option_positions(self) -> List[Dict]:
        """Get current option positions"""
        positions = []
        for option_ticker, pos in self.option_positions.items():
            if pos['contracts'] != 0:
                # Get current option price
                current_premium = None
                if self.current_time:
                    current_premium = self.get_option_price_at_time(option_ticker, self.current_time)
                
                # Calculate unrealized P&L
                unrealized_pnl = 0
                if current_premium is not None and pos['contracts'] < 0:  # Short position
                    # Profit = premium received - current premium (cost to buy back)
                    unrealized_pnl = (pos['avg_premium'] - current_premium) * abs(pos['contracts']) * 100
                
                positions.append({
                    'option_ticker': option_ticker,
                    'underlying': pos['underlying'],
                    'contracts': pos['contracts'],
                    'avg_premium': pos['avg_premium'],
                    'current_premium': current_premium,
                    'unrealized_pnl': round(unrealized_pnl, 2),
                    'strike': pos['strike'],
                    'expiry': pos['expiry'],
                    'option_type': pos['option_type'],
                })
        
        return positions
    
    def get_account_info(self) -> Dict:
        """Get account info"""
        positions = self.get_option_positions()
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
        
        return {
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'total_assets': self.cash + unrealized_pnl,
            'num_positions': len(positions),
        }
    
    def get_summary(self) -> Dict:
        """Get account summary"""
        acc_info = self.get_account_info()
        
        total_pnl = acc_info['total_assets'] - self.initial_cash
        total_pnl_ratio = total_pnl / self.initial_cash
        
        sell_orders = [o for o in self.orders if o['side'] == 'SELL']
        buy_close_orders = [o for o in self.orders if o['side'] == 'BUY_TO_CLOSE']
        
        realized_pnl = sum(o.get('pnl', 0) for o in buy_close_orders)
        unrealized_pnl = acc_info['unrealized_pnl']
        
        # Calculate premium received
        total_premium_received = sum(o.get('credit_received', 0) for o in sell_orders)
        
        return {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'total_assets': acc_info['total_assets'],
            'total_pnl': total_pnl,
            'total_pnl_ratio': total_pnl_ratio,
            'realized_pnl': realized_pnl,
            'num_trades': len(sell_orders),
            'num_positions': acc_info['num_positions'],
            'total_premium_received': total_premium_received,
        }

