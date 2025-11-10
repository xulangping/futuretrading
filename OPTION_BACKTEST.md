# 期权卖出回测系统完整说明

## 📋 系统概述

这是一个基于 Polygon.io API 的**期权卖出（Option Writing）回测系统**，用于模拟期权卖方策略的历史表现。系统支持完整的配置文件管理，可以精细控制筛选条件、止盈止损、仓位管理等参数。

### 核心特性

- ✅ **完整配置文件支持** - YAML 格式，支持70+配置项
- ✅ **多维度筛选** - 权利金、价格、Greeks、IV、成交量等
- ✅ **智能仓位管理** - 固定、风险基础、Kelly 准则
- ✅ **灵活止盈止损** - 时间、收益率、Delta、到期日等
- ✅ **本地数据缓存** - Parquet 格式，减少 API 调用
- ✅ **付费 API 优化** - 支持并发请求，更快回测
- ✅ **详细报告输出** - JSON 格式，包含所有交易明细

---

## 🚀 快速开始

### 1. 准备工作

**配置 API Key**

在项目根目录的 `.env` 文件中添加：
```bash
POLYGON_API_KEY=your_api_key_here
```

**安装依赖**
```bash
pip install pandas pytz requests python-dotenv pandas_market_calendars pyyaml
```

### 2. 配置参数

编辑 `config_option.yaml` 文件，根据您的策略调整参数。主要配置项：

```yaml
backtest:
  initial_cash: 1000000.0    # 初始资金
  hold_days: 5               # 持仓天数
  max_positions: 20          # 最大持仓数

filters:
  premium:
    min: 50000               # 最小权利金
  spot:
    min: 0.20                # 最小期权价格
  dte:
    min: 7                   # 最小到期天数
    max: 180                 # 最大到期天数
  iv_pct:
    min: 20.0                # 最小隐含波动率
    max: 200.0               # 最大隐含波动率

exit_rules:
  profit_target:
    enabled: true
    target_pct: 50.0         # 50%止盈
  stop_loss:
    enabled: true
    loss_pct: 100.0          # 100%止损
```

### 3. 运行回测

**基础用法：**
```bash
python run_option_backtest.py --config config_option.yaml
```

**指定日期范围：**
```bash
python run_option_backtest.py \
  --config config_option.yaml \
  --start-date 2023-03-01 \
  --end-date 2023-12-31
```

### 4. 查看结果

回测完成后会显示报告并生成 JSON 文件：

```
============================================================
期权卖出回测报告
============================================================

=== 账户概况 ===
  初始资金: $1,000,000.00
  最终现金: $1,052,345.00
  总盈亏: $+52,345.00
  收益率: +5.23%

=== 交易统计 ===
  总信号数: 1,500
  卖出期权数: 450
  平仓交易数: 440
  胜率: 78.5%

=== 盈亏分析 ===
  已实现盈亏: $+50,123.00
  平均盈利: $+245.50
  平均亏损: $-380.25
============================================================
```

---

## 📊 完整配置说明

### 回测基础设置

```yaml
backtest:
  initial_cash: 1000000.0      # 初始资金（美元）
  start_date: "2023-03-01"     # 回测开始日期，null = CSV最早
  end_date: "2025-10-31"       # 回测结束日期，null = CSV最晚
  hold_days: 5                 # 持仓天数
  max_positions: 20            # 最大同时持仓数
  signal_delay_minutes: 10     # 信号延迟（分钟）
```

### 交易成本配置

```yaml
costs:
  slippage: 0.01                  # 滑点 1%
  commission_per_contract: 0.65   # 每张合约佣金 $0.65
  min_commission: 1.0             # 最低佣金 $1.00
```

**成本计算：**
- **卖出收取权利金**：`实际权利金 = 市场权利金 × (1 - 滑点) × 合约数 × 100 - 佣金`
- **买入平仓支付**：`实际成本 = 市场权利金 × (1 + 滑点) × 合约数 × 100 + 佣金`

### 筛选条件详解

#### 1. 基础筛选

```yaml
filters:
  # 期权类型
  option_type:
    - "call"        # 看涨期权
    # - "put"       # 看跌期权（取消注释启用）
  
  # 交易方向
  side:
    - "ASK"         # 买方主动（通常更看涨）
    # - "BID"       # 卖方主动
```

#### 2. 金额筛选

```yaml
  # 权利金筛选
  premium:
    min: 50000           # 最小权利金（美元）
    max: 10000000        # 最大权利金
  
  # 期权价格筛选
  spot:
    min: 0.20            # 最小期权价格（美元/合约）
    max: 50.0            # 最大期权价格
  
  # 股票价格筛选
  stock_price:
    min: 5.0             # 最小股票价格
    max: 1000.0          # 最大股票价格
  
  # 行权价筛选
  strike:
    min: 1.0
    max: 5000.0
```

#### 3. 期权特性筛选

```yaml
  # 价外程度（OTM %）
  otm_pct:
    min: 0.0             # 0% = 平价
    max: 50.0            # 50% = 深度价外
  
  # 到期时间（DTE）
  dte:
    min: 7               # 最少7天（避免即将到期）
    max: 180             # 最多180天（避免太远期）
  
  # 隐含波动率（IV）
  iv_pct:
    min: 20.0            # 最小IV 20%
    max: 200.0           # 最大IV 200%（过滤异常）
```

#### 4. 流动性筛选

```yaml
  # 成交量
  volume:
    min: 100             # 最小成交量
    max: null            # 不限制
  
  # 持仓量（Open Interest）
  open_interest:
    min: 50              # 最小持仓量
    max: null
```

#### 5. Greeks 筛选

```yaml
  # Delta（期权价格对股票价格的敏感度）
  delta:
    min: 0.10
    max: 0.90
  
  # Theta（时间衰减）
  theta:
    min: -10.0           # 负值表示每天损失
    max: 0.0
  
  # Gamma（Delta的变化率）
  gamma:
    min: 0.0
    max: 1.0
  
  # Vega（对波动率的敏感度）
  vega:
    min: 0.0
    max: 1.0
```

**Greeks 说明：**
- **Delta**: 0-1之间，越接近1越接近实值（ITM）
- **Theta**: 负值，表示每天期权价值的衰减
- **Gamma**: Delta的变化率，高Gamma意味着Delta变化快
- **Vega**: 波动率每变化1%，期权价格的变化

#### 6. 财报筛选

```yaml
  # 距离财报天数
  earnings:
    min_days_to_earnings: 5    # 距离财报至少5天
    max_days_to_earnings: null # 不限制
```

**建议**：避免在财报前卖出期权，因为财报期间波动率通常很高。

#### 7. 行业和股票筛选

```yaml
  # 行业筛选
  sectors:
    - "Technology"
    - "Healthcare"
    # 更多行业...
  
  # 股票白名单/黑名单
  symbols:
    whitelist: null          # null = 全部允许
      # - "SPY"
      # - "QQQ"
    blacklist:               # 黑名单
      # - "TSLA"
```

### 仓位管理

系统支持三种仓位管理方法：

#### 方法 1: 固定仓位（推荐初学者）

```yaml
position_sizing:
  method: "fixed"
  fixed:
    contracts_per_trade: 1    # 每次固定1张合约
```

#### 方法 2: 基于风险的仓位

```yaml
position_sizing:
  method: "risk_based"
  risk_based:
    risk_per_trade_pct: 2.0   # 每次风险2%
    max_risk_per_trade: 5000  # 最大风险$5000
```

风险计算：假设最大风险 = 权利金 × 2（期权价格翻倍）

#### 方法 3: Kelly 准则

```yaml
position_sizing:
  method: "kelly"
  kelly:
    win_rate: 0.65            # 历史胜率65%
    avg_win: 100              # 平均盈利$100
    avg_loss: 150             # 平均亏损$150
    kelly_fraction: 0.25      # 使用25% Kelly建议
```

Kelly 公式：`f = (p*b - q) / b`
- p = 胜率
- q = 1-p
- b = 平均盈利/平均亏损

### 止盈止损设置

```yaml
exit_rules:
  # 1. 时间止损（持有N天）
  time_based:
    enabled: true
    hold_days: 5
  
  # 2. 收益止盈
  profit_target:
    enabled: true
    target_pct: 50.0          # 盈利50%止盈
  
  # 3. 止损
  stop_loss:
    enabled: true
    loss_pct: 100.0           # 亏损100%止损
  
  # 4. Delta止损
  delta_stop:
    enabled: false
    delta_threshold: 0.70     # Delta > 0.70平仓
  
  # 5. 到期日止损
  expiry_based:
    enabled: true
    days_before_expiry: 1     # 到期前1天平仓
```

### API 配置（付费版）

```yaml
api:
  provider: "polygon"
  plan: "paid"                # 付费版
  
  concurrent:
    enabled: true             # 启用并发
    max_workers: 10           # 最大10个并发线程
  
  rate_limit:
    enabled: false            # 付费版通常无限制
```

**付费版优势**：
- 更高的请求限制
- 支持并发请求（更快回测）
- 更长的历史数据
- 更低的延迟

---

## 📈 CSV 数据说明

### 必需字段

系统从 CSV 读取以下字段（CSV 时间为**中国时间**）：

| 字段 | 说明 | 示例 |
|------|------|------|
| `date` | 日期 | 2023-03-17 |
| `time` | 时间（中国时间） | 03:46:50 |
| `ticker` | 标的股票代码 | SPY |
| `side` | 交易方向 | ASK/BID |
| `strike` | 行权价 | 400.0 |
| `option_type` | 期权类型 | call/put |
| `expiry` | 到期日 | 2023-06-16 |
| `price` | 股票价格 | $145.48 |
| `spot` | 期权价格（权利金） | $0.65 |
| `premium` | 权利金总额 | $592K |
| `volume` | 成交量 | 10,000 |
| `oi` | 持仓量 | 204 |
| `iv_pct` | 隐含波动率 | 57.54% |
| `delta` | Delta | 0.4807 |
| `theta` | Theta | -0.0045 |
| `gamma` | Gamma | 0.1945 |
| `vega` | Vega | 0.0142 |
| `earnings` | 距离财报天数 | 10 |
| `sector` | 行业 | Technology |
| `rule` | 交易规则 | Sweeps Followed By Floor |

### 时间处理

**重要**：CSV 时间为**中国时间**（Asia/Shanghai），系统会自动转换为 ET 时间（America/New_York）并添加信号延迟。

```python
# 时间转换流程
中国时间 → ET时间 → 添加10分钟延迟 → 如果超过16:00则使用15:59:00

# 示例
CSV时间（中国）: 2023-03-17 03:46:50
       ↓ 转换到 ET
ET时间: 2023-03-16 15:46:50
       ↓ 添加 10 分钟延迟
实际交易时间: 2023-03-16 15:56:50
```

### DTE 计算说明

**重要**：CSV 中的 DTE 字段**不被使用**，系统根据交易时间动态计算。

```python
# DTE 计算方式（与股票回测一致）
DTE = (expiry_date - signal_date).days

# 示例
到期日: 2023-06-16
交易时间: 2023-03-16 15:56:50 (ET)
交易日期: 2023-03-16
DTE = (2023-06-16) - (2023-03-16) = 92 天
```

**为什么不用 CSV 中的 DTE？**
- CSV 中的 DTE 是从另一个日期（如 2099-12-31）计算的
- 不是从交易时间到期权到期日的实际天数
- 使用错误的 DTE 会导致筛选条件不准确

**DTE 筛选配置：**
```yaml
filters:
  dte:
    min: 7               # 最少7天到期
    max: 180             # 最多180天到期
```

系统会在加载 CSV 时自动计算每个信号的 DTE，然后应用筛选条件。

---

## 📁 文件结构

```
futuretrading/
├── config_option.yaml                  # 配置文件 ⭐
├── run_option_backtest.py              # 回测运行脚本 ⭐
├── OPTION_BACKTEST.md                  # 本文档 ⭐
│
├── future_v_0_1/
│   ├── database/
│   │   ├── option_cache/               # 期权价格缓存目录 ⭐
│   │   └── merged_strategy_v1_calls_bell_2023M3_2025M10.csv
│   │
│   └── market/
│       └── option_backtest_client.py   # 期权回测客户端 ⭐
│
└── .env                                # API Key配置
```

---

## 🎯 使用案例

### 案例 1: 保守策略（高胜率）

```yaml
backtest:
  hold_days: 3                  # 短期持有

filters:
  otm_pct:
    min: 15.0                   # 远离实值
    max: 30.0
  dte:
    min: 14                     # 较长到期时间
    max: 45
  delta:
    min: 0.15                   # 低Delta（深度价外）
    max: 0.30

exit_rules:
  profit_target:
    target_pct: 30.0            # 30%快速止盈
  stop_loss:
    loss_pct: 50.0              # 50%严格止损
```

**特点**：低风险，高胜率，但单次盈利较小

### 案例 2: 激进策略（高收益）

```yaml
backtest:
  hold_days: 7                  # 较长持有

filters:
  otm_pct:
    min: 0.0                    # 接近平价
    max: 10.0
  delta:
    min: 0.40                   # 高Delta（接近实值）
    max: 0.60

exit_rules:
  profit_target:
    target_pct: 70.0            # 70%止盈
  stop_loss:
    loss_pct: 150.0             # 较宽松止损
```

**特点**：高风险，高收益，胜率较低

### 案例 3: 高IV策略（波动率收益）

```yaml
filters:
  iv_pct:
    min: 50.0                   # 高波动率
    max: 150.0
  
  earnings:
    min_days_to_earnings: null  # 包含财报期

exit_rules:
  time_based:
    hold_days: 2                # 短期持有（利用时间衰减）
  profit_target:
    target_pct: 40.0
```

**特点**：利用高波动率赚取权利金，快进快出

---

## 🔧 高级功能

### 保证金管理

```yaml
advanced:
  margin:
    method: "simple"            # 简化保证金计算
    simple_margin_pct: 20.0     # 标的价值的20%
```

**实际保证金要求**更复杂，取决于：
- 期权类型（call/put）
- 是否有对冲
- 账户类型（现金/保证金账户）
- 券商要求

### 动态滑点模型

```yaml
advanced:
  slippage_model:
    type: "dynamic"             # 动态滑点
    dynamic:
      enabled: true
      base_slippage: 0.005      # 基础滑点0.5%
      volume_factor: 0.0001     # 成交量影响因子
```

动态滑点根据成交量调整：成交量越大，滑点越小。

### 数据验证

```yaml
advanced:
  data_validation:
    enabled: true
    skip_missing_data: true     # 跳过数据缺失的交易
    warn_on_gaps: true          # 数据间隙时警告
```

---

## 📊 回测报告

### JSON 输出格式

```json
{
  "backtest_time": "2025-11-10T12:00:00",
  "config": { ... },
  "report": {
    "账户概况": { ... },
    "交易统计": { ... },
    "盈亏分析": { ... }
  },
  "trades": [
    {
      "type": "SELL_TO_OPEN",
      "option_ticker": "O:SPY230617C00420000",
      "underlying": "SPY",
      "time": "2023-03-17T10:00:00",
      "contracts": 1,
      "premium": 2.48,
      "credit_received": 247.35,
      "strike": 420.0,
      "expiry": "2023-06-17",
      "dte": 92,
      "iv_pct": 25.5,
      "delta": 0.35
    },
    {
      "type": "BUY_TO_CLOSE",
      "option_ticker": "O:SPY230617C00420000",
      "time": "2023-03-22T15:00:00",
      "contracts": 1,
      "premium": 1.52,
      "debit_paid": 153.30,
      "pnl": 94.05,
      "pnl_ratio": 0.38,
      "reason": "到达目标持仓时间"
    }
  ],
  "api_stats": {
    "api_calls": 450,
    "cache_hits": 1200
  }
}
```

---

## ⚠️ 重要提示

### 风险警告

**本系统仅供学习和研究使用，不构成投资建议！**

1. **期权交易高风险** - 可能损失全部本金
2. **回测≠实盘** - 历史表现不代表未来
3. **滑点和成本** - 实际交易成本可能更高
4. **保证金要求** - 实际保证金可能不同
5. **行权风险** - 可能被指派行权

### 数据质量

1. **Polygon.io 限制** - 免费版有请求限制，付费版更好
2. **数据缺失** - 某些期权可能无历史数据
3. **时间对齐** - CSV是中国时间，确保正确转换
4. **流动性** - 低流动性期权滑点可能很大

### 最佳实践

1. **从小范围开始** - 先回测1-3个月数据
2. **验证配置** - 检查筛选条件是否合理
3. **查看日志** - 注意警告和错误信息
4. **对比基准** - 与简单策略对比
5. **样本外测试** - 用不同时期数据验证

---

## 🛠️ 故障排查

### 问题 1: "POLYGON_API_KEY not found"
**解决**：检查 `.env` 文件，确保包含正确的 API Key

### 问题 2: "无符合筛选条件的信号数据"
**解决**：放宽筛选条件，或检查配置文件语法

### 问题 3: "No option data for O:XXX"
**解决**：
- 期权可能在该日期不存在
- 检查期权代码格式
- 确认 Polygon.io 支持该期权

### 问题 4: 回测速度慢
**解决**：
- 启用并发请求（付费版）
- 缩小回测日期范围
- 检查本地缓存是否生效

### 问题 5: 内存占用高
**解决**：
- 减少并发线程数
- 缩小回测范围
- 清理旧的缓存文件

---

## 📞 期权代码格式

Polygon.io 使用以下格式：

```
O:{underlying}{YYMMDD}{C/P}{strike*1000}

组成部分：
- O:           前缀
- underlying   标的代码（如SPY）
- YYMMDD       到期日（如230617 = 2023-06-17）
- C/P          期权类型（C=Call, P=Put）
- strike*1000  行权价×1000，8位数（如00420000 = $420.00）

示例：
- O:SPY230617C00420000  → SPY 2023-06-17 Call $420.00
- O:AAPL240119P00150500 → AAPL 2024-01-19 Put $150.50
- O:QQQ231215C00380000  → QQQ 2023-12-15 Call $380.00
```

---

## 📚 参考资源

- **Polygon.io 文档**: https://polygon.io/docs/options
- **期权基础**: https://www.investopedia.com/options-basics-tutorial-4583012
- **Greeks 详解**: https://www.investopedia.com/trading/using-the-greeks-to-understand-options/
- **Kelly准则**: https://en.wikipedia.org/wiki/Kelly_criterion

---

## 📝 更新日志

- **v1.0** (2025-11-10) - 初始版本
  - 完整配置文件支持
  - 多维度筛选条件
  - 三种仓位管理方法
  - 灵活止盈止损
  - 付费API并发支持
  - 本地数据缓存

---

**创建日期**: 2025-11-10  
**版本**: 1.0  
**状态**: 生产就绪

