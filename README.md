# A股日线自动更新 + 指标筛选（离线工具）

这是一个**仅做数据处理**的离线 CLI 工具：从数据源拉取全市场 A 股日线数据到本地缓存（SQLite），按通达信风格函数计算指标并执行筛选，导出当日命中列表为 CSV/JSON。

> 合规声明：本项目不做任何破解/逆向/绕过授权取数；本工具不提供投资建议/荐股结论。

## 数据源（可选）

目前支持两种数据源：
- `baostock`：默认；无需 token。更新速度取决于网络与全市场股票数量（按股票拉取）。
- `tushare`：TuShare Pro（需要 `TUSHARE_TOKEN`）。

请确保你使用的数据源符合其官方/项目条款。

## 安装

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## 配置 TuShare Token

从环境变量读取：

```bash
export TUSHARE_TOKEN="你的token"
```

无 token 时：
- `update` 会报错并提示配置 token。
- `run` 允许仅用本地缓存离线运行（前提是本地已有数据）。

## 数据缓存（SQLite）

默认缓存目录：`./data`，数据库文件：`daily.sqlite3`。

表约束：
- 日线表 `daily`：`ts_code + trade_date` 唯一去重。
- 更新日志表 `update_log`：按交易日记录是否已成功写入（用于增量更新）。

## 更新数据

```bash
stock_screener update --start 20240101 --end 20240131
```

参数：
- `--provider baostock|tushare`：数据源（默认 `baostock`）
- `--cache ./data`：缓存目录
- `--data-backend sqlite`：当前仅实现 SQLite（默认）

示例（TuShare Pro）：

```bash
export TUSHARE_TOKEN="你的token"
stock_screener update --provider tushare --start 20240101 --end 20240131
```

## 运行筛选并导出

默认组合：`规则1 AND 规则2`。

```bash
stock_screener run --date 20240131 --combo and --out results.csv
stock_screener run --date 20240131 --combo and --out results.json
```

输出字段：
- `trade_date, ts_code, close, amount, ma60, mid_bullbear, j, rules`

其中：
- `ma60`：MA(CLOSE,60)
- `mid_bullbear`：EMA(EMA(CLOSE,10),10)
- `j`：KDJ 的 J 值
- `rules`：命中的规则名（逗号分隔）

## 通达信函数口径说明

按股票分组、按交易日升序逐日计算。

- `MA(X,N)`：简单均线（rolling mean）
- `EMA(X,N)`：指数均线（`alpha=2/(N+1)`，递推形式，与 `pandas.Series.ewm(adjust=False).mean()` 一致）
- `REF(X,N)`：`shift(N)`
- `LLV(X,N)`：rolling min；`HHV(X,N)`：rolling max
- `SMA(X,N,M)`：通达信递推口径：
  - `SMA_t = (M*X_t + (N-M)*SMA_{t-1})/N`
  - 初值取**首个非空** `X`

### KDJ 分母为 0 的处理

KDJ 中 `RSV = (CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100`：
- 当分母为 0（即 `HHV == LLV`）时，本项目固定口径：`RSV = 0`。

该规则在单元测试中有覆盖。

## 规则

### 规则1：执行中期多空线 & MA60

- 执行中期多空线：`EMA(EMA(C,10),10)`
- `MA1 = MA(C,60)`
- 条件：`(执行中期多空线 > MA1) AND (CLOSE > MA1)`

说明：通达信片段里 `A1:=REF(0,1)` 等未参与最终 `XG`，此处不纳入。

### 规则2：KDJ 超跌（J<JT 且昨日也<JT）

- `N=9, M1=3, M2=3, JT=13`
- `RSV = (CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100`（分母 0 时 RSV=0）
- `K = SMA(RSV,M1,1)`
- `D = SMA(K,M2,1)`
- `J = 3*K - 2*D`
- 条件：`(J < JT) AND (REF(J,1) < JT)`

## 自动化建议

可用 `cron`/任务计划每天运行：
1) `stock_screener update --start {最近N天} --end {今天}`
2) `stock_screener run --date {今天} --out results.csv`
