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
- `--repair-days 30`：不传 `--start` 时生效，自动向前回看 N 个自然日修补缺口并更新到今天
- `--cache ./data`：缓存目录
- `--data-backend sqlite`：当前仅实现 SQLite（默认）

只拉取最新并自动修补最近缺口（推荐日常用法）：

```bash
stock_screener update
```

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

如需导出股票名称（`name` 列），先同步名称缓存再运行：

```bash
stock_screener sync-names --provider baostock --date 20240131
stock_screener run --date 20240131 --combo and --out results.csv --with-name
```

## 导入通达信（.EBK）

通达信 `.EBK` 是“股票列表”导入/导出格式，本项目可直接导出筛选命中的股票列表：

```bash
stock_screener export-ebk --date 20240131 --combo and --out results.EBK
```

生成的 `.EBK` 按样例口径写入：
- CRLF 文本
- 每行 7 位：`0+6位`(SZ)、`1+6位`(SH)、`2+6位`(BJ)

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
1) `stock_screener update`（自动回看修补最近缺口并更新到今天）
2) `stock_screener run --date {今天} --out results.csv`

Linux（systemd）可直接使用定时器示例（每日更新一次）：
- 复制 `deploy/systemd/stock_screener-update.service` 与 `deploy/systemd/stock_screener-update.timer` 到 `/etc/systemd/system/`
- 启用：`sudo systemctl daemon-reload && sudo systemctl enable --now stock_screener-update.timer`

## Docker 一键启动（后端 + Web）

在本目录下启动（会同时启动后端 API 与 Web UI，并将 `./data` 挂载到容器 `/data` 作为持久化缓存）：

```bash
docker compose up -d --build
```

配置（可选）：复制 `./.env.example` 为 `./.env`，按需设置端口、`TUSHARE_TOKEN`、鉴权等。

访问：
- Web UI：`http://127.0.0.1:5174`
- 后端 API：`http://127.0.0.1:8000/v1`（Web 同源访问用 `/api/v1`）

停止：

```bash
docker compose down
```

如端口冲突，可临时指定：

```bash
SCREENFISH_WEB_PORT=15174 SCREENFISH_BACKEND_PORT=18000 docker compose up -d --build
```

## 后台服务（REST API）

提供一个常驻后台服务，便于你判断“当日日线是否可拉取”、轮询拉取并在数据到位后执行筛选。

启动服务（默认 `127.0.0.1:8000`）：

```bash
stock_screener serve --cache ./data
```

API 文档（OpenAPI）：
- `http://127.0.0.1:8000/docs`

公网部署建议：
- 推荐用 Cloudflare Tunnel 暴露域名（见 `deploy/cloudflared/README.md`），服务只监听本机 `127.0.0.1:8000`。
- 不建议裸奔公网：至少设置 `STOCK_SCREENER_API_KEY`（请求头 `X-API-Key`），并在 Cloudflare 上加 WAF/限流。
- Web 端访问需要配置 CORS：`STOCK_SCREENER_CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com`

可选鉴权（如果设置了环境变量则必须带请求头）：
- `export STOCK_SCREENER_API_KEY="your-key"`
- 请求头：`X-API-Key: your-key`

常用接口（Base URL：`/v1`）：
- `GET /v1/status`：查看本地缓存状态与最新交易日
- `GET /v1/data/availability?provider=baostock&date=YYYYMMDD`：探测当日日线是否已可取
- `POST /v1/update`：触发更新（支持 `start/end` 或自动模式）
- `POST /v1/update/wait`：创建“等待更新”任务并返回 `job_id`（后台轮询直到数据可用/超时）
- `GET /v1/update/wait/{job_id}`：查询任务状态（`running/succeeded/failed/timeout/canceled`）
- `DELETE /v1/update/wait/{job_id}`：取消任务（best-effort）
- `POST /v1/screen`：对指定日期/最新日期执行筛选
- `POST /v1/export/ebk`：导出筛选命中的 `.EBK` 内容（可用于通达信导入）
