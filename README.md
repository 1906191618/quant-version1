# quant-version1 — S&P 500 Stock Screener (yfinance + sklearn)

一个可随时运行的美股 S&P 500 选股/筛选脚手架（research/demo 用途）：
- 数据：yfinance 日线
- 因子：动量/趋势/波动/回撤/流动性
- 模型：scikit-learn（开源）训练一个简单回归模型来做排序（ranking）
- 约束：剔除低价股（< $5），流动性过滤（20日平均成交额），行业分散（每个 sector 最多 N 只）
- 输出：`outputs/candidates.csv`

## 快速开始
1. 打开仓库 → 点击 **Code** → **Codespaces** → **Create codespace on main**
2. 打开终端运行：
   ```bash
   pip install -r requirements.txt
   python run.py --top 30 --sector-cap 5
   ```
3. 结果会输出到：
   - `outputs/candidates.csv`

## 常用参数
- Top 数量（默认 30）：
  ```bash
  python run.py --top 30
  ```
- 行业分散（每个 sector 最多 5 只）：
  ```bash
  python run.py --sector-cap 5
  ```
- 开启趋势过滤（收盘价 > MA200）：
  ```bash
  python run.py --require-uptrend
  ```

## 文件结构
- `run.py`：入口脚本
- `quant_screener/universe_sp500.py`：获取 S&P 500 ticker + sector（Wikipedia）
- `quant_screener/features.py`：因子计算
- `quant_screener/train_rank.py`：训练模型 + 打分 + 行业分散 + 导出 CSV
- `outputs/`：输出目录
