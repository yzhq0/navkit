# navkit

`navkit` 提供**交易日感知**的净值/组合表现分析：频率识别、周/月末交易日重采样、缺失值正确填充、
年化指标计算（支持设定 `trading_days_per_year` 与 `weeks_per_year`）、超额与主动指标、可视化等。

## 安装
```bash
pip install -e .
# 或使用日历扩展：
pip install -e .[cal]
```

## 快速开始
```python
from navkit import NavAnalyzer

an = NavAnalyzer(
    nav, benchmark=benchmark,
    risk_free_annual=0.015,
    trading_days_per_year=250,  # 固定 250；设为 None 才联网估算
    weeks_per_year=50           # 中国口径
)
an.fit()
print(an.summary_)  # 扁平化的 summary（不含 dict），也可查看 an.summary_raw_
ax = an.plot_nav_vs_benchmark()
```

## 主要特性
- 交易日感知的**频率识别**与**重采样**（周/月末=最后一个交易日）；
- 年化/波动/夏普/索提诺/卡玛/回撤/修复天数等；新增：上行波动、偏度、峰度、VaR/CFaR(95%)、胜率、盈亏比、最大连涨/跌；
- 提供基准时的主动指标：TE、IR、Beta/Alpha、上下行捕获、累计超额、年化超额、超额回撤、超额卡玛、超额下行风险、回归R²与t统计、Treynor、M² 等；
- 画图：累计/回撤、月度热力图、超额累计/回撤、**超额月度热力图**、滚动 Beta/Alpha、**滚动TE**、**主动收益箱线图**。

## 指标视图与导出
支持“精简视图”和中文/格式化导出：
```python
# 精简视图（核心指标：收益/波动/夏普/卡玛/最大回撤；含基准则带超额对应）
an.metrics_dataframe(view="core")

# Excel 导出：选择视图、中文列名与格式化
an.export_report_excel("report.xlsx", view="core", chinese_label=True, formatted=True)
```

## 批量处理
```python
from navkit import NavBatch, analyze_many

nav_map = {"A": nav_a, "B": nav_b}
batch, df = analyze_many(nav_map, benchmark=bm, trading_days_per_year=250, weeks_per_year=50)
print(df.head())
# 批量导出（支持中文与格式化）
batch.summaries_to_excel("summaries.xlsx", view="core", chinese_label=True, formatted=True)
```
