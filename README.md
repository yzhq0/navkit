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
import pandas as pd
from navkit import NavAnalyzer

an = NavAnalyzer(
    nav, benchmark=benchmark,
    risk_free_annual=0.015,
    trading_days_per_year=250,  # 固定 250；设为 None 才联网估算
    weeks_per_year=50           # 中国口径
)
an.fit()
print(an.summary_)
ax = an.plot_nav_vs_benchmark()
```

## 新增：超额分析
- 统计：**累计超额**、**年化超额**、**超额波动率(=TE)**、**超额回撤**/最长水下期；
- 画图：`plot_nav_vs_benchmark()` 同时绘制超额曲线；新增 `plot_excess()` 与 `plot_excess_drawdown()`。
