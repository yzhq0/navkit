import pandas as pd
import numpy as np
from navkit import NavAnalyzer, NavBatch, analyze_many

def test_smoke():
    idx = pd.date_range("2024-01-01","2024-03-31",freq="D")
    nav = pd.Series(1.0 + np.cumsum(np.random.normal(0, 0.001, len(idx))), index=idx)
    bm  = pd.Series(1.0 + np.cumsum(np.random.normal(0, 0.0012, len(idx))), index=idx)
    an = NavAnalyzer(nav, benchmark=bm, trading_days_per_year=250, weeks_per_year=50)
    an.fit()
    assert an.summary_ is not None
    # 核心视图应包含核心字段且顺序稳定
    df_core = an.metrics_dataframe(view="core")
    for key in ["period_return","annual_return","vol_annual","sharpe","calmar","max_drawdown"]:
        assert key in df_core.index
    # 若有基准，超额对应应出现
    df_excess = an.metrics_excess_dataframe()
    for key in ["excess_period_return","excess_annual_return","active_te","active_ir","excess_calmar"]:
        assert key in df_excess.index
    # 导出（中文+格式化）
    an.export_report_excel("_tmp_report.xlsx", view="core", chinese_label=True, formatted=True)
    batch, df = analyze_many({"A": nav, "B": nav * 1.01}, benchmark=bm, trading_days_per_year=250, weeks_per_year=50)
    assert "A" in df.index and "B" in df.index
    # 批量导出
    NavBatch({"A": nav, "B": nav}, benchmark=bm, trading_days_per_year=250, weeks_per_year=50).fit_all().summaries_to_excel("_tmp_summaries.xlsx", view="core", chinese_label=True, formatted=True)
