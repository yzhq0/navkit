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
    batch, df = analyze_many({"A": nav, "B": nav * 1.01}, benchmark=bm, trading_days_per_year=250, weeks_per_year=50)
    assert "A" in df.index and "B" in df.index
