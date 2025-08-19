
from __future__ import annotations

from typing import Dict, Optional, Iterable, Tuple
import pandas as pd

from .analyzer import NavAnalyzer

class NavBatch:
    """
    负责批量处理多个净值序列：
    - 支持为所有序列共用一个基准，或为每个序列单独提供一个基准（以 dict 传入）。
    - 共享公共参数（如 trading_days_per_year / weeks_per_year 等）。
    - 输出各个序列的 summary 聚合表。
    """
    def __init__(
        self,
        nav_map: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None,
        benchmark_map: Optional[Dict[str, pd.Series]] = None,
        **analyzer_kwargs
    ) -> None:
        if benchmark is not None and benchmark_map is not None:
            raise ValueError("benchmark 与 benchmark_map 只能二选一。")
        self.nav_map = {k: v for k, v in nav_map.items()}
        self.benchmark = benchmark
        self.benchmark_map = benchmark_map
        self.analyzer_kwargs = analyzer_kwargs
        self.analyzers: Dict[str, NavAnalyzer] = {}

    def fit_all(self) -> "NavBatch":
        self.analyzers = {}
        for name, series in self.nav_map.items():
            bm = self.benchmark_map.get(name) if self.benchmark_map is not None else self.benchmark
            an = NavAnalyzer(series, benchmark=bm, **self.analyzer_kwargs)
            an.fit()
            self.analyzers[name] = an
        return self

    def summaries_dataframe(self, metrics: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        将各对象的 summary_ 萃取为 DataFrame（index=名称）。
        如果指定 metrics，则只导出所选指标列。
        """
        rows = {}
        for name, an in self.analyzers.items():
            summ = an.summary_ if an.summary_ is not None else {}
            rows[name] = summ if metrics is None else {k: summ.get(k) for k in metrics}
        return pd.DataFrame.from_dict(rows, orient="index")

    def summaries_to_csv(self, filepath: str) -> None:
        df = self.summaries_dataframe()
        df.to_csv(filepath)

    def summaries_to_excel(self, filepath: str) -> None:
        df = self.summaries_dataframe()
        with pd.ExcelWriter(filepath) as writer:
            df.to_excel(writer, sheet_name="Summaries")

def analyze_many(
    nav_map: Dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    benchmark_map: Optional[Dict[str, pd.Series]] = None,
    **analyzer_kwargs
) -> Tuple[NavBatch, pd.DataFrame]:
    """
    便捷函数：一步完成批量拟合，并返回 (NavBatch对象, 汇总DataFrame)。
    """
    batch = NavBatch(nav_map, benchmark=benchmark, benchmark_map=benchmark_map, **analyzer_kwargs).fit_all()
    df = batch.summaries_dataframe()
    return batch, df
