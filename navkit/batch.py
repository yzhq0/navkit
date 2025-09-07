
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

    def summaries_dataframe(
        self,
        metrics: Optional[Iterable[str]] = None,
        view: str = "full",
        ordered: bool = True,
    ) -> pd.DataFrame:
        """
        将各对象的 summary_ 萃取为 DataFrame（index=名称）。
        - metrics：仅导出所选指标列；
        - view："full" | "core"；
        - ordered：是否按逻辑顺序排列列。
        """
        # 收集每个对象的视图（借用 NavAnalyzer.metrics_dataframe 的列选择逻辑保持一致）
        per_name_frames = {}
        for name, an in self.analyzers.items():
            if an.summary_ is None:
                continue
            df = an.metrics_dataframe(view=view, ordered=False, metrics=metrics)
            per_name_frames[name] = df

        # 统一列集合（保持稳定顺序）
        all_keys = []
        for df in per_name_frames.values():
            for k in df.index.tolist():
                if k not in all_keys:
                    all_keys.append(k)

        # 逻辑排序
        if ordered and len(self.analyzers) > 0:
            # 取第一个 analyzer 的顺序定义
            any_an = next(iter(self.analyzers.values()))
            order_map = {k: i for i, k in enumerate(any_an._logical_order())}
            all_keys.sort(key=lambda k: order_map.get(k, len(order_map) + 1))

        # 组装为宽表
        rows = {}
        for name, df in per_name_frames.items():
            value_map = df["value"].to_dict()
            rows[name] = {k: value_map.get(k, None) for k in all_keys}
        return pd.DataFrame.from_dict(rows, orient="index")[all_keys]

    def summaries_to_csv(self, filepath: str, metrics: Optional[Iterable[str]] = None, view: str = "full", ordered: bool = True) -> None:
        df = self.summaries_dataframe(metrics=metrics, view=view, ordered=ordered)
        df.to_csv(filepath)

    def summaries_to_excel(self, filepath: str, metrics: Optional[Iterable[str]] = None, view: str = "full", ordered: bool = True, chinese_label: bool = False, formatted: bool = False) -> None:
        df = self.summaries_dataframe(metrics=metrics, view=view, ordered=ordered)
        if chinese_label or formatted:
            # 借用任一 analyzer 的格式映射
            any_an = next(iter(self.analyzers.values())) if len(self.analyzers) else None
            if any_an is not None:
                # 将宽表转换为 metric/value 长表再格式化
                fmt_map = any_an._metric_catalog_cn()
                df_fmt = df.copy()
                # 逐列格式化
                for col in df_fmt.columns:
                    meta = fmt_map.get(col, {"label": col, "fmt": "raw"})
                    if formatted:
                        df_fmt[col] = df_fmt[col].apply(lambda v, f=meta.get("fmt", "raw"): any_an._format_value(v, f))
                if chinese_label:
                    rename_map = {col: fmt_map.get(col, {"label": col}).get("label", col) for col in df_fmt.columns}
                    df = df_fmt.rename(columns=rename_map)
                else:
                    df = df_fmt
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
