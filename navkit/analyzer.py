
from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger("NavAnalyzer")
logger.setLevel(logging.INFO)

DEFAULT_TRADING_DAYS_PER_YEAR = 250  # 中国更贴近 250
DEFAULT_WEEKS_PER_YEAR = 50          # 春节与国庆整周假期，约 50 周

@dataclass
class FrequencyInfo:
    label: str  # 'calendar_daily' | 'trading_daily' | 'weekly' | 'monthly' | 'mixed'
    periods_per_year: float
    resample_rule: Optional[str]  # None for calendar_daily/trading_daily（保留字段，不再用 pandas.resample 进行周/月）
    detail: Dict[str, Any]


class NavAnalyzer:
    """
    通用 NAV/组合表现分析器：交易日感知的频率识别、重采样、年化与主动指标。
    """
    def __init__(
        self,
        nav: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trading_days: Optional[pd.DatetimeIndex] = None,
        risk_free_annual: float = 0.015,
        sortino_mar: Optional[float] = None,
        calendar_code: str = "XSHG",
        trading_days_per_year: Optional[int] = DEFAULT_TRADING_DAYS_PER_YEAR,
        weeks_per_year: int = DEFAULT_WEEKS_PER_YEAR,
    ) -> None:
        self.nav_raw = self._clean_series(nav, name="nav")
        self.bm_raw = self._clean_series(benchmark, name="benchmark") if benchmark is not None else None
        self.calendar_code = calendar_code
        self.trading_days_input = trading_days
        self.rf_annual = float(risk_free_annual)
        self.sortino_mar_annual = self.rf_annual if sortino_mar is None else float(sortino_mar)
        self.trading_days_per_year_param = trading_days_per_year  # 优先使用该值；为 None 时才估算
        self.weeks_per_year = int(weeks_per_year)

        self.freq_info: Optional[FrequencyInfo] = None
        self.freq_info_bm: Optional[FrequencyInfo] = None
        self.trading_days_: Optional[pd.DatetimeIndex] = None
        self.nav_: Optional[pd.Series] = None
        self.ret_: Optional[pd.Series] = None
        self.bm_: Optional[pd.Series] = None
        self.bm_ret_: Optional[pd.Series] = None
        self.active_ret_: Optional[pd.Series] = None
        self.summary_: Optional[Dict[str, Any]] = None

    # ----------------------------- 公共主流程 ----------------------------- #
    def fit(self) -> "NavAnalyzer":
        if len(self.nav_raw) < 3:
            raise ValueError("nav 数据太少（<3条）。")

        # 1) 交易日生成 / 识别日历日全样本
        self.trading_days_ = self._get_trading_days_for_span(self.nav_raw.index.min(), self.nav_raw.index.max())
        is_calendar_daily = self._detect_calendar_daily(self.nav_raw)

        # 2) 频率识别（先对齐，再判定）
        if is_calendar_daily:
            nav_aligned = self._reindex_calendar_daily(self.nav_raw)
            freq_info = self._classify_frequency_calendar(nav_aligned)
        else:
            nav_aligned = self._reindex_to_trading_days(self.nav_raw, self.trading_days_)
            freq_info = self._classify_frequency_trading(nav_aligned, self.trading_days_)

        # 3) 按最细可用频率重采样 + 填充（不越界）
        nav_resampled = self._resample_by_frequency(nav_aligned, freq_info, self.trading_days_)
        self.nav_ = self._ffill_until_last(nav_resampled)

        # 4) 计算收益序列
        self.ret_ = self._to_simple_returns(self.nav_)

        # 5) 基准处理与对齐
        if self.bm_raw is not None:
            self._prepare_benchmark(freq_info)

        # 6) 指标计算
        metrics = self._compute_metrics(self.ret_, freq_info, label_prefix="")
        if self.bm_ret_ is not None:
            bm_metrics = self._compute_metrics(self.bm_ret_, self.freq_info_bm or freq_info, label_prefix="bm_")
            active_metrics = self._compute_active_metrics(self.ret_, self.bm_ret_, freq_info)
            metrics.update(bm_metrics)
            metrics.update(active_metrics)

        self.freq_info = freq_info
        self.summary_ = {
            "frequency": freq_info.label,
            "periods_per_year": freq_info.periods_per_year,
            "frequency_detail": freq_info.detail,
            **metrics,
        }
        return self

    # ----------------------------- 数据清理与索引 ----------------------------- #
    @staticmethod
    def _clean_series(s: Optional[pd.Series], name: str) -> Optional[pd.Series]:
        if s is None:
            return None
        if not isinstance(s.index, pd.DatetimeIndex):
            s = s.copy()
            s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated()].sort_index()
        s.name = name
        return s.astype(float)

    def _get_trading_days_for_span(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        if self.trading_days_input is not None:
            td = self.trading_days_input
            return td[(td >= start.normalize()) & (td <= end.normalize())]
        # 可选依赖：pandas_market_calendars
        try:
            import pandas_market_calendars as mcal  # type: ignore

            cal = mcal.get_calendar(self.calendar_code)
            sched = cal.schedule(start_date=start.date(), end_date=end.date())
            sessions = pd.DatetimeIndex(sched.index.tz_localize(None))
            return sessions
        except Exception as e:
            logger.info("pandas_market_calendars 不可用或失败：%s", e)
        # 可选依赖：exchange_calendars
        try:
            import exchange_calendars as xcals  # type: ignore

            cal = xcals.get_calendar(self.calendar_code)
            sessions = cal.sessions_in_range(pd.Timestamp(start.date(), tz=None), pd.Timestamp(end.date(), tz=None))
            return pd.DatetimeIndex(sessions.tz_localize(None))
        except Exception as e:
            logger.info("exchange_calendars 不可用或失败：%s", e)
        warnings.warn("未能加载交易日历，已回退到去除周末的工作日（可能与上交所实际交易日略有出入）。")
        return pd.bdate_range(start=start.normalize(), end=end.normalize(), freq="C")

    @staticmethod
    def _detect_calendar_daily(nav: pd.Series) -> bool:
        idx = nav.index
        if len(idx) < 60:
            return False
        is_weekend = idx.weekday >= 5
        weekend_ratio = is_weekend.mean()
        deltas = idx.to_series().diff().dropna().dt.days.values
        one_day_share = (deltas == 1).mean() if len(deltas) else 0
        return (weekend_ratio >= 0.15) and (one_day_share >= 0.7)

    @staticmethod
    def _reindex_calendar_daily(nav: pd.Series) -> pd.Series:
        full = pd.date_range(nav.index.min().normalize(), nav.index.max().normalize(), freq="D")
        return nav.reindex(full)

    @staticmethod
    def _reindex_to_trading_days(nav: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
        span = trading_days[(trading_days >= nav.index.min().normalize()) & (trading_days <= nav.index.max().normalize())]
        return nav.reindex(span)

    # ----------------------------- 年化期数选择 ----------------------------- #
    def _get_trading_days_per_year(self) -> int:
        if self.trading_days_per_year_param is not None:
            return int(self.trading_days_per_year_param)
        return self._estimate_trading_days_per_year()

    def _estimate_trading_days_per_year(self, ref_years: int = 5) -> int:
        end_year = self.nav_raw.index.max().year
        start_year = max(end_year - ref_years, 2005)
        start = pd.Timestamp(f"{start_year}-01-01")
        end = pd.Timestamp(f"{end_year-1}-12-31")
        if start > end:
            return DEFAULT_TRADING_DAYS_PER_YEAR

        td_full = None
        try:
            import pandas_market_calendars as mcal  # type: ignore
            cal = mcal.get_calendar(self.calendar_code)
            sched = cal.schedule(start_date=start.date(), end_date=end.date())
            td_full = pd.DatetimeIndex(sched.index.tz_localize(None))
        except Exception:
            try:
                import exchange_calendars as xcals  # type: ignore
                cal = xcals.get_calendar(self.calendar_code)
                sessions = cal.sessions_in_range(pd.Timestamp(start.date(), tz=None), pd.Timestamp(end.date(), tz=None))
                td_full = pd.DatetimeIndex(sessions.tz_localize(None))
            except Exception:
                td_full = None

        if td_full is None or len(td_full) == 0:
            return DEFAULT_TRADING_DAYS_PER_YEAR

        df = pd.Series(1, index=td_full)
        per_year = df.resample("YE").sum()
        if per_year.empty:
            return DEFAULT_TRADING_DAYS_PER_YEAR
        return int(round(float(per_year.mean())))

    # ----------------------------- 频率识别 ----------------------------- #
    @staticmethod
    def _iso_year_week(index: pd.DatetimeIndex) -> pd.MultiIndex:
        iso = index.isocalendar()
        return pd.MultiIndex.from_arrays([iso["year"].values, iso["week"].values], names=["iso_year", "iso_week"])

    @staticmethod
    def _year_month(index: pd.DatetimeIndex) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([index.year, index.month], names=["year", "month"])

    def _classify_frequency_calendar(self, s: pd.Series) -> FrequencyInfo:
        ppy = 365.25
        detail = {
            "type": "calendar",
            "weekend_ratio": (s.index.weekday >= 5).mean() if len(s) else np.nan,
            "na_ratio": s.isna().mean() if len(s) else np.nan,
        }
        return FrequencyInfo("calendar_daily", ppy, resample_rule=None, detail=detail)

    def _classify_frequency_trading(self, s: pd.Series, trading_days: pd.DatetimeIndex) -> FrequencyInfo:
        ppy_td = self._get_trading_days_per_year()

        if len(s) > 0:
            w_groups = self._iso_year_week(s.index)
            weeks = s.groupby(w_groups).apply(lambda x: x.notna().sum())
            m_groups = self._year_month(s.index)
            months = s.groupby(m_groups).apply(lambda x: x.notna().sum())
            med_w = float(weeks.median()) if len(weeks) else 0.0
            med_m = float(months.median()) if len(months) else 0.0
        else:
            med_w = med_m = 0.0

        if med_w >= 4:
            label = "trading_daily"
            resample_rule = None
            ppy = float(ppy_td)
        elif 2.5 <= med_m <= 6:
            label = "weekly"
            resample_rule = "W"
            ppy = float(self.weeks_per_year)
        elif med_m <= 1.5:
            label = "monthly"
            resample_rule = "ME"
            ppy = 12.0
        else:
            label = "mixed"
            if med_m >= 2:
                resample_rule = "W"
                ppy = float(self.weeks_per_year)
            else:
                resample_rule = "ME"
                ppy = 12.0
        detail = {
            "type": "trading",
            "median_obs_per_week": med_w,
            "median_obs_per_month": med_m,
            "trading_days_per_year_used": ppy_td,
            "weeks_per_year_used": self.weeks_per_year,
        }
        return FrequencyInfo(label, ppy, resample_rule=resample_rule, detail=detail)

    # ----------------------------- 交易日推导的周末/月末索引 ----------------------------- #
    @staticmethod
    def _week_last_trading_days(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if len(trading_days) == 0:
            return trading_days
        iso = trading_days.isocalendar()
        df = pd.DataFrame({"date": trading_days, "iso_year": iso["year"].values, "iso_week": iso["week"].values})
        last = df.groupby(["iso_year", "iso_week"])["date"].max().sort_values()
        return pd.DatetimeIndex(last.values)

    @staticmethod
    def _month_last_trading_days(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if len(trading_days) == 0:
            return trading_days
        df = pd.DataFrame({"date": trading_days, "year": trading_days.year, "month": trading_days.month})
        last = df.groupby(["year", "month"])["date"].max().sort_values()
        return pd.DatetimeIndex(last.values)

    # ----------------------------- 重采样 ----------------------------- #
    def _resample_by_frequency(self, s: pd.Series, freq_info: FrequencyInfo, trading_days: pd.DatetimeIndex) -> pd.Series:
        label = freq_info.label
        if label in ("calendar_daily", "trading_daily"):
            return s
        s_ff = self._ffill_until_last(s)
        if label in ("weekly", "mixed") and freq_info.resample_rule == "W":
            week_ends = self._week_last_trading_days(trading_days)
            return s_ff.reindex(week_ends)
        if label in ("monthly", "mixed") and freq_info.resample_rule == "ME":
            month_ends = self._month_last_trading_days(trading_days)
            return s_ff.reindex(month_ends)
        return s_ff

    # ----------------------------- 缺失值与收益 ----------------------------- #
    @staticmethod
    def _ffill_until_last(s: pd.Series) -> pd.Series:
        if len(s) == 0:
            return s
        last_idx = s.last_valid_index()
        if last_idx is None:
            return s
        out = s.ffill()
        mask = out.index > last_idx
        if mask.any():
            out.loc[mask] = np.nan
        first_idx = s.first_valid_index()
        if first_idx is not None:
            out.loc[out.index < first_idx] = np.nan
        return out

    @staticmethod
    def _to_simple_returns(price: pd.Series) -> pd.Series:
        return price.pct_change().dropna()

    @staticmethod
    def _ann_factor(freq_info: FrequencyInfo) -> float:
        return float(freq_info.periods_per_year)

    # ----------------------------- 基准对齐 ----------------------------- #
    def _prepare_benchmark(self, nav_freq: FrequencyInfo) -> None:
        bm = self.bm_raw.loc[self.nav_raw.index.min() : self.nav_raw.index.max()]

        is_bm_calendar_daily = self._detect_calendar_daily(bm)
        if is_bm_calendar_daily:
            bm_aligned = self._reindex_calendar_daily(bm)
            bm_freq = self._classify_frequency_calendar(bm_aligned)
        else:
            bm_aligned = self._reindex_to_trading_days(bm, self.trading_days_)
            bm_freq = self._classify_frequency_trading(bm_aligned, self.trading_days_)

        rank = {"monthly": 0, "weekly": 1, "trading_daily": 2, "calendar_daily": 3, "mixed": 1}
        target = bm_freq if rank[bm_freq.label] < rank[nav_freq.label] else nav_freq
        self.freq_info_bm = bm_freq

        if target.label == "calendar_daily":
            nav_target = self.nav_ if self.nav_ is not None else self._reindex_calendar_daily(self.nav_raw)
            bm_target = self._reindex_calendar_daily(bm_aligned)
        elif target.label == "trading_daily":
            nav_td = self.nav_ if self.nav_ is not None else self._reindex_to_trading_days(self.nav_raw, self.trading_days_)
            bm_td = bm_aligned
            nav_target, bm_target = nav_td, bm_td
        else:
            nav_td = self.nav_ if self.nav_ is not None else self._reindex_to_trading_days(self.nav_raw, self.trading_days_)
            bm_td = bm_aligned
            nav_target = self._resample_by_frequency(nav_td, target, self.trading_days_)
            bm_target = self._resample_by_frequency(bm_td, target, self.trading_days_)

        self.nav_ = self._ffill_until_last(nav_target)
        self.bm_ = self._ffill_until_last(bm_target)

        align_index = self.nav_.dropna().index.intersection(self.bm_.dropna().index)
        self.nav_ = self.nav_.reindex(align_index)
        self.bm_ = self.bm_.reindex(align_index)

        self.ret_ = self._to_simple_returns(self.nav_)
        self.bm_ret_ = self._to_simple_returns(self.bm_)
        # 存储超额收益序列
        self.active_ret_ = self.ret_.reindex(self.bm_ret_.index).dropna() - self.bm_ret_.reindex(self.ret_.index).dropna()

    # ----------------------------- 指标计算 ----------------------------- #
    def _compute_metrics(self, ret: pd.Series, freq: FrequencyInfo, label_prefix: str = "") -> Dict[str, Any]:
        ppy = self._ann_factor(freq)
        rf_per = (1.0 + self.rf_annual) ** (1.0 / ppy) - 1.0
        mar_annual = self.sortino_mar_annual
        mar_per = (1.0 + mar_annual) ** (1.0 / ppy) - 1.0

        n = ret.shape[0]
        if n == 0:
            return {}

        growth = (1.0 + ret).prod()
        ann_return = growth ** (ppy / n) - 1.0
        vol_ann = ret.std(ddof=1) * np.sqrt(ppy) if n > 1 else np.nan

        ex_ret = ret - rf_per
        sharpe = ex_ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(ppy) if n > 1 else np.nan

        downside = np.clip(ret - mar_per, a_max=0, a_min=None)
        downside_std = downside.std(ddof=1)
        sortino = ((ret - mar_per).mean() / (downside_std + 1e-12) * np.sqrt(ppy)) if n > 1 else np.nan

        wealth = (1.0 + ret).cumprod()
        wealth_idx = ret.index
        wealth = pd.Series(wealth.values, index=wealth_idx)
        dd, dd_start, dd_trough, dd_recover, longest_underwater = self._drawdown_stats(wealth)
        calmar = (ann_return / abs(dd)) if abs(dd) > 1e-12 else np.nan

        period_return = growth - 1.0

        metrics = {
            f"{label_prefix}period_return": period_return,
            f"{label_prefix}annual_return": ann_return,
            f"{label_prefix}vol_annual": vol_ann,
            f"{label_prefix}sharpe": sharpe,
            f"{label_prefix}sortino": sortino,
            f"{label_prefix}downside_risk_annual": downside_std * np.sqrt(ppy) if n > 1 else np.nan,
            f"{label_prefix}max_drawdown": dd,
            f"{label_prefix}max_drawdown_start": dd_start,
            f"{label_prefix}max_drawdown_trough": dd_trough,
            f"{label_prefix}max_drawdown_recover": dd_recover,
            f"{label_prefix}calmar": calmar,
            f"{label_prefix}longest_underwater_periods": longest_underwater[0],
            f"{label_prefix}longest_underwater_start": longest_underwater[1],
            f"{label_prefix}longest_underwater_end": longest_underwater[2],
            f"{label_prefix}n_periods": n,
        }
        return metrics

    @staticmethod
    def _drawdown_stats(wealth: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp], Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
        if wealth.isna().all():
            return np.nan, None, None, None, (0, None, None)
        x = wealth.dropna()
        if x.empty:
            return np.nan, None, None, None, (0, None, None)
        running_max = x.cummax()
        drawdown = x / running_max - 1.0
        min_dd = drawdown.min()
        trough = drawdown.idxmin()
        pre_slice = x.loc[:trough]
        peak = pre_slice.idxmax()
        after = x.loc[trough:]
        recover = after[after >= running_max.loc[peak]].first_valid_index()

        under = drawdown < 0
        longest_len = 0
        longest_start = None
        longest_end = None
        current_start = None
        for t, flag in under.items():
            if flag and current_start is None:
                current_start = t
            if (not flag or t == under.index[-1]) and current_start is not None:
                end_t = t
                current_len = under.loc[current_start:end_t].sum()
                if current_len > longest_len:
                    longest_len = int(current_len)
                    longest_start = current_start
                    longest_end = end_t
                current_start = None if not flag else current_start
        return float(min_dd), peak, trough, recover, (longest_len, longest_start, longest_end)

    def _compute_active_metrics(self, ret: pd.Series, bm_ret: pd.Series, freq: FrequencyInfo) -> Dict[str, Any]:
        idx = ret.index.intersection(bm_ret.index)
        r = ret.reindex(idx).dropna()
        b = bm_ret.reindex(idx).dropna()
        common = r.index.intersection(b.index)
        r = r.reindex(common)
        b = b.reindex(common)
        if len(common) < 3:
            return {}
        ppy = self._ann_factor(freq)
        active = r - b
        # TE / IR / 回归 Alpha Beta
        te = active.std(ddof=1) * np.sqrt(ppy) if len(active) > 1 else np.nan
        ir = (active.mean() / (active.std(ddof=1) + 1e-12)) * np.sqrt(ppy) if len(active) > 1 else np.nan
        cov = np.cov(b, r, ddof=1)
        var_b = cov[0, 0]
        beta = cov[0, 1] / (var_b + 1e-12)
        alpha_per = r.mean() - beta * b.mean()
        alpha_ann = (1.0 + alpha_per) ** ppy - 1.0
        # 捕获比
        up_mask = b > 0
        down_mask = b < 0
        up_capture = (r[up_mask].mean() / (b[up_mask].mean() + 1e-12)) if up_mask.any() else np.nan
        down_capture = (r[down_mask].mean() / (b[down_mask].mean() + 1e-12)) if down_mask.any() else np.nan
        # 超额累计/年化、超额回撤
        if len(active) >= 1:
            active_growth = float((1.0 + active).prod())
            excess_period_return = active_growth - 1.0
            excess_ann = active_growth ** (ppy / len(active)) - 1.0
            excess_vol_ann = te  # TE 等价于超额波动率
            wealth_active = (1.0 + active).cumprod()
            wealth_active = pd.Series(wealth_active.values, index=active.index)
            dd, dd_start, dd_trough, dd_recover, longest_underwater = self._drawdown_stats(wealth_active)
        else:
            excess_period_return = np.nan
            excess_ann = np.nan
            excess_vol_ann = np.nan
            dd = np.nan; dd_start = None; dd_trough = None; dd_recover = None; longest_underwater = (0, None, None)
        return {
            "active_te": te,
            "active_ir": ir,
            "beta": float(beta),
            "alpha_annual": float(alpha_ann),
            "up_capture": float(up_capture) if np.isfinite(up_capture) else np.nan,
            "down_capture": float(down_capture) if np.isfinite(down_capture) else np.nan,
            "excess_period_return": float(excess_period_return) if np.isfinite(excess_period_return) else np.nan,
            "excess_annual_return": float(excess_ann) if np.isfinite(excess_ann) else np.nan,
            "excess_vol_annual": float(excess_vol_ann) if np.isfinite(excess_vol_ann) else np.nan,
            "excess_max_drawdown": float(dd) if isinstance(dd, (int, float, np.floating)) else np.nan,
            "excess_max_drawdown_start": dd_start,
            "excess_max_drawdown_trough": dd_trough,
            "excess_max_drawdown_recover": dd_recover,
            "excess_longest_underwater_periods": int(longest_underwater[0]) if isinstance(longest_underwater[0], (int, np.integer)) else 0,
            "excess_longest_underwater_start": longest_underwater[1],
            "excess_longest_underwater_end": longest_underwater[2],
        }

    # ----------------------------- 可视化 ----------------------------- #
    def plot_nav_vs_benchmark(self) -> plt.Axes:
        if self.nav_ is None:
            raise RuntimeError("请先调用 fit()。")
        base = self.nav_.dropna()
        wealth = base / base.iloc[0]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        wealth.plot(ax=ax, linewidth=1.5, label="Series")
        if self.bm_ is not None:
            bm = self.bm_.dropna()
            (bm / bm.iloc[0]).plot(ax=ax, linewidth=1.0, linestyle="--", label="Benchmark")
            if self.active_ret_ is not None and len(self.active_ret_) > 0:
                active_wealth = (1.0 + self.active_ret_).cumprod()
                active_wealth = pd.Series(active_wealth.values, index=self.active_ret_.index)
                active_wealth.plot(ax=ax, linewidth=1.0, label="Excess (Fund-BM)")
        ax.set_title("Cumulative Index (normalized)")
        ax.set_ylabel("Wealth Index / Excess Index")
        ax.legend()
        ax.grid(True, alpha=0.25)
        return ax

    def plot_drawdown(self) -> plt.Axes:
        if self.nav_ is None:
            raise RuntimeError("请先调用 fit()。")
        curve = self.nav_.dropna() / self.nav_.dropna().iloc[0]
        wealth = curve.cummax()
        dd = curve / wealth - 1.0
        fig, ax = plt.subplots(figsize=(9, 3.5))
        dd.plot(ax=ax, linewidth=1.0)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.25)
        return ax

    def plot_excess(self) -> plt.Axes:
        """
        绘制累计超额（主动）曲线：wealth_active = Π(1 + r_fund - r_bm)
        """
        if self.active_ret_ is None or len(self.active_ret_) == 0:
            raise RuntimeError("需要先提供基准并调用 fit() 才能绘制超额。")
        wealth_active = (1.0 + self.active_ret_).cumprod()
        wealth_active = pd.Series(wealth_active.values, index=self.active_ret_.index)
        fig, ax = plt.subplots(figsize=(9, 4.0))
        wealth_active.plot(ax=ax, linewidth=1.2)
        ax.set_title("Cumulative Excess (Fund - Benchmark)")
        ax.set_ylabel("Excess Wealth Index")
        ax.grid(True, alpha=0.25)
        return ax

    def plot_excess_drawdown(self) -> plt.Axes:
        """
        绘制超额回撤（基于累计超额曲线）。
        """
        if self.active_ret_ is None or len(self.active_ret_) == 0:
            raise RuntimeError("需要先提供基准并调用 fit() 才能绘制超额回撤。")
        wealth_active = (1.0 + self.active_ret_).cumprod()
        wealth_active = pd.Series(wealth_active.values, index=self.active_ret_.index)
        running_max = wealth_active.cummax()
        dd = wealth_active / running_max - 1.0
        fig, ax = plt.subplots(figsize=(9, 3.5))
        dd.plot(ax=ax, linewidth=1.0)
        ax.set_title("Excess Drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.25)
        return ax

    def plot_rolling(self, window: Optional[int] = None) -> Tuple[plt.Axes, plt.Axes]:
        if self.ret_ is None or self.freq_info is None:
            raise RuntimeError("请先调用 fit()。")
        ppy = int(round(self.freq_info.periods_per_year)) if window is None else int(window)
        if ppy <= 2:
            raise ValueError("窗口太小，无法计算滚动统计。")
        roll_ret = (1 + self.ret_).rolling(ppy).apply(lambda x: x.prod() ** (self.freq_info.periods_per_year / len(x)) - 1.0, raw=False)
        roll_vol = self.ret_.rolling(ppy).std(ddof=1) * np.sqrt(self.freq_info.periods_per_year)
        fig1, ax1 = plt.subplots(figsize=(9, 3.5))
        roll_ret.plot(ax=ax1)
        ax1.set_title("Rolling Annualized Return")
        ax1.grid(True, alpha=0.25)
        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        roll_vol.plot(ax=ax2)
        ax2.set_title("Rolling Annualized Volatility")
        ax2.grid(True, alpha=0.25)
        return ax1, ax2

    def plot_monthly_heatmap(self) -> plt.Axes:
        if self.ret_ is None:
            raise RuntimeError("请先调用 fit()。")
        # 月度聚合对收益采用乘法聚合自然月末即可
        mret = (1 + self.ret_).resample("ME").prod() - 1.0
        if mret.empty:
            raise ValueError("样本太少，无法绘制月度热力图。")
        df = mret.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="ret")
        fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * pivot.shape[0])))
        data = pivot.values
        c = ax.pcolor(data, edgecolors="white", linewidths=0.2)
        ax.set_yticks(np.arange(0.5, data.shape[0] + 0.5))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(0.5, data.shape[1] + 0.5))
        ax.set_xticklabels(["%02d" % m for m in pivot.columns])
        for (i, j), val in np.ndenumerate(data):
            if not np.isnan(val):
                ax.text(j + 0.5, i + 0.5, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)
        ax.set_title("Monthly Return Heatmap")
        fig.colorbar(c, ax=ax)
        return ax

    # ----------------------------- 工具方法 ----------------------------- #
    def metrics_dataframe(self) -> pd.DataFrame:
        if self.summary_ is None:
            raise RuntimeError("请先调用 fit()。")
        return pd.DataFrame({"metric": list(self.summary_.keys()), "value": list(self.summary_.values())}).set_index("metric")
