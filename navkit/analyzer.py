
from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_CANDIDATES_FOR_CN = [
    "Microsoft YaHei",
    "SimHei",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
]
_FONT_INITIALIZED = False

def _init_chinese_font() -> None:
    """
    尝试设置可显示中文的字体，避免图例/标题出现乱码。
    仅执行一次；未找到候选字体时保持默认字体。
    """
    global _FONT_INITIALIZED
    if _FONT_INITIALIZED:
        return
    for fam in FONT_CANDIDATES_FOR_CN:
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_INITIALIZED = True


_init_chinese_font()

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
        self.summary_raw_: Optional[Dict[str, Any]] = None

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

        # 4) 先不算收益，若有基准，先进行“价格级别”的统一对齐，再计算收益（避免跨市场错配）
        if self.bm_raw is not None:
            self._prepare_benchmark(freq_info)
        else:
            # 无基准：按当前价格直接转为收益
            self.ret_ = self._to_simple_returns(self.nav_)

        # 5) 指标计算
        metrics = self._compute_metrics(self.ret_, freq_info, label_prefix="")
        if self.bm_ret_ is not None:
            bm_metrics = self._compute_metrics(self.bm_ret_, self.freq_info_bm or freq_info, label_prefix="bm_")
            active_metrics = self._compute_active_metrics(self.ret_, self.bm_ret_, freq_info)
            metrics.update(bm_metrics)
            metrics.update(active_metrics)

        self.freq_info = freq_info
        stats_start = self.ret_.index.min() if self.ret_ is not None and len(self.ret_) else (self.nav_.dropna().index.min() if self.nav_ is not None else None)
        stats_end = self.ret_.index.max() if self.ret_ is not None and len(self.ret_) else (self.nav_.dropna().index.max() if self.nav_ is not None else None)
        # 原始汇总（保留频率细节为 dict，便于调试/兼容）
        self.summary_raw_ = {
            "frequency": freq_info.label,
            "periods_per_year": freq_info.periods_per_year,
            "frequency_detail": freq_info.detail,
            "stats_start": stats_start,
            "stats_end": stats_end,
            **metrics,
        }
        # 展示友好版：扁平化 frequency_detail，避免 DataFrame 中混入 dict
        freq_flat = self._flatten_frequency_detail(freq_info.detail)
        self.summary_ = {
            "frequency": freq_info.label,
            "periods_per_year": freq_info.periods_per_year,
            "stats_start": stats_start,
            "stats_end": stats_end,
            **freq_flat,
            **metrics,
        }
        return self

    # ----------------------------- 数据清理与索引 ----------------------------- #
    @staticmethod
    def _clean_series(s: Optional[pd.Series], name: str) -> Optional[pd.Series]:
        if s is None:
            return None
        s = s.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated()].sort_index()
        if s.name is None or (isinstance(s.name, str) and s.name.strip() == ""):
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

    # ----------------------------- 基准对齐（先价格后收益） ----------------------------- #
    def _prepare_benchmark(self, nav_freq: FrequencyInfo) -> None:
        """
        先对齐价格序列到统一目标频率索引，再计算收益，避免 QDII/跨市场因交易日不同导致的区间错配。
        """
        bm_raw = self.bm_raw.loc[self.nav_raw.index.min(): self.nav_raw.index.max()]

        # 各自先对齐到交易日/日历日
        is_bm_calendar_daily = self._detect_calendar_daily(bm_raw)
        bm_base = self._reindex_calendar_daily(bm_raw) if is_bm_calendar_daily else self._reindex_to_trading_days(bm_raw, self.trading_days_)
        bm_freq = self._classify_frequency_calendar(bm_base) if is_bm_calendar_daily else self._classify_frequency_trading(bm_base, self.trading_days_)

        is_nav_calendar_daily = self._detect_calendar_daily(self.nav_raw)
        nav_base = self._reindex_calendar_daily(self.nav_raw) if is_nav_calendar_daily else self._reindex_to_trading_days(self.nav_raw, self.trading_days_)
        nav_freq0 = self._classify_frequency_calendar(nav_base) if is_nav_calendar_daily else self._classify_frequency_trading(nav_base, self.trading_days_)

        # 更粗的频率为目标频率
        rank = {"monthly": 0, "weekly": 1, "trading_daily": 2, "calendar_daily": 3, "mixed": 1}
        target = bm_freq if rank[bm_freq.label] < rank[nav_freq0.label] else nav_freq0
        self.freq_info_bm = bm_freq

        # 样本内FFILL
        nav_base_ff = self._ffill_until_last(nav_base)
        bm_base_ff = self._ffill_until_last(bm_base)

        # 覆盖期交集
        first = max(nav_base_ff.first_valid_index(), bm_base_ff.first_valid_index())
        last = min(nav_base_ff.last_valid_index(), bm_base_ff.last_valid_index())

        if first is None or last is None or first >= last:
            self.nav_ = self._ffill_until_last(nav_base)
            self.bm_ = self._ffill_until_last(bm_base)
            self.ret_ = self._to_simple_returns(self.nav_)
            self.bm_ret_ = self._to_simple_returns(self.bm_)
            self.active_ret_ = None
            return

        # 目标索引（交易日感知）
        if target.label == "calendar_daily":
            target_index = pd.date_range(first.normalize(), last.normalize(), freq="D")
        elif target.label == "trading_daily":
            td = self._get_trading_days_for_span(first, last)
            target_index = td
        elif target.label in ("weekly", "mixed") and target.resample_rule == "W":
            td = self._get_trading_days_for_span(first, last)
            target_index = self._week_last_trading_days(td)
        elif target.label in ("monthly", "mixed") and target.resample_rule == "ME":
            td = self._get_trading_days_for_span(first, last)
            target_index = self._month_last_trading_days(td)
        else:
            td = self._get_trading_days_for_span(first, last)
            target_index = td

        # 重索引并取双方共同有效日期
        nav_target = nav_base_ff.reindex(target_index)
        bm_target = bm_base_ff.reindex(target_index)
        valid = nav_target.notna() & bm_target.notna()
        nav_target = nav_target[valid]
        bm_target = bm_target[valid]

        # 存储价格与收益
        self.nav_ = nav_target
        self.bm_ = bm_target
        self.ret_ = self._to_simple_returns(self.nav_)
        self.bm_ret_ = self._to_simple_returns(self.bm_)
        self.active_ret_ = (self.ret_ - self.bm_ret_).dropna()

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

        # upside volatility（对称于 downside，使用正的偏离）
        upside = np.clip(ret - mar_per, a_min=0, a_max=None)
        upside_std_ann = upside.std(ddof=1) * np.sqrt(ppy) if n > 1 else np.nan

        wealth = (1.0 + ret).cumprod()
        wealth_idx = ret.index
        wealth = pd.Series(wealth.values, index=wealth_idx)
        dd, dd_start, dd_trough, dd_recover, longest_underwater = self._drawdown_stats(wealth)
        calmar = (ann_return / abs(dd)) if abs(dd) > 1e-12 else np.nan

        period_return = growth - 1.0

        # 分布/尾部
        try:
            skewness = ret.skew()
            kurtosis = ret.kurtosis()
        except Exception:
            skewness = np.nan
            kurtosis = np.nan
        try:
            q5 = float(np.nanpercentile(ret.values, 5))
            var_95 = -q5  # 以正的损失表达
            cvar_95 = -float(np.nanmean(ret[ret <= q5])) if np.isfinite(q5) else np.nan
        except Exception:
            var_95 = np.nan
            cvar_95 = np.nan

        # 一致性
        up_mask = ret > 0
        down_mask = ret < 0
        win_rate = float(up_mask.mean()) if n > 0 else np.nan
        avg_up = float(ret[up_mask].mean()) if up_mask.any() else np.nan
        avg_down = float(ret[down_mask].mean()) if down_mask.any() else np.nan
        gain_loss_ratio = (abs(avg_up) / (abs(avg_down) + 1e-12)) if np.isfinite(avg_up) and np.isfinite(avg_down) else np.nan
        # 最大连续涨/跌
        max_up_streak = 0
        max_down_streak = 0
        cur_up = 0
        cur_down = 0
        for v in ret.values:
            if np.isfinite(v) and v > 0:
                cur_up += 1; max_up_streak = max(max_up_streak, cur_up)
                cur_down = 0
            elif np.isfinite(v) and v < 0:
                cur_down += 1; max_down_streak = max(max_down_streak, cur_down)
                cur_up = 0
            else:
                cur_up = 0; cur_down = 0

        metrics = {
            f"{label_prefix}period_return": period_return,
            f"{label_prefix}annual_return": ann_return,
            f"{label_prefix}vol_annual": vol_ann,
            f"{label_prefix}sharpe": sharpe,
            f"{label_prefix}sortino": sortino,
            f"{label_prefix}downside_risk_annual": downside_std * np.sqrt(ppy) if n > 1 else np.nan,
            f"{label_prefix}upside_vol_annual": upside_std_ann,
            f"{label_prefix}max_drawdown": dd,
            f"{label_prefix}max_drawdown_start": dd_start,
            f"{label_prefix}max_drawdown_trough": dd_trough,
            f"{label_prefix}max_drawdown_recover": dd_recover,
            f"{label_prefix}calmar": calmar,
            f"{label_prefix}longest_underwater_periods": longest_underwater[0],
            f"{label_prefix}longest_underwater_start": longest_underwater[1],
            f"{label_prefix}longest_underwater_end": longest_underwater[2],
            f"{label_prefix}n_periods": n,
            f"{label_prefix}skewness": float(skewness) if np.isfinite(skewness) else np.nan,
            f"{label_prefix}kurtosis": float(kurtosis) if np.isfinite(kurtosis) else np.nan,
            f"{label_prefix}VaR_95": float(var_95) if np.isfinite(var_95) else np.nan,
            f"{label_prefix}CVaR_95": float(cvar_95) if np.isfinite(cvar_95) else np.nan,
            f"{label_prefix}win_rate": float(win_rate) if np.isfinite(win_rate) else np.nan,
            f"{label_prefix}avg_up_return": float(avg_up) if np.isfinite(avg_up) else np.nan,
            f"{label_prefix}avg_down_return": float(avg_down) if np.isfinite(avg_down) else np.nan,
            f"{label_prefix}gain_loss_ratio": float(gain_loss_ratio) if np.isfinite(gain_loss_ratio) else np.nan,
            f"{label_prefix}max_consecutive_up": int(max_up_streak),
            f"{label_prefix}max_consecutive_down": int(max_down_streak),
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
        # 回归统计
        n = len(common)
        resid = r - (alpha_per + beta * b)
        sse = float(np.sum((resid.values if isinstance(resid, pd.Series) else resid) ** 2))
        sst = float(np.sum(((r - r.mean()).values if isinstance(r, pd.Series) else (r - r.mean())) ** 2))
        r2 = 1.0 - (sse / (sst + 1e-12)) if n > 2 else np.nan
        s2 = sse / max(n - 2, 1)
        sxx = float(np.sum(((b - b.mean()).values if isinstance(b, pd.Series) else (b - b.mean())) ** 2))
        se_beta = np.sqrt(s2 / (sxx + 1e-12)) if n > 2 else np.nan
        se_alpha = np.sqrt(s2 * (1.0 / n + (b.mean() ** 2) / (sxx + 1e-12))) if n > 2 else np.nan
        beta_t = float(beta) / (se_beta + 1e-12) if np.isfinite(se_beta) else np.nan
        alpha_t = float(alpha_per) / (se_alpha + 1e-12) if np.isfinite(se_alpha) else np.nan
        alpha_ann = (1.0 + alpha_per) ** ppy - 1.0
        # 捕获比
        up_mask = b > 0
        down_mask = b < 0
        up_capture = (r[up_mask].mean() / (b[up_mask].mean() + 1e-12)) if up_mask.any() else np.nan
        down_capture = (r[down_mask].mean() / (b[down_mask].mean() + 1e-12)) if down_mask.any() else np.nan
        # Hit ratio vs BM（胜率）
        hit_ratio = float((r - b > 0).mean()) if len(r) > 0 else np.nan
        # 超额累计/年化、超额回撤
        if len(active) >= 1:
            active_growth = float((1.0 + active).prod())
            excess_period_return = active_growth - 1.0
            excess_ann = active_growth ** (ppy / len(active)) - 1.0
            wealth_active = (1.0 + active).cumprod()
            wealth_active = pd.Series(wealth_active.values, index=active.index)
            dd, dd_start, dd_trough, dd_recover, longest_underwater = self._drawdown_stats(wealth_active)
        else:
            excess_period_return = np.nan
            excess_ann = np.nan
            dd = np.nan; dd_start = None; dd_trough = None; dd_recover = None; longest_underwater = (0, None, None)
        excess_calmar = (excess_ann / abs(dd)) if isinstance(dd, (int, float, np.floating)) and abs(dd) > 1e-12 else np.nan
        # 主动 downside 风险（以 0 作为 MAR）
        active_down = np.clip(active, a_max=0, a_min=None)
        excess_downside_risk_ann = active_down.std(ddof=1) * np.sqrt(ppy) if len(active) > 1 else np.nan
        # Treynor（年化）：(E[r]-rf)/beta * ppy
        rf_per = (1.0 + self.rf_annual) ** (1.0 / ppy) - 1.0
        treynor = ((r.mean() - rf_per) * ppy) / (float(beta) + 1e-12) if np.isfinite(beta) else np.nan
        # M^2（Modigliani）：Sharpe * σ_bm + R_f
        sharpe_r = ((r - rf_per).mean() / (r.std(ddof=1) + 1e-12)) * np.sqrt(ppy) if len(r) > 1 else np.nan
        bm_vol_ann = b.std(ddof=1) * np.sqrt(ppy) if len(b) > 1 else np.nan
        m2 = sharpe_r * bm_vol_ann + self.rf_annual if np.isfinite(sharpe_r) and np.isfinite(bm_vol_ann) else np.nan
        return {
            "active_te": te,
            "active_ir": ir,
            "beta": float(beta),
            "alpha_annual": float(alpha_ann),
            "alpha_tstat": float(alpha_t) if np.isfinite(alpha_t) else np.nan,
            "beta_tstat": float(beta_t) if np.isfinite(beta_t) else np.nan,
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "up_capture": float(up_capture) if np.isfinite(up_capture) else np.nan,
            "down_capture": float(down_capture) if np.isfinite(down_capture) else np.nan,
            "hit_ratio_vs_bm": float(hit_ratio) if np.isfinite(hit_ratio) else np.nan,
            "excess_win_rate": float(hit_ratio) if np.isfinite(hit_ratio) else np.nan,
            "excess_period_return": float(excess_period_return) if np.isfinite(excess_period_return) else np.nan,
            "excess_annual_return": float(excess_ann) if np.isfinite(excess_ann) else np.nan,
            "excess_max_drawdown": float(dd) if isinstance(dd, (int, float, np.floating)) else np.nan,
            "excess_max_drawdown_start": dd_start,
            "excess_max_drawdown_trough": dd_trough,
            "excess_max_drawdown_recover": dd_recover,
            "excess_longest_underwater_periods": int(longest_underwater[0]) if isinstance(longest_underwater[0], (int, np.integer)) else 0,
            "excess_longest_underwater_start": longest_underwater[1],
            "excess_longest_underwater_end": longest_underwater[2],
            "excess_calmar": float(excess_calmar) if np.isfinite(excess_calmar) else np.nan,
            "excess_downside_risk_annual": float(excess_downside_risk_ann) if np.isfinite(excess_downside_risk_ann) else np.nan,
            "treynor": float(treynor) if np.isfinite(treynor) else np.nan,
            "m2": float(m2) if np.isfinite(m2) else np.nan,
        }

    # ----------------------------- 可视化 ----------------------------- #
    @staticmethod
    def _series_label(s: Optional[pd.Series], fallback: str, prefer_name: bool) -> str:
        if not prefer_name or s is None:
            return fallback
        name = getattr(s, "name", None)
        if name is None:
            return fallback
        name_str = str(name).strip()
        return name_str if name_str else fallback

    def plot_nav_vs_benchmark(self, use_series_name: bool = True) -> plt.Axes:
        if self.nav_ is None:
            raise RuntimeError("请先调用 fit()。")
        base = self.nav_.dropna()
        wealth = base / base.iloc[0]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        nav_label = self._series_label(self.nav_, fallback="Series", prefer_name=use_series_name)
        wealth.plot(ax=ax, linewidth=1.5, label=nav_label)
        if self.bm_ is not None:
            bm = self.bm_.dropna()
            bm_label = self._series_label(self.bm_, fallback="Benchmark", prefer_name=use_series_name)
            (bm / bm.iloc[0]).plot(ax=ax, linewidth=1.0, linestyle="--", label=bm_label)
            if self.active_ret_ is not None and len(self.active_ret_) > 0:
                active_wealth = (1.0 + self.active_ret_).cumprod()
                active_wealth = pd.Series(active_wealth.values, index=self.active_ret_.index)
                active_label = "Excess (Fund-BM)"
                if use_series_name and (nav_label != "Series" or bm_label != "Benchmark"):
                    active_label = f"Excess ({nav_label}-{bm_label})"
                active_wealth.plot(ax=ax, linewidth=1.0, label=active_label)
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

    def plot_monthly_heatmap(self) -> plt.Axes:
        """
        月度收益热力图（基于已对齐后的收益 self.ret_ 聚合为月度乘法收益）。
        """
        if self.ret_ is None:
            raise RuntimeError("请先调用 fit()。")
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

    def plot_excess_monthly_heatmap(self) -> plt.Axes:
        """
        超额（月度）热力图：使用月内超额收益的乘法聚合 ∏(1 + r_active) - 1。
        """
        if self.active_ret_ is None or len(self.active_ret_) == 0:
            raise RuntimeError("需要先提供基准并调用 fit()。")
        mret = (1 + self.active_ret_).resample("ME").prod() - 1.0
        if mret.empty:
            raise ValueError("样本太少，无法绘制超额月度热力图。")
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
        ax.set_title("Monthly Excess Heatmap")
        fig.colorbar(c, ax=ax)
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

    def plot_rolling_beta_alpha(self, window: Optional[int] = None) -> Tuple[plt.Axes, plt.Axes]:
        """
        滚动 Beta 与年化 Alpha。
        """
        if self.bm_ret_ is None or self.ret_ is None:
            raise RuntimeError("需要先提供基准并调用 fit()。")
        if self.freq_info is None:
            raise RuntimeError("请先调用 fit()。")
        ppy = int(round(self.freq_info.periods_per_year)) if window is None else int(window)
        if ppy < 5:
            raise ValueError("窗口过小，建议至少为每年期数。")
        r = self.ret_.copy()
        b = self.bm_ret_.reindex(r.index).dropna()
        r = r.reindex(b.index).dropna()
        mean_r = r.rolling(ppy).mean()
        mean_b = b.rolling(ppy).mean()
        var_b = b.rolling(ppy).var(ddof=1)
        cov_rb = r.rolling(ppy).cov(b)
        beta = cov_rb / (var_b + 1e-12)
        alpha_per = mean_r - beta * mean_b
        alpha_ann = (1.0 + alpha_per).pow(self.freq_info.periods_per_year) - 1.0

        fig1, ax1 = plt.subplots(figsize=(9, 3.5))
        beta.plot(ax=ax1, linewidth=1.2)
        ax1.set_title("Rolling Beta")
        ax1.grid(True, alpha=0.25)

        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        alpha_ann.plot(ax=ax2, linewidth=1.2)
        ax2.set_title("Rolling Alpha (annualized)")
        ax2.grid(True, alpha=0.25)
        return ax1, ax2

    def plot_rolling_te(self, window: Optional[int] = None) -> plt.Axes:
        """
        滚动 TE：std(active_ret) * sqrt(periods_per_year)，默认窗口=每年期数。
        """
        if self.active_ret_ is None or len(self.active_ret_) == 0 or self.freq_info is None:
            raise RuntimeError("需要先提供基准并调用 fit()。")
        ppy = int(round(self.freq_info.periods_per_year)) if window is None else int(window)
        if ppy < 5:
            raise ValueError("窗口过小，建议至少为每年期数。")
        roll_te = self.active_ret_.rolling(ppy).std(ddof=1) * np.sqrt(self.freq_info.periods_per_year)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        roll_te.plot(ax=ax, linewidth=1.2)
        ax.set_title("Rolling Tracking Error")
        ax.grid(True, alpha=0.25)
        return ax

    def plot_active_hist(self, bins: int = 50, density: bool = False) -> plt.Axes:
        """
        主动收益分布直方图。
        """
        if self.active_ret_ is None or len(self.active_ret_) == 0:
            raise RuntimeError("需要先提供基准并调用 fit() 才能绘制超额分布。")
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.hist(self.active_ret_.dropna().values, bins=bins, density=density)
        ax.set_title("Active Return Distribution")
        ax.set_xlabel("Active Return")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, alpha=0.25)
        return ax

    def plot_active_box(self, by: str = "month") -> plt.Axes:
        """
        主动收益分布的时段箱线图。
        by: 'month'（默认），'year'，'weekday'。
        """
        if self.active_ret_ is None or len(self.active_ret_) == 0:
            raise RuntimeError("需要先提供基准并调用 fit()。")
        s = self.active_ret_.dropna()
        if by == "month":
            groups = {m: s[s.index.month == m].values for m in range(1, 13) if (s.index.month == m).any()}
            labels = [f"{m:02d}" for m in groups.keys()]
        elif by == "year":
            years = sorted(set(s.index.year))
            groups = {y: s[s.index.year == y].values for y in years}
            labels = [str(y) for y in groups.keys()]
        elif by == "weekday":
            groups = {d: s[s.index.weekday == d].values for d in range(7) if (s.index.weekday == d).any()}
            labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(groups)]
        else:
            raise ValueError("by 仅支持 'month' | 'year' | 'weekday'")
        data = list(groups.values())
        fig, ax = plt.subplots(figsize=(max(8, 0.6*len(data)+4), 4.0))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"Active Return Boxplot by {by.title()}")
        ax.grid(True, axis="y", alpha=0.25)
        return ax

    # ----------------------------- 导出 ----------------------------- #
    def export_report_excel(self, filepath: str) -> None:
        """
        导出多表 Excel 报告：Summary、ExcessMetrics、MonthlyReturns、MonthlyExcess。
        """
        if self.summary_ is None:
            raise RuntimeError("请先调用 fit()。")
        with pd.ExcelWriter(filepath) as writer:
            self.metrics_dataframe().to_excel(writer, sheet_name="Summary")
            try:
                self.metrics_excess_dataframe().to_excel(writer, sheet_name="ExcessMetrics")
            except Exception:
                pass
            if self.ret_ is not None and len(self.ret_) > 0:
                mret = (1 + self.ret_).resample("ME").prod() - 1.0
                df = mret.to_frame("ret")
                df["year"] = df.index.year
                df["month"] = df.index.month
                pivot = df.pivot(index="year", columns="month", values="ret")
                pivot.to_excel(writer, sheet_name="MonthlyReturns")
            if self.active_ret_ is not None and len(self.active_ret_) > 0:
                mret = (1 + self.active_ret_).resample("ME").prod() - 1.0
                df = mret.to_frame("ret")
                df["year"] = df.index.year
                df["month"] = df.index.month
                pivot = df.pivot(index="year", columns="month", values="ret")
                pivot.to_excel(writer, sheet_name="MonthlyExcess")

    # ----------------------------- 工具方法 ----------------------------- #
    def metrics_dataframe(self, view: str = "full", ordered: bool = True, metrics: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if self.summary_ is None:
            raise RuntimeError("请先调用 fit()。")
        data = self.summary_.copy()
        # 选择列
        if metrics is not None:
            keys = [k for k in metrics if k in data]
        elif view == "core":
            keys = self._core_metric_keys(has_benchmark=(self.bm_ret_ is not None and len(self.bm_ret_) > 0))
            keys = [k for k in keys if k in data]
        else:
            keys = list(data.keys())
        # 排序
        if ordered:
            order_map = {k: i for i, k in enumerate(self._logical_order())}
            keys.sort(key=lambda k: order_map.get(k, len(order_map) + 1))
        return pd.DataFrame({"metric": keys, "value": [data.get(k) for k in keys]}).set_index("metric")

    def metrics_excess_dataframe(self) -> pd.DataFrame:
        """
        返回与超额/主动相关的指标表（便于导出/对比）。
        包含：excess_*、active_te、active_ir、beta、alpha_annual、up_capture、down_capture、r2、alpha_tstat、beta_tstat、treynor、m2。
        """
        if self.summary_ is None:
            raise RuntimeError("请先调用 fit()。")
        keys = [k for k in self.summary_.keys() if k.startswith("excess_")] + [
            "active_te", "active_ir", "beta", "alpha_annual", "up_capture", "down_capture",
            "r2", "alpha_tstat", "beta_tstat", "treynor", "m2"
        ]
        data = {k: self.summary_.get(k, None) for k in keys}
        return pd.DataFrame({"metric": list(data.keys()), "value": list(data.values())}).set_index("metric")

    # ----------------------------- 内部：扁平化/核心/排序 ----------------------------- #
    @staticmethod
    def _flatten_frequency_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not isinstance(detail, dict):
            return out
        if "type" in detail:
            out["freq_type"] = detail.get("type")
        # calendar
        if "weekend_ratio" in detail:
            out["weekend_ratio"] = detail.get("weekend_ratio")
        if "na_ratio" in detail:
            out["na_ratio"] = detail.get("na_ratio")
        # trading
        if "median_obs_per_week" in detail:
            out["median_obs_per_week"] = detail.get("median_obs_per_week")
        if "median_obs_per_month" in detail:
            out["median_obs_per_month"] = detail.get("median_obs_per_month")
        if "trading_days_per_year_used" in detail:
            out["trading_days_per_year_used"] = detail.get("trading_days_per_year_used")
        if "weeks_per_year_used" in detail:
            out["weeks_per_year_used"] = detail.get("weeks_per_year_used")
        return out

    @staticmethod
    def _core_metric_keys(has_benchmark: bool) -> list:
        base = [
            "period_return",
            "annual_return",
            "vol_annual",
            "sharpe",
            "calmar",
            "max_drawdown",
        ]
        if has_benchmark:
            base += [
                "excess_period_return",
                "excess_annual_return",
                "active_te",      # 对应 vol_annual（主动波动=TE）
                "active_ir",      # 对应 sharpe（主动信息比）
                "excess_calmar",
                "excess_max_drawdown",
            ]
        return base

    @staticmethod
    def _logical_order() -> list:
        return [
            # 样本信息
            "frequency", "periods_per_year", "stats_start", "stats_end", "n_periods",
            # 频率细节（已扁平化）
            "freq_type", "weekend_ratio", "na_ratio", "median_obs_per_week", "median_obs_per_month",
            "trading_days_per_year_used", "weeks_per_year_used",
            # 收益水平
            "period_return", "annual_return",
            # 风险水平
            "vol_annual", "downside_risk_annual",
            # 风险调整
            "sharpe", "sortino", "calmar",
            # 回撤
            "max_drawdown", "max_drawdown_start", "max_drawdown_trough", "max_drawdown_recover",
            "longest_underwater_periods", "longest_underwater_start", "longest_underwater_end",
            # 基准对应（bm_*）
            "bm_period_return", "bm_annual_return", "bm_vol_annual", "bm_sharpe", "bm_sortino", "bm_calmar",
            # 主动/超额
            "excess_period_return", "excess_annual_return", "active_te", "active_ir", "excess_calmar",
            "excess_max_drawdown", "excess_max_drawdown_start", "excess_max_drawdown_trough", "excess_max_drawdown_recover",
            "excess_longest_underwater_periods", "excess_longest_underwater_start", "excess_longest_underwater_end",
            "beta", "alpha_annual", "up_capture", "down_capture",
        ]

    # ----------------------------- 导出（视图选择） ----------------------------- #
    def export_report_excel(
        self,
        filepath: str,
        view: str = "full",
        ordered: bool = True,
        metrics: Optional[Iterable[str]] = None,
        chinese_label: bool = False,
        formatted: bool = False,
    ) -> None:
        """
        导出多表 Excel 报告：Summary、ExcessMetrics、MonthlyReturns、MonthlyExcess。
        支持选择精简视图（core）。
        """
        if self.summary_ is None:
            raise RuntimeError("请先调用 fit()。")
        with pd.ExcelWriter(filepath) as writer:
            df_summary = self.metrics_dataframe(view=view, ordered=ordered, metrics=metrics)
            if chinese_label or formatted:
                df_summary = self._format_metrics_dataframe(df_summary, chinese_label=chinese_label, formatted=formatted)
            df_summary.to_excel(writer, sheet_name="Summary")
            try:
                df_excess = self.metrics_excess_dataframe()
                if chinese_label or formatted:
                    df_excess = self._format_metrics_dataframe(df_excess, chinese_label=chinese_label, formatted=formatted)
                df_excess.to_excel(writer, sheet_name="ExcessMetrics")
            except Exception:
                pass
            if self.ret_ is not None and len(self.ret_) > 0:
                mret = (1 + self.ret_).resample("ME").prod() - 1.0
                df = mret.to_frame("ret")
                df["year"] = df.index.year
                df["month"] = df.index.month
                pivot = df.pivot(index="year", columns="month", values="ret")
                pivot.to_excel(writer, sheet_name="MonthlyReturns")
            if self.active_ret_ is not None and len(self.active_ret_) > 0:
                mret = (1 + self.active_ret_).resample("ME").prod() - 1.0
                df = mret.to_frame("ret")
                df["year"] = df.index.year
                df["month"] = df.index.month
                pivot = df.pivot(index="year", columns="month", values="ret")
                pivot.to_excel(writer, sheet_name="MonthlyExcess")

    # -------- 输出格式与中文列名 -------- #
    def _format_metrics_dataframe(self, df: pd.DataFrame, chinese_label: bool, formatted: bool) -> pd.DataFrame:
        mapping = self._metric_catalog_cn()
        out = df.copy()
        # 添加 label 列，并按需格式化 value
        labels = []
        values = []
        for k, row in out.iterrows():
            meta = mapping.get(k, {"label": k, "fmt": "raw"})
            label = meta.get("label", k)
            val = row["value"]
            if formatted:
                val = self._format_value(val, meta.get("fmt", "raw"))
            labels.append(label)
            values.append(val)
        out["value"] = values
        if chinese_label:
            out.insert(0, "label", labels)
        return out

    @staticmethod
    def _format_value(val: Any, fmt: str) -> Any:
        if val is None or (isinstance(val, float) and (not np.isfinite(val))):
            return val
        if fmt == "pct":
            return f"{val*100:.2f}%"
        if fmt == "pct1":
            return f"{val*100:.1f}%"
        if fmt == "ratio":
            return f"{val:.3f}"
        if fmt == "number":
            return f"{val:.4f}"
        if fmt == "int":
            try:
                return int(val)
            except Exception:
                return val
        if fmt == "date":
            try:
                return pd.Timestamp(val).strftime("%Y-%m-%d") if val is not None else None
            except Exception:
                return val
        return val

    @staticmethod
    def _metric_catalog_cn() -> Dict[str, Dict[str, str]]:
        # 仅列出常见指标；未列出的按原 key 与原值返回
        return {
            # 样本
            "frequency": {"label": "频率", "fmt": "raw"},
            "periods_per_year": {"label": "年化期数", "fmt": "int"},
            "stats_start": {"label": "起始日期", "fmt": "date"},
            "stats_end": {"label": "结束日期", "fmt": "date"},
            "n_periods": {"label": "样本期数", "fmt": "int"},
            # 收益
            "period_return": {"label": "区间收益", "fmt": "pct"},
            "annual_return": {"label": "年化收益", "fmt": "pct"},
            # 风险
            "vol_annual": {"label": "年化波动", "fmt": "pct"},
            "downside_risk_annual": {"label": "年化下行风险", "fmt": "pct"},
            "upside_vol_annual": {"label": "年化上行波动", "fmt": "pct"},
            # 风险调整
            "sharpe": {"label": "夏普比", "fmt": "number"},
            "sortino": {"label": "索提诺比", "fmt": "number"},
            "calmar": {"label": "卡玛比", "fmt": "number"},
            # 回撤
            "max_drawdown": {"label": "最大回撤", "fmt": "pct"},
            "max_drawdown_start": {"label": "回撤开始", "fmt": "date"},
            "max_drawdown_trough": {"label": "回撤谷底", "fmt": "date"},
            "max_drawdown_recover": {"label": "回撤修复", "fmt": "date"},
            # 分布/尾部
            "skewness": {"label": "偏度", "fmt": "number"},
            "kurtosis": {"label": "峰度", "fmt": "number"},
            "VaR_95": {"label": "VaR(95%)", "fmt": "pct"},
            "CVaR_95": {"label": "CVaR(95%)", "fmt": "pct"},
            # 一致性
            "win_rate": {"label": "胜率", "fmt": "pct"},
            "avg_up_return": {"label": "平均上涨收益", "fmt": "pct"},
            "avg_down_return": {"label": "平均下跌收益", "fmt": "pct"},
            "gain_loss_ratio": {"label": "盈亏比", "fmt": "number"},
            "max_consecutive_up": {"label": "最大连涨期数", "fmt": "int"},
            "max_consecutive_down": {"label": "最大连跌期数", "fmt": "int"},
            # 基准
            "beta": {"label": "Beta", "fmt": "number"},
            "alpha_annual": {"label": "Alpha(年化)", "fmt": "pct"},
            "alpha_tstat": {"label": "Alpha t 统计", "fmt": "number"},
            "beta_tstat": {"label": "Beta t 统计", "fmt": "number"},
            "r2": {"label": "拟合优度 R^2", "fmt": "number"},
            "treynor": {"label": "特雷诺比率", "fmt": "number"},
            "m2": {"label": "M平方", "fmt": "pct"},
            "up_capture": {"label": "上行捕获", "fmt": "ratio"},
            "down_capture": {"label": "下行捕获", "fmt": "ratio"},
            # 超额
            "excess_period_return": {"label": "超额区间收益", "fmt": "pct"},
            "excess_annual_return": {"label": "超额年化收益", "fmt": "pct"},
            "active_te": {"label": "跟踪误差(年化)", "fmt": "pct"},
            "active_ir": {"label": "信息比率", "fmt": "number"},
            "excess_calmar": {"label": "超额卡玛比", "fmt": "number"},
            "excess_max_drawdown": {"label": "超额最大回撤", "fmt": "pct"},
            "excess_max_drawdown_start": {"label": "超额回撤开始", "fmt": "date"},
            "excess_max_drawdown_trough": {"label": "超额回撤谷底", "fmt": "date"},
            "excess_max_drawdown_recover": {"label": "超额回撤修复", "fmt": "date"},
            "excess_longest_underwater_periods": {"label": "超额最长回撤期数", "fmt": "int"},
            "excess_longest_underwater_start": {"label": "超额最长回撤开始", "fmt": "date"},
            "excess_longest_underwater_end": {"label": "超额最长回撤结束", "fmt": "date"},
            "excess_downside_risk_annual": {"label": "超额下行风险(年化)", "fmt": "pct"},
            "hit_ratio_vs_bm": {"label": "战胜基准胜率", "fmt": "pct"},
            "excess_win_rate": {"label": "超额胜率", "fmt": "pct"},
        }
