"""
AgentStrategyV2 — 多智能体涌现决策系统 (V2 Architecture)

三阶段流水线:
    QuantPrep (纯计算)  →  Analyst Agent (LLM)  →  RiskMgr Agent (LLM)
      ↓ 多周期因子           ↓ 选股+权重+理由         ↓ 风控微调+终裁
      筛选Top-N候选          结构化JSON输出            平衡型风控

7 项核心设计:
    1. 多资产轮动: 分析全部 N 只标的, 选 Top-N 配置
    2. 信号先行: 量化效率分筛选候选, LLM 做精调而非主决策
    3. 周期调仓: 仅调仓日调 LLM (3个月仅6次 vs V1的180次)
    4. Prompt 重构: 去除保守偏见, 引入定量投资框架, 强制量化推理
    5. 决策记忆: 传递上期持仓、收益、决策理由上下文
    6. 佣金感知下单: 计算手数时扣除佣金
    7. 健壮 fallback: LLM 失败自动退化为纯效率轮动
"""

import json
import asyncio
import time
import os
import re
from typing import TypedDict, Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from langgraph.graph import StateGraph, END

from .base_strategy import BreadFreeStrategy
from ..utils.llm_client import async_hunyuan_chat, parse_llm_response
from ..utils.metrics import calculate_efficiency_metrics, stable_linear_regression
from ..utils.portfolio import normalize_weights
from ..utils.logger import get_logger
from ..data.market_intel import MarketIntel

logger = get_logger(__name__)

_intel = MarketIntel()


def _load_etf_names() -> Dict[str, str]:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("etf_pool", {})
    except Exception:
        return {}


_ETF_NAMES: Dict[str, str] = _load_etf_names()


def _n(symbol: str) -> str:
    name = _ETF_NAMES.get(symbol, "")
    return f"{symbol}-{name}" if name else symbol


# ═══════════════════════════════════════════════════════════════
# 量化引擎：多周期因子计算
# ═══════════════════════════════════════════════════════════════

def compute_advanced_metrics(history: List[float], lookback: int = 20) -> Optional[Dict]:
    """计算多周期高级因子 — 提供 5d/10d/20d 三个维度的动量和效率"""
    if len(history) < lookback + 10:
        return None
    prices = np.array(history[-(lookback + 10):])
    current = prices[-lookback:]
    prev = prices[-(lookback + 5):-5]

    if current[0] <= 0 or prev[0] <= 0:
        return None

    # 多周期动量
    mom_20d = current[-1] / current[0] - 1
    mom_10d = current[-1] / current[-10] - 1 if len(current) >= 10 else mom_20d
    mom_5d = current[-1] / current[-5] - 1 if len(current) >= 5 else mom_20d
    prev_mom = prev[-1] / prev[0] - 1
    mom_accel = mom_20d - prev_mom

    returns = np.diff(current) / current[:-1]
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

    x = np.arange(len(current))
    try:
        slope, intercept, r_value, _, _ = stats.linregress(x, current)
        r2 = r_value ** 2
    except Exception:
        slope, r2 = 0.0, 0.0

    epsilon = 1e-6
    period_vol = volatility * np.sqrt(len(returns)) + epsilon
    efficiency = (mom_20d / period_vol) * r2

    dd_from_high = current[-1] / np.max(current) - 1

    # 动量一致性: 5d/10d/20d 同向为强信号
    mom_alignment = sum(1 for m in [mom_5d, mom_10d, mom_20d] if m > 0) / 3.0

    return {
        "momentum_5d": float(mom_5d),
        "momentum_10d": float(mom_10d),
        "momentum_20d": float(mom_20d),
        "momentum_accel": float(mom_accel),
        "volatility": float(volatility),
        "r2": float(r2),
        "efficiency": float(efficiency),
        "close": float(current[-1]),
        "drawdown_from_high": float(dd_from_high),
        "trend_slope": float(slope),
        "momentum_alignment": float(mom_alignment),
    }


def classify_regime(all_metrics: Dict[str, Dict]) -> str:
    if not all_metrics:
        return "unknown"
    efficiencies = [m["efficiency"] for m in all_metrics.values()]
    momentums = [m["momentum_20d"] for m in all_metrics.values()]
    alignments = [m["momentum_alignment"] for m in all_metrics.values()]
    pct_positive = sum(1 for m in momentums if m > 0.01) / len(momentums)
    avg_eff = np.mean(efficiencies)
    avg_align = np.mean(alignments)

    if pct_positive >= 0.6 and avg_eff > 0.5 and avg_align > 0.6:
        return "strong_bull"
    elif pct_positive >= 0.45 and max(efficiencies) > 1.0:
        return "selective_bull"
    elif pct_positive <= 0.25:
        return "bear"
    else:
        return "choppy"


# ═══════════════════════════════════════════════════════════════
# Prompts — 专业投委会框架
# ═══════════════════════════════════════════════════════════════

ANALYST_PROMPT = """\
你是一位顶级量化策略师,负责从候选池中精选最优ETF/股票构建投资组合。

【决策框架】
1. 效率分(Efficiency)是核心alpha: 衡量"单位风险获取趋势收益的能力"
2. 动量一致性: 5日/10日/20日动量同向(一致性>0.66)=趋势可信
3. 动量加速度>0 = 趋势在加强, 应增加配置
4. R²>0.7 = 趋势线性度高, 可预测性强, 给予更高置信度
5. 持仓延续性: 当前持仓仍在Top候选中 → 优先保留(减少换手摩擦)
6. 敢于集中: 当一只标的效率分远超其他(≥1.5倍)时, 可集中配置至60%

【候选池数据 (近{lookback}日)】
{metrics_table}

【市场情报】
{market_intel}

【市场政权】{regime}
【当前持仓】{holdings}
【上期决策回顾】{last_context}

【任务】从候选池中选择1-{top_n}只标的, 分配权重(总和≤0.95), 输出纯JSON:
{{
  "allocations": {{
    "代码": {{"weight": 0.0-0.95, "conviction": "high/medium/low", "rationale": "30字内理由"}}
  }},
  "total_invested": 0.0-0.95,
  "market_view": "20字内市场判断"
}}"""

RISK_MGR_PROMPT = """\
你是投委会风险管理官。策略师已提交配置方案,你需要从风险角度审核并微调。

【策略师方案】
{analyst_proposal}

【量化数据】
{metrics_table}

【市场情报】
{market_intel}

【你的风控框架】
1. 波动率检查: 日波动率>2.5%的标的, 权重不应超过40%
2. 动量衰减风险: 加速度<0 且 R²在下降的标的, 建议减仓10-20%
3. 资金面验证: 策略师重仓标的是否有主力资金流入支撑
4. 集中度风险: 单一标的>60%时, 需要R²>0.75且加速度>0的双重确认
5. 回撤保护: 离近期高点>5%的标的, 需要重新评估

【重要原则】
- 你不是否决者。好的方案应该认可并放行
- 只在发现明确量化风险信号时才建议调整
- 调整幅度通常在±15%以内, 不做大幅改动
- 输出最终配置权重(考虑佣金后, 总和≤0.95)

【任务】输出风控审核结果(纯JSON):
{{
  "final_weights": {{"代码": 0.0-0.95}},
  "risk_assessment": "30字内风险评估",
  "adjustments_made": true/false,
  "risk_score": 1-10
}}"""


# ═══════════════════════════════════════════════════════════════
# LangGraph State & Nodes
# ═══════════════════════════════════════════════════════════════

class V2State(TypedDict):
    date: str
    bars: Dict[str, Any]
    all_metrics: Dict[str, Dict[str, float]]
    regime: str
    top_candidates: List[str]
    current_holdings: Dict[str, int]
    cash: float
    total_equity: float
    lookback: int
    top_n: int
    last_decision_context: str
    metrics_table_str: str
    market_intel_str: str
    analyst_output: Dict[str, Any]
    risk_output: Dict[str, Any]
    target_weights: Dict[str, float]
    llm_calls: List[Dict[str, Any]]


def _get_model() -> Optional[str]:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if not os.path.exists(config_path):
        return None
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        llm = cfg.get("llm") or {}
        active = (llm.get("active") or os.environ.get("LLM_PROVIDER") or "volcano").lower()
        spec = (llm.get("providers") or {}).get(active) or {}
        return spec.get("model")
    except Exception:
        return None


def _clean_json(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ── Node 1: 量化引擎 (纯计算, 多周期因子) ──

def quant_engine_node(state: V2State) -> dict:
    bars = state["bars"]
    all_metrics = state.get("all_metrics", {})
    top_n = state.get("top_n", 3)
    holdings = state.get("current_holdings", {})

    valid = {s: m for s, m in all_metrics.items() if s in bars}
    if not valid:
        return {"top_candidates": [], "regime": "unknown",
                "metrics_table_str": "", "market_intel_str": ""}

    sorted_syms = sorted(valid, key=lambda s: valid[s]["efficiency"], reverse=True)
    candidates = sorted_syms[:max(top_n + 2, 6)]
    for s in holdings:
        if s not in candidates and s in valid:
            candidates.append(s)

    # 构建丰富的指标表
    lines = []
    for sym in candidates:
        m = valid[sym]
        tag = f" ★持仓" if sym in holdings else ""
        rank = sorted_syms.index(sym) + 1 if sym in sorted_syms else "?"
        align_str = "↑↑↑" if m["momentum_alignment"] >= 0.9 else ("↑↑" if m["momentum_alignment"] >= 0.6 else "↑↓")
        lines.append(
            f"  #{rank} {_n(sym)}: 效率={m['efficiency']:.2f}, "
            f"动量20d={m['momentum_20d']:.2%}, 10d={m['momentum_10d']:.2%}, 5d={m['momentum_5d']:.2%}, "
            f"加速度={m['momentum_accel']:+.2%}, "
            f"R²={m['r2']:.2f}, 波动={m['volatility']:.2%}, "
            f"离高点={m['drawdown_from_high']:.2%}, 一致性={align_str}{tag}"
        )
    table_str = "\n".join(lines)

    date_ts = pd.Timestamp(state.get("date", ""))
    regime = _intel.get_regime_enhanced(date_ts, valid)
    intel_str = _intel.generate_intel_summary(date_ts, valid, candidates[:top_n])

    top_named = [_n(s) for s in sorted_syms[:top_n]]
    logger.info(f"[Quant] regime={regime} top={top_named}\n{table_str}")
    return {"top_candidates": candidates, "regime": regime,
            "metrics_table_str": table_str, "market_intel_str": intel_str}


# ── Node 2: 策略分析师 (LLM) ──

async def analyst_node(state: V2State) -> dict:
    if not state.get("top_candidates"):
        return {"analyst_output": {}, "llm_calls": state.get("llm_calls", [])}

    top_n = state.get("top_n", 3)
    prompt = ANALYST_PROMPT.format(
        lookback=state.get("lookback", 20),
        metrics_table=state["metrics_table_str"],
        market_intel=state.get("market_intel_str", "暂无"),
        regime=state["regime"],
        holdings=state.get("current_holdings") or "无",
        last_context=state.get("last_decision_context") or "首次决策",
        top_n=top_n,
    )

    # 构建 fallback: 按效率分 Top-N 等权
    all_m = state.get("all_metrics", {})
    candidates = state["top_candidates"]
    fb_alloc = {}
    sel = [s for s in candidates[:top_n] if all_m.get(s, {}).get("efficiency", 0) > 0.1]
    if sel:
        w = round(0.95 / len(sel), 2)
        for sym in sel:
            fb_alloc[sym] = {"weight": w, "conviction": "medium",
                             "rationale": "quant fallback"}
    fallback = {"allocations": fb_alloc,
                "total_invested": sum(v["weight"] for v in fb_alloc.values()),
                "market_view": state.get("regime", "unknown")}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="分析候选池并输出配置方案。", prompt=prompt,
            temperature=0.3, max_tokens=1024, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(_clean_json(resp), fallback)
        used_fb = not result.get("allocations")
        if used_fb:
            result = fallback
        llm_calls.append({"agent": "analyst", "tokens": tokens,
                          "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        logger.error(f"[Analyst ERROR] {e}")
        result = fallback
        llm_calls.append({"agent": "analyst", "tokens": 0,
                          "latency_ms": int(time.time() * 1000) - t0,
                          "error": str(e), "used_fallback": True})

    w = {s: v.get("weight", 0) if isinstance(v, dict) else 0
         for s, v in result.get("allocations", {}).items()}
    w_named = {_n(s): f"{v:.0%}" for s, v in w.items() if v > 0}
    logger.info(f"[Analyst] {w_named}")
    return {"analyst_output": result, "llm_calls": llm_calls}


# ── Node 3: 风控管理官 (LLM) — 终裁 ──

async def risk_mgr_node(state: V2State) -> dict:
    analyst = state.get("analyst_output", {})
    allocs = analyst.get("allocations", {})

    if not allocs:
        return {"risk_output": {}, "target_weights": {},
                "llm_calls": state.get("llm_calls", [])}

    analyst_weights = {s: v.get("weight", 0) if isinstance(v, dict) else 0
                       for s, v in allocs.items()}

    prompt = RISK_MGR_PROMPT.format(
        analyst_proposal=json.dumps(allocs, ensure_ascii=False),
        metrics_table=state["metrics_table_str"],
        market_intel=state.get("market_intel_str", "暂无"),
    )

    fallback = {"final_weights": analyst_weights,
                "risk_assessment": "fallback-直接采纳策略师方案",
                "adjustments_made": False, "risk_score": 5}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="审核策略师方案并输出最终配置。", prompt=prompt,
            temperature=0.15, max_tokens=600, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(_clean_json(resp), fallback)
        used_fb = not result.get("final_weights")
        if used_fb:
            result = fallback
        llm_calls.append({"agent": "risk_mgr", "tokens": tokens,
                          "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        logger.error(f"[RiskMgr ERROR] {e}")
        result = fallback
        llm_calls.append({"agent": "risk_mgr", "tokens": 0,
                          "latency_ms": int(time.time() * 1000) - t0,
                          "error": str(e), "used_fallback": True})

    tw = result.get("final_weights", {})
    tw = {k: max(v, 0) for k, v in tw.items() if isinstance(v, (int, float))}
    total = sum(tw.values())
    if total > 0.98:
        scale = 0.95 / total
        tw = {k: round(v * scale, 4) for k, v in tw.items()}
    tw = {k: v for k, v in tw.items() if v >= 0.02}

    tw_named = {_n(s): f"{v:.0%}" for s, v in tw.items()}
    risk_score = result.get("risk_score", "?")
    logger.info(f"[RiskMgr] Final: {tw_named} | risk_score={risk_score}")
    return {"risk_output": result, "target_weights": tw, "llm_calls": llm_calls}


# ── Graph ──

def build_v2_graph():
    wf = StateGraph(V2State)
    wf.add_node("quant_engine", quant_engine_node)
    wf.add_node("analyst", analyst_node)
    wf.add_node("risk_mgr", risk_mgr_node)
    wf.set_entry_point("quant_engine")
    wf.add_edge("quant_engine", "analyst")
    wf.add_edge("analyst", "risk_mgr")
    wf.add_edge("risk_mgr", END)
    return wf.compile()


# ═══════════════════════════════════════════════════════════════
# Strategy Class
# ═══════════════════════════════════════════════════════════════

class AgentStrategyV2(BreadFreeStrategy):
    """
    V2 多智能体涌现决策系统

    QuantPrep → Analyst (LLM) → RiskMgr (LLM)

    Analyst 提出配置 → RiskMgr 风控终裁
    LLM 全部失败时自动退化为纯效率轮动
    """

    def __init__(self, broker, lookback_period=20, hold_period=20, top_n=3,
                 lot_size=100, graph_timeout_seconds=180, **kwargs):
        super().__init__(broker, lot_size=lot_size)
        self.lookback_period = lookback_period
        self.hold_period = hold_period
        self.top_n = top_n
        self.graph_timeout = graph_timeout_seconds
        self.days_counter = 0
        self.app = build_v2_graph()
        self.last_decision_context = ""
        self._total_llm_calls = 0
        self._total_llm_tokens = 0
        self._last_equity = 0.0

    def on_bar(self, date, bars):
        for symbol, bar in bars.items():
            if symbol not in self.history:
                self.history[symbol] = []
            self.history[symbol].append(bar['close'])

        self.days_counter += 1

        min_len = self.lookback_period + 10
        if not all(len(self.history.get(s, [])) >= min_len for s in bars):
            return

        if self.days_counter % self.hold_period != 0 and self.days_counter > 1:
            return

        all_metrics = {}
        for symbol in bars:
            m = compute_advanced_metrics(self.history.get(symbol, []), self.lookback_period)
            if m:
                all_metrics[symbol] = m

        if not all_metrics:
            return

        pos_snapshot = {}
        for s, p in self.broker.positions.items():
            pos_snapshot[s] = getattr(p, 'quantity', p) if hasattr(p, 'quantity') else p

        total_equity = self.broker.cash
        for s, qty in pos_snapshot.items():
            price = bars.get(s, {}).get('close', 0)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty else 0
            total_equity += qty * price

        # 决策记忆: 包含上期收益反馈
        pnl_str = ""
        if self._last_equity > 0:
            period_ret = (total_equity / self._last_equity - 1)
            pnl_str = f", 本期收益={period_ret:+.2%}"

        state: V2State = {
            "date": str(date),
            "bars": bars,
            "all_metrics": all_metrics,
            "regime": "",
            "top_candidates": [],
            "current_holdings": pos_snapshot,
            "cash": self.broker.cash,
            "total_equity": total_equity,
            "lookback": self.lookback_period,
            "top_n": self.top_n,
            "last_decision_context": self.last_decision_context,
            "metrics_table_str": "",
            "market_intel_str": "",
            "analyst_output": {},
            "risk_output": {},
            "target_weights": {},
            "llm_calls": [],
        }

        holdings_named = [_n(s) for s in pos_snapshot.keys()]
        logger.info(f"\n{'=' * 55}\n[V2] {date} 调仓 | "
                    f"资产¥{total_equity:,.0f}{pnl_str} | 持仓{holdings_named}")

        target_weights = {}
        try:
            final = self._run_graph(state)
            target_weights = final.get("target_weights", {})
            for c in final.get("llm_calls", []):
                self._total_llm_calls += 1
                self._total_llm_tokens += c.get("tokens", 0)
        except Exception as e:
            logger.error(f"[V2 GRAPH ERROR] {e}")
            target_weights = self._quant_fallback(all_metrics)

        if target_weights:
            self._execute_trades(date, target_weights, bars, total_equity)
            tw_str = json.dumps({_n(s): f"{w:.0%}" for s, w in target_weights.items()},
                                ensure_ascii=False)
            self.last_decision_context = (
                f"上期({str(date)[:10]}): 配置={tw_str}, "
                f"资产=¥{total_equity:,.0f}{pnl_str}"
            )
            self._last_equity = total_equity

    def _quant_fallback(self, metrics: Dict) -> Dict[str, float]:
        valid = {s: m for s, m in metrics.items() if m.get("efficiency", 0) > 0.1}
        if not valid:
            return {}
        ranked = sorted(valid, key=lambda s: valid[s]["efficiency"], reverse=True)
        sel = ranked[:self.top_n]
        w = round(0.95 / len(sel), 4)
        return {s: w for s in sel}

    def _run_graph(self, state: dict) -> dict:
        async def _go():
            return await asyncio.wait_for(
                self.app.ainvoke(state), timeout=self.graph_timeout)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _go()).result(
                    timeout=self.graph_timeout + 10)
        return asyncio.run(_go())

    def _execute_trades(self, date, tw: Dict[str, float], bars, total_equity: float):
        if not tw:
            return
        tw_named = {_n(s): f"{v:.0%}" for s, v in tw.items()}
        logger.info(f"[EXEC] {tw_named}")
        cr = self.broker.commission_rate

        # 先卖: 清仓不在目标中的持仓
        for symbol in list(self.broker.positions.keys()):
            if tw.get(symbol, 0) == 0:
                pos = self.broker.positions.get(symbol)
                if pos is None:
                    continue
                qty = getattr(pos, 'quantity', pos)
                price = bars.get(symbol, {}).get('close', 0)
                if isinstance(price, pd.Series):
                    price = float(price.iloc[0]) if not price.empty else 0
                if qty > 0 and price > 0:
                    self.broker.sell(date, symbol, price, qty)
                    logger.info(f"  清仓 {_n(symbol)}: {qty}股")

        # 减仓: 降低超配持仓
        for symbol in list(self.broker.positions.keys()):
            w = tw.get(symbol, 0)
            if w <= 0:
                continue
            pos = self.broker.positions.get(symbol)
            if not pos:
                continue
            cur_qty = getattr(pos, 'quantity', pos)
            price = bars.get(symbol, {}).get('close', 0)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty else 0
            if price <= 0:
                continue
            target_qty = int(total_equity * w / price / self.lot_size) * self.lot_size
            if cur_qty > target_qty:
                sell_q = ((cur_qty - target_qty) // self.lot_size) * self.lot_size
                if sell_q > 0:
                    self.broker.sell(date, symbol, price, sell_q)
                    logger.info(f"  减仓 {_n(symbol)}: -{sell_q}股")

        # 买入: 按权重高低排序, 佣金感知
        cash = self.broker.cash
        for symbol, weight in sorted(tw.items(), key=lambda x: -x[1]):
            if weight <= 0 or symbol not in bars:
                continue
            price = bars[symbol].get('close', 0)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty else 0
            if price <= 0:
                continue
            pos = self.broker.positions.get(symbol)
            cur_qty = getattr(pos, 'quantity', pos) if pos else 0
            target_qty = int(total_equity * weight / price / self.lot_size) * self.lot_size
            if target_qty > cur_qty:
                buy_qty = target_qty - cur_qty
                cost = buy_qty * price * (1 + cr)
                if cost > cash:
                    buy_qty = int(cash / (price * (1 + cr)) / self.lot_size) * self.lot_size
                    cost = buy_qty * price * (1 + cr)
                if buy_qty > 0 and cost <= cash:
                    self.broker.buy(date, symbol, price, buy_qty)
                    cash -= cost
                    logger.info(f"  买入 {_n(symbol)}: +{buy_qty}股")
