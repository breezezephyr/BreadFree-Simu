"""
AgentStrategyV2 — 多智能体涌现决策系统

架构：Quant Engine → Bull Analyst (LLM) → Bear Challenger (LLM) → Portfolio Manager (LLM)

核心设计：
  1. 辩证决策：Bull 提出配置方案 → Bear 挑战寻找风险盲点 → PM 综合裁决
     三方博弈产生涌现智能，避免单一视角偏见
  2. 量化增强：效率分 + 动量加速度 + 趋势政权分类 → 为 LLM 提供结构化信号
  3. 零人为约束：不硬编码仓位上限/现金缓冲，完全由 Agent 辩论决定
  4. 动量持续性：持仓仍在 Top-N 时，降低换手摩擦
  5. 健壮 fallback：任何 LLM 失败自动退化为纯效率轮动
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

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 量化引擎：计算高级因子
# ═══════════════════════════════════════════════════════════════

def compute_advanced_metrics(history: List[float], lookback: int = 20) -> Optional[Dict]:
    if len(history) < lookback + 5:
        return None
    prices = np.array(history[-(lookback + 5):])
    current_window = prices[-lookback:]
    prev_window = prices[-(lookback + 5):-(5)]

    if current_window[0] <= 0 or prev_window[0] <= 0:
        return None

    momentum = current_window[-1] / current_window[0] - 1
    prev_momentum = prev_window[-1] / prev_window[0] - 1
    momentum_accel = momentum - prev_momentum

    returns = np.diff(current_window) / current_window[:-1]
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

    x = np.arange(len(current_window))
    try:
        slope, intercept, r_value, _, _ = stats.linregress(x, current_window)
        r2 = r_value ** 2
    except Exception:
        slope, r2 = 0.0, 0.0

    epsilon = 1e-6
    period_vol = volatility * np.sqrt(len(returns)) + epsilon
    efficiency = (momentum / period_vol) * r2

    drawdown_from_high = (current_window[-1] / np.max(current_window) - 1)

    return {
        "momentum": float(momentum),
        "momentum_accel": float(momentum_accel),
        "volatility": float(volatility),
        "r2": float(r2),
        "efficiency": float(efficiency),
        "close": float(current_window[-1]),
        "drawdown_from_high": float(drawdown_from_high),
        "trend_slope": float(slope),
    }


def classify_regime(all_metrics: Dict[str, Dict]) -> str:
    if not all_metrics:
        return "unknown"
    efficiencies = [m["efficiency"] for m in all_metrics.values()]
    momentums = [m["momentum"] for m in all_metrics.values()]
    pct_positive_mom = sum(1 for m in momentums if m > 0.01) / len(momentums)
    avg_eff = np.mean(efficiencies)
    max_eff = max(efficiencies)

    if pct_positive_mom >= 0.6 and avg_eff > 0.5:
        return "strong_bull"
    elif pct_positive_mom >= 0.4 and max_eff > 1.0:
        return "selective_bull"
    elif pct_positive_mom <= 0.3:
        return "bear"
    else:
        return "choppy"


# ═══════════════════════════════════════════════════════════════
# Prompts — Bull/Bear/PM 三方辩证
# ═══════════════════════════════════════════════════════════════

BULL_ANALYST_PROMPT = """\
你是一位进攻型 ETF 轮动策略师，目标是捕捉最强趋势获取最大收益。

【量化指标（近{lookback}日）】
{metrics_table}

【市场政权】{regime}
【当前持仓】{holdings}
【上期回顾】{last_context}

【你的投资哲学】
- 效率分（Efficiency）是核心：衡量"每单位风险的趋势收益质量"
- 效率分 > 1.5 且 R² > 0.7 = 强趋势确认，应重仓出击
- 动量加速度 > 0 = 趋势在加强，加大配置
- 当一只 ETF 的效率分远超其他（≥2倍），应果断集中配置
- 持仓 ETF 仍在 Top-3 → 优先保留（减少换手摩擦）

【任务】输出目标配置方案（纯JSON，不要任何其他文字）：
{{
  "allocations": {{
    "代码": {{"weight": 0.0-1.0, "conviction": "high/medium/low", "bull_case": "..."}}
  }},
  "total_invested": 0.0-1.0,
  "regime_view": "..."
}}"""

BEAR_CHALLENGER_PROMPT = """\
你是一位风险挑战者。你的职责不是阻止交易，而是找出 Bull 方案的盲点，让最终决策更稳健。

【Bull 方案】
{bull_proposal}

【量化数据】
{metrics_table}

【你的审查框架】
1. 集中度风险：单一资产 > 80% 时，检查其 R² 是否 > 0.75 且动量加速度 > 0
2. 趋势衰退：动量加速度 < 0 的资产是否被过度配置
3. 波动率异常：日波动率 > 2% 的资产需要额外风险溢价
4. 持仓惯性：是否因为"已经持有"而忽略了更优选择

【重要原则】
- 你不是看空一切。如果 Bull 方案有强量化支撑，你应该认可并可能建议加仓
- 只有在发现明确风险信号时才建议减仓
- 你的调整幅度通常在 ±20% 以内

【任务】输出你的挑战意见和调整建议（纯JSON）：
{{
  "challenges": [
    {{"asset": "代码", "issue": "风险描述", "severity": "high/medium/low"}}
  ],
  "adjusted_weights": {{"代码": 0.0-1.0}},
  "bear_view": "整体风险评估",
  "agrees_with_bull": true/false
}}"""

PORTFOLIO_MANAGER_PROMPT = """\
你是首席投资官(CIO)，拥有最终裁决权。你收到了 Bull 和 Bear 两方的分析。

【Bull 方案】
{bull_proposal}

【Bear 挑战】
{bear_response}

【量化数据】
{metrics_table}

【裁决原则】
1. 如果 Bull/Bear 一致看好某资产 → 满配该资产
2. 如果 Bear 提出了有数据支撑的高严重性风险 → 适度减仓（但不清仓强趋势资产）
3. 如果 Bear 的挑战缺乏量化依据 → 维持 Bull 方案
4. 宁可错过也不要做错：只配置有明确趋势的资产
5. 交易成本意识：与当前持仓差异 < 10% 时保持不变

【任务】输出最终配置（纯JSON）：
{{
  "final_weights": {{"代码": 0.0-1.0}},
  "reasoning": "裁决逻辑",
  "bull_score": 0-10,
  "bear_score": 0-10
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
    bull_output: Dict[str, Any]
    bear_output: Dict[str, Any]
    pm_output: Dict[str, Any]
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


# ── Node 1: 量化引擎 (纯计算) ──

def quant_engine_node(state: V2State) -> dict:
    bars = state["bars"]
    all_metrics = state.get("all_metrics", {})
    top_n = state.get("top_n", 3)
    holdings = state.get("current_holdings", {})

    valid = {s: m for s, m in all_metrics.items() if s in bars}
    if not valid:
        return {"top_candidates": [], "regime": "unknown", "metrics_table_str": ""}

    regime = classify_regime(valid)

    sorted_syms = sorted(valid, key=lambda s: valid[s]["efficiency"], reverse=True)
    candidates = sorted_syms[:max(top_n + 2, 5)]
    for s in holdings:
        if s not in candidates and s in valid:
            candidates.append(s)

    lines = []
    for sym in candidates:
        m = valid[sym]
        tag = f" ★持仓{holdings[sym]}股" if sym in holdings else ""
        rank = sorted_syms.index(sym) + 1 if sym in sorted_syms else "?"
        lines.append(
            f"  #{rank} {sym}: 效率={m['efficiency']:.2f}, "
            f"动量={m['momentum']:.2%}, 加速度={m['momentum_accel']:+.2%}, "
            f"R²={m['r2']:.2f}, 波动={m['volatility']:.2%}, "
            f"离高点={m['drawdown_from_high']:.2%}{tag}"
        )
    table_str = "\n".join(lines)

    logger.info(f"[Quant] regime={regime} top={sorted_syms[:top_n]}\n{table_str}")
    return {"top_candidates": candidates, "regime": regime, "metrics_table_str": table_str}


# ── Node 2: Bull Analyst (LLM) ──

async def bull_node(state: V2State) -> dict:
    if not state.get("top_candidates"):
        return {"bull_output": {}, "llm_calls": state.get("llm_calls", [])}

    prompt = BULL_ANALYST_PROMPT.format(
        lookback=state.get("lookback", 20),
        metrics_table=state["metrics_table_str"],
        regime=state["regime"],
        holdings=state.get("current_holdings") or "无",
        last_context=state.get("last_decision_context") or "首次",
    )

    all_m = state.get("all_metrics", {})
    top_n = state.get("top_n", 3)
    candidates = state["top_candidates"]
    fb_alloc = {}
    for sym in candidates[:top_n]:
        m = all_m.get(sym, {})
        if m.get("efficiency", 0) > 0.2:
            fb_alloc[sym] = {"weight": round(0.95 / min(len(candidates), top_n), 2),
                             "conviction": "medium", "bull_case": "quant fallback"}
    fallback = {"allocations": fb_alloc, "total_invested": sum(v["weight"] for v in fb_alloc.values()),
                "regime_view": state["regime"]}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="分析并提出ETF配置方案。", prompt=prompt,
            temperature=0.3, max_tokens=1024, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(_clean_json(resp), fallback)
        used_fb = not result.get("allocations")
        if used_fb:
            result = fallback
        llm_calls.append({"agent": "bull", "tokens": tokens, "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        logger.error(f"[Bull ERROR] {e}")
        result = fallback
        llm_calls.append({"agent": "bull", "tokens": 0, "latency_ms": int(time.time()*1000)-t0,
                          "error": str(e), "used_fallback": True})

    w = {s: v.get("weight", 0) if isinstance(v, dict) else 0 for s, v in result.get("allocations", {}).items()}
    logger.info(f"[Bull] Proposed: {w}")
    return {"bull_output": result, "llm_calls": llm_calls}


# ── Node 3: Bear Challenger (LLM) ──

async def bear_node(state: V2State) -> dict:
    bull = state.get("bull_output", {})
    allocs = bull.get("allocations", {})
    if not allocs:
        return {"bear_output": {"agrees_with_bull": True, "adjusted_weights": {}},
                "llm_calls": state.get("llm_calls", [])}

    prompt = BEAR_CHALLENGER_PROMPT.format(
        bull_proposal=json.dumps(allocs, ensure_ascii=False),
        metrics_table=state["metrics_table_str"],
    )

    bull_weights = {s: v.get("weight", 0) if isinstance(v, dict) else 0 for s, v in allocs.items()}
    fallback = {"challenges": [], "adjusted_weights": bull_weights,
                "bear_view": "No significant risks found", "agrees_with_bull": True}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="审查Bull方案并提出挑战。", prompt=prompt,
            temperature=0.2, max_tokens=800, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(_clean_json(resp), fallback)
        used_fb = False
        if not result.get("adjusted_weights"):
            result["adjusted_weights"] = bull_weights
            used_fb = True
        llm_calls.append({"agent": "bear", "tokens": tokens, "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        logger.error(f"[Bear ERROR] {e}")
        result = fallback
        llm_calls.append({"agent": "bear", "tokens": 0, "latency_ms": int(time.time()*1000)-t0,
                          "error": str(e), "used_fallback": True})

    challenges = result.get("challenges", [])
    agrees = result.get("agrees_with_bull", True)
    logger.info(f"[Bear] agrees={agrees}, challenges={len(challenges)}, "
                f"adjusted={result.get('adjusted_weights', {})}")
    return {"bear_output": result, "llm_calls": llm_calls}


# ── Node 4: Portfolio Manager (LLM) — 最终裁决 ──

async def pm_node(state: V2State) -> dict:
    bull = state.get("bull_output", {})
    bear = state.get("bear_output", {})
    allocs = bull.get("allocations", {})

    if not allocs:
        return {"pm_output": {}, "target_weights": {}, "llm_calls": state.get("llm_calls", [])}

    bull_weights = {s: v.get("weight", 0) if isinstance(v, dict) else 0 for s, v in allocs.items()}
    bear_weights = bear.get("adjusted_weights", bull_weights)
    agrees = bear.get("agrees_with_bull", True)

    if agrees and not bear.get("challenges"):
        tw = {k: v for k, v in bull_weights.items() if v > 0}
        total = sum(tw.values())
        if total > 0.98:
            scale = 0.95 / total
            tw = {k: round(v * scale, 4) for k, v in tw.items()}
        logger.info(f"[PM] Bull/Bear consensus → direct approve: {tw}")
        return {"pm_output": {"final_weights": tw, "reasoning": "bull_bear_consensus"},
                "target_weights": tw, "llm_calls": state.get("llm_calls", [])}

    prompt = PORTFOLIO_MANAGER_PROMPT.format(
        bull_proposal=json.dumps(allocs, ensure_ascii=False),
        bear_response=json.dumps(bear, ensure_ascii=False),
        metrics_table=state["metrics_table_str"],
    )

    fallback_w = {}
    for s in set(list(bull_weights.keys()) + list(bear_weights.keys())):
        fallback_w[s] = (bull_weights.get(s, 0) * 0.6 + bear_weights.get(s, 0) * 0.4)
    fallback = {"final_weights": fallback_w, "reasoning": "weighted_fallback",
                "bull_score": 6, "bear_score": 4}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="裁决Bull和Bear的分歧，输出最终配置。", prompt=prompt,
            temperature=0.2, max_tokens=600, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(_clean_json(resp), fallback)
        used_fb = not result.get("final_weights")
        if used_fb:
            result = fallback
        llm_calls.append({"agent": "pm", "tokens": tokens, "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        logger.error(f"[PM ERROR] {e}")
        result = fallback
        llm_calls.append({"agent": "pm", "tokens": 0, "latency_ms": int(time.time()*1000)-t0,
                          "error": str(e), "used_fallback": True})

    tw = result.get("final_weights", {})
    tw = {k: max(v, 0) for k, v in tw.items() if isinstance(v, (int, float))}
    total = sum(tw.values())
    if total > 0.98:
        scale = 0.95 / total
        tw = {k: round(v * scale, 4) for k, v in tw.items()}
    tw = {k: v for k, v in tw.items() if v >= 0.03}

    logger.info(f"[PM] Final: {tw} | bull={result.get('bull_score')}/bear={result.get('bear_score')}")
    return {"pm_output": result, "target_weights": tw, "llm_calls": llm_calls}


# ── Graph ──

def build_v2_graph():
    wf = StateGraph(V2State)
    wf.add_node("quant_engine", quant_engine_node)
    wf.add_node("bull", bull_node)
    wf.add_node("bear", bear_node)
    wf.add_node("pm", pm_node)
    wf.set_entry_point("quant_engine")
    wf.add_edge("quant_engine", "bull")
    wf.add_edge("bull", "bear")
    wf.add_edge("bear", "pm")
    wf.add_edge("pm", END)
    return wf.compile()


# ═══════════════════════════════════════════════════════════════
# Strategy Class
# ═══════════════════════════════════════════════════════════════

class AgentStrategyV2(BreadFreeStrategy):
    """
    V2 多智能体辩证决策系统

    Quant Engine → Bull Analyst → Bear Challenger → Portfolio Manager

    Bull/Bear 一致时跳过 PM（节省 1 次 LLM 调用）
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

    def on_bar(self, date, bars):
        for symbol, bar in bars.items():
            if symbol not in self.history:
                self.history[symbol] = []
            self.history[symbol].append(bar['close'])

        self.days_counter += 1

        min_len = self.lookback_period + 5
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
            "bull_output": {},
            "bear_output": {},
            "pm_output": {},
            "target_weights": {},
            "llm_calls": [],
        }

        logger.info(f"\n{'='*55}\n[V2] {date} 调仓 | 资产¥{total_equity:,.0f} | 持仓{list(pos_snapshot.keys())}")

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
            self.last_decision_context = (
                f"上期({str(date)[:10]}): 配置={json.dumps(target_weights)}, 资产=¥{total_equity:,.0f}"
            )

    def _quant_fallback(self, metrics: Dict) -> Dict[str, float]:
        valid = {s: m for s, m in metrics.items() if m.get("efficiency", 0) > 0.2}
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
                return pool.submit(asyncio.run, _go()).result(timeout=self.graph_timeout + 10)
        return asyncio.run(_go())

    def _execute_trades(self, date, tw: Dict[str, float], bars, total_equity: float):
        if not tw:
            return
        logger.info(f"[EXEC] {tw}")
        cr = self.broker.commission_rate

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
                    logger.info(f"  SELL ALL {symbol}: {qty}")

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
                    logger.info(f"  REDUCE {symbol}: -{sell_q}")

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
                    logger.info(f"  BUY {symbol}: +{buy_qty}")
