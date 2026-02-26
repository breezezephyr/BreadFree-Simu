"""
AgentStrategyV2 — 量化信号驱动 + LLM 多 Agent 投委会优化版

核心设计理念：
  "量化先行、LLM 精调、风控护航"

V1 -> V2 关键改进：
  1. 多资产轮动：分析全部 ETF 池，而非只看第一只
  2. 信号先行：先用量化指标（效率分、动量、波动率、R²）筛选 Top-N 候选
  3. 周期调仓：仅在调仓日调用 LLM（降低成本、减少噪声交易）
  4. Prompt 重构：去除过度保守偏见，引入结构化定量框架
  5. 决策记忆：携带上期持仓和收益上下文，保持一致性
  6. 佣金感知下单：计算目标手数时扣除佣金，避免 broker 拒单
  7. 健壮 fallback：LLM 失败时自动降级为纯效率轮动
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
# Prompts — 核心优化区
# ═══════════════════════════════════════════════════════════════

ANALYST_V2_PROMPT = """\
你是一位顶级量化 ETF 轮动分析师。你的目标是从候选 ETF 中挑选最值得配置的资产。

【投资框架】
- 核心因子：效率分 = (动量收益 / 波动率) × R²（趋势稳定性）
- 效率分 > 1.5 = 强趋势，优先满仓配置
- 效率分 0.5~1.5 = 中等趋势，适度配置
- 效率分 < 0.5 = 弱趋势或震荡，降低配置或跳过
- 动量为负且效率分为负 = 下行趋势，坚决回避

【候选池量化指标（近 {lookback} 个交易日）】
{metrics_table}

【当前持仓】
{current_holdings}

【上期决策回顾】
{last_decision_context}

【任务】
1. 结合效率分排名和趋势质量，选出 1~{top_n} 只最优 ETF
2. 为每只 ETF 分配目标权重（所有权重之和 ≤ 0.95，保留 5% 现金缓冲）
3. 如果所有候选效率分 < 0.3，可以全部持有现金（空仓）

输出纯 JSON，不要任何额外文字：
{{
  "allocations": {{
    "代码1": {{"weight": 0.5, "conviction": "high/medium/low", "reason": "..."}},
    "代码2": {{"weight": 0.4, "conviction": "medium", "reason": "..."}}
  }},
  "market_regime": "trending_up/trending_down/range_bound/mixed",
  "cash_reserve": 0.1
}}"""

RISK_V2_PROMPT = """\
你是一位平衡型风控经理。你的目标是在追求收益的同时管理好回撤风险。

【核心原则】
- 你不是要消灭风险，而是要管理风险。适度的风险暴露是获取收益的前提。
- 只在以下情况大幅减仓：1) 市场明确处于下行趋势  2) 波动率急剧放大  3) 多个资产同时走弱
- 正常市况下，总仓位应保持在 0.7~0.95 之间
- 单一资产权重上限 0.60（集中度控制）

【分析师方案】
{analyst_proposal}

【资产量化数据】
{metrics_summary}

【当前组合状态】
现金: ¥{cash:.2f} | 总资产: ¥{total_equity:.2f} | 持仓: {positions}

【任务】
1. 审核分析师的配置方案
2. 如果分析师方案合理（效率分支撑、趋势明确），应当批准或微调
3. 如果发现高风险（多资产效率分骤降、波动率飙升），可以缩减至 0.5 以下
4. 输出最终目标权重

输出纯 JSON：
{{
  "target_weights": {{"代码1": 0.5, "代码2": 0.4}},
  "total_exposure": 0.9,
  "risk_level": "low/moderate/elevated/high",
  "adjustment_reason": "..."
}}"""


# ═══════════════════════════════════════════════════════════════
# LangGraph State & Nodes
# ═══════════════════════════════════════════════════════════════

class V2AgentState(TypedDict):
    date: str
    bars: Dict[str, Any]
    all_metrics: Dict[str, Dict[str, float]]
    top_candidates: List[str]
    current_holdings: Dict[str, int]
    cash: float
    total_equity: float
    lookback: int
    top_n: int
    last_decision_context: str
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
        providers = llm.get("providers") or {}
        spec = providers.get(active) or {}
        return spec.get("model")
    except Exception:
        return None


# ── Node 1: 量化数据准备 (纯计算，无 LLM) ──

def quant_data_prep_node(state: V2AgentState) -> dict:
    bars = state["bars"]
    all_metrics = state.get("all_metrics", {})
    top_n = state.get("top_n", 3)

    valid = {s: m for s, m in all_metrics.items() if s in bars}
    if not valid:
        return {"top_candidates": [], "all_metrics": {}}

    sorted_symbols = sorted(valid.keys(), key=lambda s: valid[s]["efficiency"], reverse=True)
    top_candidates = sorted_symbols[:max(top_n + 2, 5)]

    current_holdings = state.get("current_holdings", {})
    for s in current_holdings:
        if s not in top_candidates and s in valid:
            top_candidates.append(s)

    logger.info(f"[QuantPrep] Top candidates: {top_candidates[:top_n]} "
                f"(+holdings: {[s for s in current_holdings if s not in sorted_symbols[:top_n]]})")
    return {"top_candidates": top_candidates}


# ── Node 2: 分析师 Agent (LLM) ──

async def analyst_v2_node(state: V2AgentState) -> dict:
    candidates = state.get("top_candidates", [])
    all_metrics = state.get("all_metrics", {})
    if not candidates:
        return {"analyst_output": {}, "llm_calls": state.get("llm_calls", [])}

    lines = []
    for sym in candidates:
        m = all_metrics.get(sym, {})
        tag = ""
        if sym in state.get("current_holdings", {}):
            tag = f" [持仓: {state['current_holdings'][sym]}股]"
        lines.append(
            f"  {sym}: 动量={m.get('momentum', 0):.2%}, "
            f"波动率={m.get('volatility', 0):.2%}, "
            f"R²={m.get('r2', 0):.2f}, "
            f"效率分={m.get('efficiency', 0):.2f}, "
            f"现价={m.get('close', 0):.3f}{tag}"
        )
    metrics_table = "\n".join(lines)

    holdings_str = "无持仓" if not state.get("current_holdings") else str(state["current_holdings"])
    last_ctx = state.get("last_decision_context", "首次决策，无历史参考")

    prompt = ANALYST_V2_PROMPT.format(
        lookback=state.get("lookback", 20),
        metrics_table=metrics_table,
        current_holdings=holdings_str,
        last_decision_context=last_ctx,
        top_n=state.get("top_n", 3),
    )

    fallback_alloc = {}
    for sym in candidates[:state.get("top_n", 3)]:
        m = all_metrics.get(sym, {})
        if m.get("efficiency", 0) > 0.3:
            fallback_alloc[sym] = {"weight": round(0.9 / min(len(candidates), state.get("top_n", 3)), 2),
                                   "conviction": "medium", "reason": "quant fallback"}
    fallback = {"allocations": fallback_alloc, "market_regime": "mixed", "cash_reserve": 0.1}

    llm_calls = list(state.get("llm_calls", []))
    used_fallback = False
    start_ms = int(time.time() * 1000)

    try:
        response, tokens = await async_hunyuan_chat(
            query="基于量化指标，选择最优ETF并分配权重。", prompt=prompt,
            temperature=0.3, max_tokens=1024, timeout_seconds=60, max_retries=2,
        )
        latency = int(time.time() * 1000) - start_ms
        cleaned = _clean_llm_json(response)
        result = parse_llm_response(cleaned, fallback)
        if not result.get("allocations"):
            result = fallback
            used_fallback = True
        llm_calls.append({"agent": "analyst_v2", "tokens": tokens,
                          "latency_ms": latency, "used_fallback": used_fallback})
    except Exception as e:
        logger.error(f"[AnalystV2 ERROR] {e}")
        result = fallback
        used_fallback = True
        llm_calls.append({"agent": "analyst_v2", "tokens": 0,
                          "latency_ms": int(time.time() * 1000) - start_ms,
                          "error": str(e), "used_fallback": True})

    proposed = {s: v.get("weight", 0) if isinstance(v, dict) else 0
                for s, v in result.get("allocations", {}).items()}
    logger.info(f"[AnalystV2] Proposed: {proposed} regime={result.get('market_regime')}"
                f"{' (FALLBACK)' if used_fallback else ''}")
    return {"analyst_output": result, "llm_calls": llm_calls}


# ── Node 3: 风控 Agent (LLM) ──

async def risk_v2_node(state: V2AgentState) -> dict:
    analyst = state.get("analyst_output", {})
    allocations = analyst.get("allocations", {})
    if not allocations:
        return {"risk_output": {}, "target_weights": {},
                "llm_calls": state.get("llm_calls", [])}

    all_metrics = state.get("all_metrics", {})
    metrics_lines = []
    for sym in allocations:
        m = all_metrics.get(sym, {})
        metrics_lines.append(f"  {sym}: eff={m.get('efficiency', 0):.2f}, "
                             f"mom={m.get('momentum', 0):.2%}, vol={m.get('volatility', 0):.2%}")

    prompt = RISK_V2_PROMPT.format(
        analyst_proposal=json.dumps(allocations, ensure_ascii=False),
        metrics_summary="\n".join(metrics_lines),
        cash=state.get("cash", 0),
        total_equity=state.get("total_equity", 0),
        positions=str(state.get("current_holdings", {})),
    )

    proposed_weights = {s: v.get("weight", 0) if isinstance(v, dict) else 0
                        for s, v in allocations.items()}
    fallback = {"target_weights": proposed_weights,
                "total_exposure": sum(proposed_weights.values()),
                "risk_level": "moderate", "adjustment_reason": "direct pass-through"}

    llm_calls = list(state.get("llm_calls", []))
    used_fallback = False
    start_ms = int(time.time() * 1000)

    try:
        response, tokens = await async_hunyuan_chat(
            query="审核并调整投资组合权重。", prompt=prompt,
            temperature=0.2, max_tokens=512, timeout_seconds=60, max_retries=2,
        )
        latency = int(time.time() * 1000) - start_ms
        cleaned = _clean_llm_json(response)
        result = parse_llm_response(cleaned, fallback)
        tw = result.get("target_weights", {})
        if not tw or sum(tw.values()) <= 0:
            tw = proposed_weights
            used_fallback = True
        llm_calls.append({"agent": "risk_v2", "tokens": tokens,
                          "latency_ms": latency, "used_fallback": used_fallback})
    except Exception as e:
        logger.error(f"[RiskV2 ERROR] {e}")
        tw = proposed_weights
        result = fallback
        used_fallback = True
        llm_calls.append({"agent": "risk_v2", "tokens": 0,
                          "latency_ms": int(time.time() * 1000) - start_ms,
                          "error": str(e), "used_fallback": True})

    tw = {k: max(v, 0) for k, v in tw.items()}
    total = sum(tw.values())
    if total > 0.98:
        scale = 0.95 / total
        tw = {k: round(v * scale, 4) for k, v in tw.items()}
    tw = {k: v for k, v in tw.items() if v >= 0.05}

    logger.info(f"[RiskV2] Final weights: {tw} risk={result.get('risk_level')}"
                f"{' (FALLBACK)' if used_fallback else ''}")
    return {"risk_output": result, "target_weights": tw, "llm_calls": llm_calls}


def _clean_llm_json(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ── Graph ──

def build_v2_graph():
    wf = StateGraph(V2AgentState)
    wf.add_node("quant_prep", quant_data_prep_node)
    wf.add_node("analyst", analyst_v2_node)
    wf.add_node("risk_mgr", risk_v2_node)
    wf.set_entry_point("quant_prep")
    wf.add_edge("quant_prep", "analyst")
    wf.add_edge("analyst", "risk_mgr")
    wf.add_edge("risk_mgr", END)
    return wf.compile()


# ═══════════════════════════════════════════════════════════════
# Strategy Class
# ═══════════════════════════════════════════════════════════════

class AgentStrategyV2(BreadFreeStrategy):
    """
    V2 多资产 LLM 投委会策略

    架构: QuantPrep (rule) -> Analyst (LLM) -> RiskMgr (LLM)
    - 仅在调仓日调用 LLM（每 hold_period 天一次）
    - 支持全部 ETF 池的多资产轮动
    - 量化信号兜底：LLM 失败时退化为纯效率轮动
    """

    def __init__(self, broker, lookback_period=20, hold_period=20, top_n=3,
                 lot_size=100, graph_timeout_seconds=120, **kwargs):
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

        data_ready = all(
            len(self.history.get(s, [])) >= self.lookback_period
            for s in bars
        )
        if not data_ready:
            return

        if self.days_counter % self.hold_period != 0 and self.days_counter > 1:
            return

        all_metrics = {}
        for symbol in bars:
            m = calculate_efficiency_metrics(
                self.history.get(symbol, []), lookback=self.lookback_period
            )
            if m:
                all_metrics[symbol] = m

        if not all_metrics:
            return

        pos_snapshot = {}
        for s, p in self.broker.positions.items():
            qty = getattr(p, 'quantity', p) if hasattr(p, 'quantity') else p
            pos_snapshot[s] = qty

        total_equity = self.broker.cash
        for s, qty in pos_snapshot.items():
            price = bars.get(s, {}).get('close', 0)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty else 0
            total_equity += qty * price

        initial_state: V2AgentState = {
            "date": str(date),
            "bars": bars,
            "all_metrics": all_metrics,
            "top_candidates": [],
            "current_holdings": pos_snapshot,
            "cash": self.broker.cash,
            "total_equity": total_equity,
            "lookback": self.lookback_period,
            "top_n": self.top_n,
            "last_decision_context": self.last_decision_context,
            "analyst_output": {},
            "risk_output": {},
            "target_weights": {},
            "llm_calls": [],
        }

        logger.info(f"\n{'='*50}\n[V2] {date} 调仓日 | 总资产 ¥{total_equity:,.2f}")

        target_weights = {}
        try:
            final_state = self._run_graph(initial_state)
            target_weights = final_state.get("target_weights", {})
            for call in final_state.get("llm_calls", []):
                self._total_llm_calls += 1
                self._total_llm_tokens += call.get("tokens", 0)
        except Exception as e:
            logger.error(f"[V2 GRAPH ERROR] {e}")
            target_weights = self._quant_fallback(all_metrics)
            logger.info(f"[V2 FALLBACK] weights={target_weights}")

        if target_weights:
            self._execute_trades(date, target_weights, bars, total_equity)
            self.last_decision_context = (
                f"上期({date}): 配置={target_weights}, "
                f"总资产=¥{total_equity:,.0f}"
            )

    def _quant_fallback(self, metrics: Dict[str, Dict]) -> Dict[str, float]:
        valid = {s: m for s, m in metrics.items() if m.get("efficiency", 0) > 0.3}
        if not valid:
            return {}
        sorted_syms = sorted(valid, key=lambda s: valid[s]["efficiency"], reverse=True)
        selected = sorted_syms[:self.top_n]
        w = round(0.9 / len(selected), 4)
        return {s: w for s in selected}

    def _run_graph(self, initial_state: dict) -> dict:
        async def _go():
            return await asyncio.wait_for(
                self.app.ainvoke(initial_state),
                timeout=self.graph_timeout,
            )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _go()).result(timeout=self.graph_timeout + 10)
        return asyncio.run(_go())

    def _execute_trades(self, date, target_weights: Dict[str, float], bars, total_equity: float):
        if not target_weights:
            return

        logger.info(f"[V2 EXECUTE] {target_weights}")
        commission_rate = self.broker.commission_rate

        current_holdings = list(self.broker.positions.keys())
        for symbol in current_holdings:
            if symbol not in target_weights or target_weights.get(symbol, 0) == 0:
                pos = self.broker.positions.get(symbol)
                if pos is None:
                    continue
                qty = getattr(pos, 'quantity', pos)
                price = bars.get(symbol, {}).get('close', 0)
                if isinstance(price, pd.Series):
                    price = float(price.iloc[0]) if not price.empty else 0
                if qty > 0 and price > 0:
                    self.broker.sell(date, symbol, price, qty)
                    logger.info(f"  SELL ALL {symbol}: {qty}股 @ {price}")

        for symbol in current_holdings:
            tw = target_weights.get(symbol, 0)
            if tw <= 0:
                continue
            pos = self.broker.positions.get(symbol)
            if pos is None:
                continue
            current_qty = getattr(pos, 'quantity', pos)
            price = bars.get(symbol, {}).get('close', 0)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty else 0
            if price <= 0:
                continue
            target_val = total_equity * tw
            target_qty = int(target_val / price / self.lot_size) * self.lot_size
            if current_qty > target_qty:
                sell_qty = ((current_qty - target_qty) // self.lot_size) * self.lot_size
                if sell_qty > 0:
                    self.broker.sell(date, symbol, price, sell_qty)
                    logger.info(f"  REDUCE {symbol}: -{sell_qty}股")

        available_cash = self.broker.cash
        for symbol, weight in sorted(target_weights.items(), key=lambda x: -x[1]):
            if weight <= 0 or symbol not in bars:
                continue
            price = bars[symbol].get('close', 0)
            if isinstance(price, pd.Series):
                price = float(price.iloc[0]) if not price.empty else 0
            if price <= 0:
                continue

            pos = self.broker.positions.get(symbol)
            current_qty = 0
            if pos is not None:
                current_qty = getattr(pos, 'quantity', pos)

            target_val = total_equity * weight
            target_qty = int(target_val / price / self.lot_size) * self.lot_size

            if target_qty > current_qty:
                buy_qty = target_qty - current_qty
                cost = buy_qty * price * (1 + commission_rate)
                if cost <= available_cash:
                    self.broker.buy(date, symbol, price, buy_qty)
                    available_cash -= cost
                    logger.info(f"  BUY {symbol}: +{buy_qty}股 @ {price}")
                else:
                    affordable_qty = int(available_cash / (price * (1 + commission_rate))
                                         / self.lot_size) * self.lot_size
                    if affordable_qty > 0:
                        cost2 = affordable_qty * price * (1 + commission_rate)
                        self.broker.buy(date, symbol, price, affordable_qty)
                        available_cash -= cost2
                        logger.info(f"  BUY {symbol}: +{affordable_qty}股 (部分成交)")
