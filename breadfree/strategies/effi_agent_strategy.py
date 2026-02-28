"""
EffiAgentRotationStrategy (EffiA) — 量化锚定 LLM 效率轮动

与 AgentStrategyV2 共享"量化选股, LLM 定权"理念, 但架构更轻量:
    DataPrep: 计算效率分, 筛选 Top-3 候选 + 持仓标的
    Analyst:  LLM 在候选池内分配权重 (不可引入新标的)
    RiskMgr:  LLM 风控微调 (不可引入新标的)

改进点:
    1. 强制多标的输出 (2-3 只), 禁止单标的 100% 集中
    2. Analyst 输出包含候选池所有标的权重, 避免遗漏
    3. RiskMgr 输出强制映射到候选代码, 修复标的幻觉
    4. 佣金感知下单
"""

import asyncio
import json
import time
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from .base_strategy import BreadFreeStrategy
from ..utils.llm_client import async_hunyuan_chat, parse_llm_response
from ..utils.metrics import calculate_efficiency_metrics
from ..utils.portfolio import normalize_weights
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════

ANALYST_PROMPT = """\
你是进攻型量化ETF分析师。效率分=alpha信号，你应该重仓最强标的。
量化引擎已筛选出效率分最高的候选:
{market_data_summary}

【原则】效率分排名第1的标的就是最强的，给它最高权重。不要为了"分散"而削弱最强标的。
【约束】
- 只能在上方候选分配，不可引入新标的
- 至少2只标的有权重，效率分最高者权重 45-60%
- 总投资 93%-100%（满仓为王，少留现金）
- 持仓标的(标记[HOLDING])仍在候选中 → 优先保留不换
【输出】纯JSON:
{{
    "{example}": {{"weight": 0.55, "reason": "10字内"}},
    ...
}}"""

RISK_MGR_PROMPT = """\
你是风控审核官。审核策略师方案，默认放行，仅在极端风险时微调。
方案: {analyst_proposal}
数据: {metrics_summary}
组合: 现金{cash}, 持仓{positions}

【审核原则】
- 默认放行策略师方案（效率分排名是量化引擎的最优解）
- 仅当波动率>3.5%且效率分<0时，才减该标的5%
- 不可增加新标的，总投资 93%-100%
- 调整幅度不超过±5%
【输出】纯JSON:
{{"target_weights": {{"代码": 0.xx}}, "note": "10字内"}}"""


class AgentState(TypedDict):
    date: str
    bars: Dict[str, Any]
    history_snapshot: Dict[str, List[float]]
    metrics: Dict[str, Dict[str, float]]
    candidates: List[str]
    cash: float
    positions: Dict[str, Any]
    analyst_output: Dict[str, Any]
    risk_output: Dict[str, Any]
    lot_size: int
    target_weights: Dict[str, float]
    llm_calls: List[Dict[str, Any]]


# ═══════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════

def data_prep_node(state: AgentState) -> dict:
    history = state.get("history_snapshot", {})
    positions = state.get("positions", {})

    metrics = {}
    for sym, prices in history.items():
        m = calculate_efficiency_metrics(prices, lookback=20)
        if m:
            metrics[sym] = m

    ranked = sorted(metrics.keys(), key=lambda x: metrics[x]["efficiency"], reverse=True)
    top_n = min(len(ranked), 3)
    top_picks = ranked[:top_n]
    held = [s for s, q in positions.items() if q > 0]

    candidates = list(top_picks)
    for s in held:
        if s in ranked[:top_n + 2] and s not in candidates:
            candidates.append(s)

    filtered = {}
    for s in candidates:
        if s in metrics:
            filtered[s] = dict(metrics[s])
            filtered[s]["is_holding"] = s in held
            if s in held:
                filtered[s]["holding_qty"] = positions[s]

    logger.info(f"[DataPrep] candidates={candidates}, held_extra={[s for s in held if s not in top_picks]}")
    return {"metrics": filtered, "candidates": candidates}


async def analyst_node(state: AgentState) -> dict:
    metrics = state.get("metrics", {})
    candidates = state.get("candidates", [])
    if not metrics or not candidates:
        return {"analyst_output": {}, "llm_calls": state.get("llm_calls", [])}

    sorted_items = sorted(metrics.items(), key=lambda x: x[1].get("efficiency", 0), reverse=True)
    lines = []
    for sym, d in sorted_items:
        held = f" [HOLDING: {d['holding_qty']}]" if d.get("is_holding") else ""
        lines.append(f"- {sym}: Return={d['momentum']:.2%}, Vol={d['volatility']:.2%}, "
                     f"R2={d['r2']:.2f}, Efficiency={d['efficiency']:.2f}{held}")
    summary = "\n".join(lines)

    prompt = ANALYST_PROMPT.format(
        market_data_summary=summary,
        example=candidates[0] if candidates else "510300",
    )

    n = len(candidates)
    base_w = round(0.98 / max(n, 2), 2)
    fb = {s: {"weight": base_w, "reason": "等权 fallback"} for s in candidates}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="在候选池内分配权重。", prompt=prompt,
            temperature=0.3, max_tokens=512, timeout_seconds=60, max_retries=2)
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(resp, fb)
        used_fb = result is fb
        llm_calls.append({"agent": "analyst", "tokens": tokens,
                          "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        logger.error(f"[Analyst ERROR] {e}")
        result = fb
        llm_calls.append({"agent": "analyst", "tokens": 0,
                          "latency_ms": int(time.time() * 1000) - t0,
                          "error": str(e), "used_fallback": True})

    # 提取并约束权重: 只保留候选池标的, 单只≤60%, 至少2只有权重
    cand_set = set(candidates)
    weights = {}
    for s, v in result.items():
        if s in cand_set:
            w = v.get("weight", 0) if isinstance(v, dict) else (v if isinstance(v, (int, float)) else 0)
            weights[s] = max(min(float(w), 0.60), 0)

    # 确保至少2只标的有权重
    if sum(1 for w in weights.values() if w > 0.05) < 2:
        weights = {s: base_w for s in candidates}

    total = sum(weights.values())
    if total > 0:
        scale = min(1.0, max(0.93, total)) / total
        weights = {s: round(v * scale, 4) for s, v in weights.items() if v > 0}

    logger.info(f"[Analyst] {weights}")
    return {"analyst_output": weights, "llm_calls": llm_calls}


async def risk_mgr_node(state: AgentState) -> dict:
    a_weights = state.get("analyst_output", {})
    candidates = state.get("candidates", [])
    if not a_weights:
        return {"risk_output": {}, "target_weights": {},
                "llm_calls": state.get("llm_calls", [])}

    metrics = state.get("metrics", {})
    lines = []
    for s in candidates:
        if s in metrics:
            d = metrics[s]
            lines.append(f"{s}: Eff={d['efficiency']:.2f} Vol={d['volatility']:.2%}")
    metrics_str = "; ".join(lines)

    prompt = RISK_MGR_PROMPT.format(
        analyst_proposal=json.dumps(a_weights),
        metrics_summary=metrics_str,
        cash=f"{state['cash']:.0f}",
        positions=str(state["positions"]),
    )

    fallback = {"target_weights": a_weights, "note": "直接采纳"}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="风控审核。", prompt=prompt,
            temperature=0.15, max_tokens=300, timeout_seconds=60, max_retries=2)
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(resp, fallback)
        used_fb = not result.get("target_weights")
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

    raw = result.get("target_weights", {})
    cand_set = set(candidates)
    tw = {}
    for k, v in raw.items():
        sym = k.split("-")[0] if "-" in k else k
        if sym in cand_set and isinstance(v, (int, float)):
            tw[sym] = max(v, 0)

    if not tw or sum(tw.values()) < 0.5:
        tw = a_weights

    tw = normalize_weights(tw)
    logger.info(f"[RiskMgr] {tw} | {result.get('note', '')}")
    return {"risk_output": result, "target_weights": tw, "llm_calls": llm_calls}


def build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("data_prep", data_prep_node)
    wf.add_node("analyst", analyst_node)
    wf.add_node("risk_mgr", risk_mgr_node)
    wf.set_entry_point("data_prep")
    wf.add_edge("data_prep", "analyst")
    wf.add_edge("analyst", "risk_mgr")
    wf.add_edge("risk_mgr", END)
    return wf.compile()


# ═══════════════════════════════════════════════════════════════
# Strategy Class
# ═══════════════════════════════════════════════════════════════

class EffiAgentRotationStrategy(BreadFreeStrategy):
    """量化锚定 LLM 效率轮动"""

    def __init__(self, broker, lookback_period=20, hold_period=20, lot_size=100,
                 graph_timeout_seconds=180, audit_logger=None, **kwargs):
        super().__init__(broker, lot_size=lot_size)
        self.lookback_period = lookback_period
        self.hold_period = hold_period
        self.days_counter = 0
        self.graph_timeout = graph_timeout_seconds
        self.audit_logger = audit_logger
        self.app = build_graph()
        self._total_llm_calls = 0
        self._total_llm_tokens = 0
        self._total_fallbacks = 0

    def on_bar(self, date, bars):
        for sym, bar in bars.items():
            self.history.setdefault(sym, []).append(bar["close"])
        self.days_counter += 1

        pool = list(set(list(self.broker.positions.keys()) + list(bars.keys())))
        if any(len(self.history.get(s, [])) < self.lookback_period for s in pool):
            return
        if self.days_counter % self.hold_period != 0 and self.days_counter > 1:
            return

        logger.info(f"[EffiA] {date}")

        pos_snap = {s: getattr(self.broker.positions[s], "quantity",
                                self.broker.positions[s])
                    for s in self.broker.positions}

        state = {
            "date": str(date), "bars": bars,
            "history_snapshot": {s: list(self.history[s]) for s in bars},
            "metrics": {}, "candidates": [],
            "cash": self.broker.cash, "positions": pos_snap,
            "analyst_output": {}, "risk_output": {},
            "lot_size": self.lot_size, "target_weights": {},
            "llm_calls": [],
        }

        try:
            final = self._run(state)
            tw = final.get("target_weights", {})
            for c in final.get("llm_calls", []):
                self._total_llm_calls += 1
                self._total_llm_tokens += c.get("tokens", 0)
                if c.get("used_fallback"):
                    self._total_fallbacks += 1
            logger.info(f"[EffiA DONE] {tw}")
            self._trade(date, tw, bars)
        except Exception as e:
            logger.error(f"[EffiA ERROR] {e}", exc_info=True)

    def _run(self, state):
        async def _go():
            return await asyncio.wait_for(self.app.ainvoke(state), timeout=self.graph_timeout)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _go()).result(timeout=self.graph_timeout + 10)
        return asyncio.run(_go())

    def _trade(self, date, tw: Dict[str, float], bars):
        if not tw:
            return

        total = self.broker.cash
        for s, pos in self.broker.positions.items():
            p = bars.get(s, {}).get("close", 0)
            q = getattr(pos, "quantity", pos) if hasattr(pos, "quantity") else pos
            if p > 0:
                total += q * p

        cr = self.broker.commission_rate

        for sym in list(self.broker.positions.keys()):
            pos = self.broker.positions[sym]
            cur = getattr(pos, "quantity", pos) if hasattr(pos, "quantity") else pos
            price = bars.get(sym, {}).get("close", 0)
            if price == 0:
                continue
            tgt_q = int(total * tw.get(sym, 0) / price / self.lot_size) * self.lot_size
            if cur > tgt_q:
                self.broker.sell(date, sym, price, cur - tgt_q)

        cash = self.broker.cash
        for sym, w in sorted(tw.items(), key=lambda x: -x[1]):
            if sym not in bars:
                continue
            price = bars[sym]["close"]
            if price <= 0:
                continue
            pos = self.broker.positions.get(sym)
            cur = getattr(pos, "quantity", pos) if pos and hasattr(pos, "quantity") else (pos or 0)
            tgt_q = int(total * w / price / self.lot_size) * self.lot_size
            if tgt_q > cur:
                buy_q = tgt_q - cur
                cost = buy_q * price * (1 + cr)
                if cost > cash:
                    buy_q = int(cash / (price * (1 + cr)) / self.lot_size) * self.lot_size
                    cost = buy_q * price * (1 + cr)
                if buy_q > 0 and cost <= cash:
                    self.broker.buy(date, sym, price, buy_q)
                    cash -= cost

    def get_llm_stats(self) -> dict:
        return {"calls": self._total_llm_calls, "tokens": self._total_llm_tokens,
                "fallbacks": self._total_fallbacks}
