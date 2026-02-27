"""
EffiAgentRotationStrategy — LLM 效率轮动策略 (EffiA)

架构: QuantPrep (纯计算) → Analyst Agent (LLM) → RiskMgr Agent (LLM)
- QuantPrep: 效率分/多周期动量 筛选 Top-N 候选
- Analyst: LLM 分析选股 + 权重分配 (结构化JSON)
- RiskMgr: LLM 风控微调, 输出最终目标权重

仅在调仓日调 LLM, 其余交易日跳过.
LLM 全部失败时自动 fallback 为纯效率轮动.
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
# Prompts — 量化投研框架
# ═══════════════════════════════════════════════════════════════

ANALYST_PROMPT = """
你是一位资深量化ETF分析师。根据下方技术指标从候选池中精选最优标的。

【策略定义】
- 选择具有高动量(Return)和高趋势稳定性(Efficiency/R²)的ETF
- 波动率高但收益不对等的标的应回避
- 可选0到N只标的 — 只有一只强势就集中配置, 无合适标的就空仓
- 当前持仓(标记[HOLDING])仍在Top候选中时优先保留, 减少换手

【候选池数据(20日)】
{market_data_summary}

【任务】分析候选并选择持有标的, 输出纯JSON:
{{
    "代码": {{"weight": 0.0-1.0, "view": "bullish/neutral/bearish", "reason": "20字内理由"}},
    ...
}}
"""

RISK_MANAGER_PROMPT = """
你是风控管理官。策略师提交了如下分析: {analyst_proposal}

【当前组合状态】
- 现金: {cash}
- 持仓: {positions}

【任务】
1. 审核策略师对每只标的的观点
2. 确定最终目标权重(总和≤1.0), 低于1.0表示持有现金
3. bearish标的权重应为0
4. 输出格式(纯JSON):
{{
    "target_weights": {{"代码": 0.0-1.0}},
    "risk_comment": "20字内风控意见",
    "approved": true
}}
"""

# ═══════════════════════════════════════════════════════════════
# State & Helper
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    date: str
    bars: Dict[str, Any]
    history_snapshot: Dict[str, List[float]]
    metrics: Dict[str, Dict[str, float]]
    broker_state: str
    cash: float
    positions: Dict[str, Any]
    analyst_output: Dict[str, Any]
    risk_output: Dict[str, Any]
    lot_size: int
    target_weights: Dict[str, float]
    llm_calls: List[Dict[str, Any]]


def get_fallback_weights(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    valid = {s: m for s, m in metrics.items() if m["efficiency"] > 0}
    if not valid:
        return {}
    ranked = sorted(valid.keys(), key=lambda x: valid[x]["efficiency"], reverse=True)
    sel = ranked[:3]
    w = round(0.95 / len(sel), 4)
    return {s: w for s in sel}


# ═══════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════

def data_prep_node(state: AgentState) -> AgentState:
    history = state.get("history_snapshot", {})
    positions = state.get("positions", {})

    metrics_map = {}
    for symbol, prices in history.items():
        m = calculate_efficiency_metrics(prices, lookback=20)
        if m:
            metrics_map[symbol] = m

    ranked = sorted(metrics_map.keys(),
                    key=lambda x: metrics_map[x]["efficiency"], reverse=True)
    top_3 = ranked[:3]
    current_holdings = [s for s, qty in positions.items() if qty > 0]
    selected = top_3 + [s for s in current_holdings if s not in top_3]
    filtered = {k: metrics_map[k] for k in selected if k in metrics_map}

    for symbol in filtered:
        is_held = symbol in current_holdings
        filtered[symbol]["is_current_holding"] = is_held
        if is_held:
            filtered[symbol]["holding_qty"] = positions[symbol]

    logger.info(f"[DataPrep] Top 3: {top_3}, holdings: "
                f"{[s for s in current_holdings if s not in top_3]}")
    return {"metrics": filtered}


async def analyst_agent_node(state: AgentState) -> AgentState:
    metrics = state["metrics"]
    if not metrics:
        return {"analyst_output": {}, "llm_calls": state.get("llm_calls", [])}

    sorted_items = sorted(metrics.items(),
                          key=lambda x: x[1].get("efficiency", 0), reverse=True)
    lines = []
    for sym, d in sorted_items:
        held = f" [HOLDING: {d['holding_qty']}]" if d.get("is_current_holding") else ""
        lines.append(f"- {sym}: Return={d['momentum']:.2%}, "
                     f"Vol={d['volatility']:.2%}, R2={d['r2']:.2f}, "
                     f"Efficiency={d['efficiency']:.2f}{held}")
    summary = "\n".join(lines)
    logger.info(f"[Analyst] Data:\n{summary}")
    prompt = ANALYST_PROMPT.format(market_data_summary=summary)

    fb = {}
    fb_w = get_fallback_weights(metrics)
    for s, w in fb_w.items():
        fb[s] = {"weight": w, "view": "bullish", "reason": "top efficiency"}

    llm_calls = list(state.get("llm_calls", []))
    used_fb = False
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="分析候选并选择最优ETF。", prompt=prompt,
            temperature=0.3, max_tokens=1024, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(resp, fb)
        if resp == "" or result is fb:
            used_fb = True
        llm_calls.append({"agent": "analyst", "tokens": tokens,
                          "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        lat = int(time.time() * 1000) - t0
        logger.error(f"[Analyst ERROR] {e}")
        result = fb
        used_fb = True
        llm_calls.append({"agent": "analyst", "tokens": 0,
                          "latency_ms": lat, "error": str(e), "used_fallback": True})

    proposed = {s: (v.get("weight") if isinstance(v, dict) else 0)
                for s, v in result.items()}
    logger.info(f"[Analyst] Proposed: {proposed}"
                f"{' (FALLBACK)' if used_fb else ''}")
    return {"analyst_output": result, "llm_calls": llm_calls}


async def risk_manager_node(state: AgentState) -> AgentState:
    analyst = state.get("analyst_output", {})
    if not analyst:
        return {"risk_output": {"target_weights": {}, "approved": True},
                "target_weights": {}, "llm_calls": state.get("llm_calls", [])}

    prompt = RISK_MANAGER_PROMPT.format(
        analyst_proposal=json.dumps(analyst),
        cash=f"{state['cash']:.2f}",
        positions=str(state["positions"]),
    )
    fb_w = get_fallback_weights(state.get("metrics", {}))
    fb = {"target_weights": fb_w, "approved": True}

    llm_calls = list(state.get("llm_calls", []))
    used_fb = False
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="审核分析并分配权重。", prompt=prompt,
            temperature=0.15, max_tokens=600, timeout_seconds=60, max_retries=2,
        )
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(resp, fb)
        if resp == "" or result is fb:
            used_fb = True
        llm_calls.append({"agent": "risk_mgr", "tokens": tokens,
                          "latency_ms": lat, "used_fallback": used_fb})
    except Exception as e:
        lat = int(time.time() * 1000) - t0
        logger.error(f"[RiskMgr ERROR] {e}")
        result = fb
        used_fb = True
        llm_calls.append({"agent": "risk_mgr", "tokens": 0,
                          "latency_ms": lat, "error": str(e), "used_fallback": True})

    tw = result.get("target_weights", {})
    if not tw or sum(tw.values()) <= 0:
        tw = fb_w
        used_fb = True

    tw = normalize_weights(tw)
    result["target_weights"] = tw
    logger.info(f"[RiskMgr] Final: {tw}{' (FALLBACK)' if used_fb else ''}")
    return {"risk_output": result, "target_weights": tw, "llm_calls": llm_calls}


# ═══════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════

def build_agent_graph():
    wf = StateGraph(AgentState)
    wf.add_node("data_prep", data_prep_node)
    wf.add_node("analyst", analyst_agent_node)
    wf.add_node("risk_manager", risk_manager_node)
    wf.set_entry_point("data_prep")
    wf.add_edge("data_prep", "analyst")
    wf.add_edge("analyst", "risk_manager")
    wf.add_edge("risk_manager", END)
    return wf.compile()


# ═══════════════════════════════════════════════════════════════
# Strategy Class
# ═══════════════════════════════════════════════════════════════

class EffiAgentRotationStrategy(BreadFreeStrategy):
    """LLM 效率轮动策略 — DataPrep → Analyst → RiskMgr"""

    def __init__(self, broker, lookback_period=20, hold_period=20, lot_size=100,
                 graph_timeout_seconds=180, audit_logger=None, **kwargs):
        super().__init__(broker, lot_size=lot_size)
        self.lookback_period = lookback_period
        self.hold_period = hold_period
        self.days_counter = 0
        self.graph_timeout = graph_timeout_seconds
        self.audit_logger = audit_logger
        self.app = build_agent_graph()
        self._total_llm_calls = 0
        self._total_llm_tokens = 0
        self._total_fallbacks = 0

    def on_bar(self, date, bars):
        for symbol, bar in bars.items():
            if symbol not in self.history:
                self.history[symbol] = []
            self.history[symbol].append(bar["close"])
        self.days_counter += 1

        not_ready = [s for s in self._positions_and_pool(bars)
                     if len(self.history.get(s, [])) < self.lookback_period]
        if not_ready:
            return

        if self.days_counter % self.hold_period != 0 and self.days_counter > 1:
            return

        logger.info(f"[EffiA] Agents triggered on {date}")

        history_snap = {s: list(self.history[s]) for s in bars.keys()}
        pos_snap = {s: getattr(self.broker.positions[s], "quantity",
                                self.broker.positions[s])
                    if s in self.broker.positions else 0
                    for s in self.broker.positions}

        initial_state = {
            "date": str(date), "bars": bars,
            "history_snapshot": history_snap, "metrics": {},
            "cash": self.broker.cash, "positions": pos_snap,
            "analyst_output": {}, "risk_output": {},
            "lot_size": self.lot_size, "target_weights": {},
            "llm_calls": [],
        }

        graph_start = time.time()
        try:
            final = self._run_graph(initial_state)
            elapsed = time.time() - graph_start
            tw = final.get("target_weights", {})
            calls = final.get("llm_calls", [])

            logger.info(f"[EffiA DONE] weights={tw} time={elapsed:.1f}s calls={len(calls)}")
            self._audit(calls, date, tw)
            self._execute_trades(date, tw, bars)

        except asyncio.TimeoutError:
            elapsed = time.time() - graph_start
            logger.error(f"[EffiA TIMEOUT] {elapsed:.1f}s > {self.graph_timeout}s")
            fb = get_fallback_weights(
                final.get("metrics", {}) if 'final' in dir() else {})
            if fb:
                fb = normalize_weights(fb)
                self._execute_trades(date, fb, bars)
            self._total_fallbacks += 1
        except Exception as e:
            logger.error(f"[EffiA ERROR] {e}", exc_info=True)

    def _run_graph(self, state: dict) -> dict:
        async def _run():
            return await asyncio.wait_for(
                self.app.ainvoke(state), timeout=self.graph_timeout)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _run()).result(
                    timeout=self.graph_timeout + 10)
        return asyncio.run(_run())

    def _audit(self, calls: list, date, tw: dict):
        for c in calls:
            self._total_llm_calls += 1
            self._total_llm_tokens += c.get("tokens", 0)
            if c.get("used_fallback"):
                self._total_fallbacks += 1
            if self.audit_logger:
                agent = c.get("agent", "unknown")
                self.audit_logger.log_llm_call(
                    model=f"agent:{agent}",
                    prompt_summary=f"{agent} on {date}",
                    response_summary=f"fallback={c.get('used_fallback', False)}",
                    tokens_used=c.get("tokens", 0),
                    latency_ms=c.get("latency_ms", 0),
                )

    def _positions_and_pool(self, bars):
        return list(set(list(self.broker.positions.keys()) + list(bars.keys())))

    def _execute_trades(self, date, tw: Dict[str, float], bars):
        if not tw:
            return
        logger.info(f"[EXEC] {tw}")

        total_val = self.broker.cash
        for s, pos in self.broker.positions.items():
            price = bars.get(s, {}).get("close", 0)
            qty = getattr(pos, "quantity", pos) if hasattr(pos, "quantity") else pos
            if price > 0:
                total_val += qty * price

        cr = self.broker.commission_rate

        # 先卖
        for symbol in list(self.broker.positions.keys()):
            pos_obj = self.broker.positions[symbol]
            cur_qty = getattr(pos_obj, "quantity", pos_obj) if hasattr(pos_obj, "quantity") else pos_obj
            price = bars.get(symbol, {}).get("close", 0)
            if price == 0:
                continue
            target_pct = tw.get(symbol, 0.0)
            target_qty = int(total_val * target_pct / price / self.lot_size) * self.lot_size
            if cur_qty > target_qty:
                sell_qty = cur_qty - target_qty
                self.broker.sell(date, symbol, price, sell_qty)

        # 后买 (佣金感知)
        cash = self.broker.cash
        for symbol, weight in sorted(tw.items(), key=lambda x: -x[1]):
            if symbol not in bars:
                continue
            price = bars[symbol]["close"]
            if price <= 0:
                continue
            pos_obj = self.broker.positions.get(symbol)
            cur_qty = getattr(pos_obj, "quantity", pos_obj) if pos_obj and hasattr(pos_obj, "quantity") else (pos_obj or 0)
            target_qty = int(total_val * weight / price / self.lot_size) * self.lot_size
            if target_qty > cur_qty:
                buy_qty = target_qty - cur_qty
                cost = buy_qty * price * (1 + cr)
                if cost > cash:
                    buy_qty = int(cash / (price * (1 + cr)) / self.lot_size) * self.lot_size
                    cost = buy_qty * price * (1 + cr)
                if buy_qty > 0 and cost <= cash:
                    self.broker.buy(date, symbol, price, buy_qty)
                    cash -= cost

    def get_llm_stats(self) -> dict:
        return {
            "total_llm_calls": self._total_llm_calls,
            "total_llm_tokens": self._total_llm_tokens,
            "total_fallbacks": self._total_fallbacks,
        }
