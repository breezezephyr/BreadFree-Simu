"""
AgentStrategyV2 — 量化锚定 + LLM 精调决策系统

核心理念: "量化选股, LLM 定权"
    QuantPrep 选出 Top-N 候选 (不可更改)
    Analyst LLM 在 Top-N 内分配权重 (可集中/分散, 但不能引入新标的)
    RiskMgr LLM 微调权重 (可减仓+留现金, 但不能引入新标的)

为何不让 LLM 自由选股:
    - 效率分/动量等量化因子已高度提炼 alpha 信号
    - LLM 自由选股会引入"防御偏差"(过度配置债券/红利), 拖累趋势收益
    - LLM 的真正价值在于: 理解因子间微妙关系, 做出更精准的权重分配

架构改进 (V2.1):
    1. 量化锚定: LLM 只能在 QuantPrep 的 Top-N 内分配权重
    2. 基准权重: 提供等权基准, LLM 在此基础上调整 ±20%
    3. 最低投资度: 总投资比例不低于 85% (杜绝过度保守)
    4. 决策记忆: 传递上期收益反馈, 形成学习闭环
    5. 多周期因子: 5d/10d/20d 动量一致性 + 加速度
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
# 量化引擎
# ═══════════════════════════════════════════════════════════════

def compute_metrics(history: List[float], lookback: int = 20) -> Optional[Dict]:
    if len(history) < lookback + 10:
        return None
    prices = np.array(history[-(lookback + 10):])
    cur = prices[-lookback:]
    prev = prices[-(lookback + 5):-5]
    if cur[0] <= 0 or prev[0] <= 0:
        return None

    mom_20d = cur[-1] / cur[0] - 1
    mom_10d = cur[-1] / cur[-10] - 1 if len(cur) >= 10 else mom_20d
    mom_5d = cur[-1] / cur[-5] - 1 if len(cur) >= 5 else mom_20d
    mom_accel = mom_20d - (prev[-1] / prev[0] - 1)

    rets = np.diff(cur) / cur[:-1]
    vol = float(np.std(rets)) if len(rets) > 1 else 0.0

    try:
        slope, _, r_val, _, _ = stats.linregress(np.arange(len(cur)), cur)
        r2 = r_val ** 2
    except Exception:
        slope, r2 = 0.0, 0.0

    period_vol = vol * np.sqrt(len(rets)) + 1e-6
    eff = (mom_20d / period_vol) * r2
    dd_high = cur[-1] / np.max(cur) - 1
    alignment = sum(1 for m in [mom_5d, mom_10d, mom_20d] if m > 0) / 3.0

    return {
        "momentum_5d": float(mom_5d), "momentum_10d": float(mom_10d),
        "momentum_20d": float(mom_20d), "momentum_accel": float(mom_accel),
        "volatility": float(vol), "r2": float(r2), "efficiency": float(eff),
        "close": float(cur[-1]), "drawdown_from_high": float(dd_high),
        "trend_slope": float(slope), "momentum_alignment": float(alignment),
    }


# ═══════════════════════════════════════════════════════════════
# Prompts — 量化锚定框架
# ═══════════════════════════════════════════════════════════════

ANALYST_PROMPT = """\
你是量化投资策略师。量化引擎已从25只标的中筛选出效率分最高的{top_n}只候选。
你的任务是在这{top_n}只候选中分配投资权重。

【重要约束】
- 你只能在下方候选池中分配权重, 不可引入其他标的
- 总投资比例必须在 85%-95% 之间 (即最多持有15%现金)
- 单只标的权重范围: 15%-65%
- 基准配置是等权({base_weight:.0%}每只), 你在此基础上调整

【候选池及指标 (近{lookback}日)】
{metrics_table}

【市场情报】
{market_intel}

【市场政权】{regime}
【当前持仓】{holdings}
【上期决策回顾】{last_context}

【决策要点】
- 效率分最高+动量加速度>0+一致性↑↑↑ → 该标的可上调至50-65%
- 效率分虽高但加速度<0 → 趋势可能衰减, 权重不超过基准
- 持仓标的仍在候选池中 → 优先保留(降低换手)
- 全部候选效率分<0.3 → 降低总投资度至85%

【输出】纯JSON, 无其他文字:
{{
  "weights": {{"{example_sym}": 0.40, ...}},
  "reasoning": "20字内决策理由"
}}"""

RISK_MGR_PROMPT = """\
你是风控管理官。策略师已在量化引擎筛选的候选池内完成权重分配。
你需要审核并微调权重。

【策略师方案】
{analyst_weights}

【候选池数据】
{metrics_table}

【市场情报】
{market_intel}

【风控规则】
1. 波动率>2.5%的标的权重不超过45%
2. 加速度<-0.02且R²<0.5的标的建议减仓10%
3. 你只能调整权重, 不可增加新标的
4. 总投资比例保持 85%-95%
5. 调整幅度通常在±10%以内

【输出】纯JSON:
{{
  "final_weights": {{...}},
  "risk_note": "15字内"
}}"""


# ═══════════════════════════════════════════════════════════════
# LangGraph State & Nodes
# ═══════════════════════════════════════════════════════════════

class V2State(TypedDict):
    date: str
    bars: Dict[str, Any]
    all_metrics: Dict[str, Dict]
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


def _clean_json(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def quant_engine_node(state: V2State) -> dict:
    bars = state["bars"]
    all_m = state.get("all_metrics", {})
    top_n = state.get("top_n", 3)
    holdings = state.get("current_holdings", {})

    valid = {s: m for s, m in all_m.items() if s in bars}
    if not valid:
        return {"top_candidates": [], "regime": "unknown",
                "metrics_table_str": "", "market_intel_str": ""}

    ranked = sorted(valid, key=lambda s: valid[s]["efficiency"], reverse=True)

    # 量化锚定: 候选池 = Top-N (+ 仍在 Top-N+2 内的持仓)
    candidates = ranked[:top_n]
    for s in holdings:
        if s in ranked[:top_n + 2] and s not in candidates:
            candidates.append(s)

    lines = []
    for sym in candidates:
        m = valid[sym]
        tag = " ★持仓" if sym in holdings else ""
        rank = ranked.index(sym) + 1
        ali = "↑↑↑" if m["momentum_alignment"] >= 0.9 else ("↑↑" if m["momentum_alignment"] >= 0.6 else "↑↓")
        lines.append(
            f"  #{rank} {_n(sym)}: 效率={m['efficiency']:.2f}, "
            f"20d={m['momentum_20d']:.2%}, 10d={m['momentum_10d']:.2%}, 5d={m['momentum_5d']:.2%}, "
            f"加速={m['momentum_accel']:+.2%}, R²={m['r2']:.2f}, "
            f"波动={m['volatility']:.2%}, 离高点={m['drawdown_from_high']:.2%}, {ali}{tag}")
    table = "\n".join(lines)

    date_ts = pd.Timestamp(state.get("date", ""))
    regime = _intel.get_regime_enhanced(date_ts, valid)
    intel = _intel.generate_intel_summary(date_ts, valid, candidates[:top_n])

    logger.info(f"[Quant] regime={regime} candidates={[_n(s) for s in candidates]}\n{table}")
    return {"top_candidates": candidates, "regime": regime,
            "metrics_table_str": table, "market_intel_str": intel}


async def analyst_node(state: V2State) -> dict:
    candidates = state.get("top_candidates", [])
    if not candidates:
        return {"analyst_output": {}, "llm_calls": state.get("llm_calls", [])}

    top_n = len(candidates)
    base_w = round(0.95 / top_n, 2)

    prompt = ANALYST_PROMPT.format(
        top_n=top_n, lookback=state.get("lookback", 20),
        metrics_table=state["metrics_table_str"],
        market_intel=state.get("market_intel_str", "暂无"),
        regime=state["regime"],
        holdings=state.get("current_holdings") or "无",
        last_context=state.get("last_decision_context") or "首次决策",
        base_weight=base_w,
        example_sym=candidates[0] if candidates else "510300",
    )

    # fallback = 等权
    fb_w = {s: base_w for s in candidates}
    fallback = {"weights": fb_w, "reasoning": "quant fallback 等权"}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="在候选池内分配权重。", prompt=prompt,
            temperature=0.3, max_tokens=512, timeout_seconds=60, max_retries=2)
        lat = int(time.time() * 1000) - t0
        result = parse_llm_response(_clean_json(resp), fallback)
        used_fb = not result.get("weights")
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

    # 强制约束: 只保留候选池内的标的
    raw_w = result.get("weights", {})
    w = {s: max(min(raw_w.get(s, 0), 0.65), 0) for s in candidates}
    # 补齐: 如果 LLM 漏掉某些候选, 给最低 15%
    for s in candidates:
        if w.get(s, 0) < 0.05:
            w[s] = 0.15
    total = sum(w.values())
    if total > 0:
        scale = min(0.95, max(0.85, total)) / total
        w = {s: round(v * scale, 4) for s, v in w.items()}

    w_named = {_n(s): f"{v:.0%}" for s, v in w.items() if v > 0}
    logger.info(f"[Analyst] {w_named} | {result.get('reasoning', '')}")
    return {"analyst_output": {"weights": w, "reasoning": result.get("reasoning", "")},
            "llm_calls": llm_calls}


async def risk_mgr_node(state: V2State) -> dict:
    analyst = state.get("analyst_output", {})
    a_weights = analyst.get("weights", {})
    if not a_weights:
        return {"risk_output": {}, "target_weights": {},
                "llm_calls": state.get("llm_calls", [])}

    prompt = RISK_MGR_PROMPT.format(
        analyst_weights=json.dumps({_n(s): f"{v:.0%}" for s, v in a_weights.items()},
                                   ensure_ascii=False),
        metrics_table=state["metrics_table_str"],
        market_intel=state.get("market_intel_str", "暂无"),
    )

    fallback = {"final_weights": a_weights, "risk_note": "直接采纳"}

    llm_calls = list(state.get("llm_calls", []))
    t0 = int(time.time() * 1000)
    try:
        resp, tokens = await async_hunyuan_chat(
            query="风控审核并微调权重。", prompt=prompt,
            temperature=0.15, max_tokens=400, timeout_seconds=60, max_retries=2)
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

    candidates = set(state.get("top_candidates", []))
    raw_tw = result.get("final_weights", {})

    # 规范化: 去掉中文名后缀, 只保留候选池标的
    tw = {}
    for k, v in raw_tw.items():
        sym = k.split("-")[0] if "-" in k else k
        if sym in candidates and isinstance(v, (int, float)):
            tw[sym] = max(v, 0)

    # 如果 RiskMgr 输出垃圾, 回退到 Analyst 方案
    if not tw or sum(tw.values()) < 0.5:
        tw = a_weights

    total = sum(tw.values())
    if total > 0.98:
        s = 0.95 / total
        tw = {k: round(v * s, 4) for k, v in tw.items()}
    tw = {k: v for k, v in tw.items() if v >= 0.02}

    tw_named = {_n(s): f"{v:.0%}" for s, v in tw.items()}
    logger.info(f"[RiskMgr] {tw_named} | {result.get('risk_note', '')}")
    return {"risk_output": result, "target_weights": tw, "llm_calls": llm_calls}


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
    """量化锚定 + LLM 精调: QuantPrep → Analyst → RiskMgr"""

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
        for sym, bar in bars.items():
            self.history.setdefault(sym, []).append(bar["close"])
        self.days_counter += 1

        min_len = self.lookback_period + 10
        if not all(len(self.history.get(s, [])) >= min_len for s in bars):
            return
        if self.days_counter % self.hold_period != 0 and self.days_counter > 1:
            return

        all_metrics = {}
        for sym in bars:
            m = compute_metrics(self.history.get(sym, []), self.lookback_period)
            if m:
                all_metrics[sym] = m
        if not all_metrics:
            return

        pos_snap = {s: getattr(p, "quantity", p) for s, p in self.broker.positions.items()}
        equity = self.broker.cash + sum(
            qty * float(bars[s]["close"]) for s, qty in pos_snap.items() if s in bars)

        pnl = f", 本期={equity / self._last_equity - 1:+.2%}" if self._last_equity > 0 else ""

        state: V2State = {
            "date": str(date), "bars": bars, "all_metrics": all_metrics,
            "regime": "", "top_candidates": [], "current_holdings": pos_snap,
            "cash": self.broker.cash, "total_equity": equity,
            "lookback": self.lookback_period, "top_n": self.top_n,
            "last_decision_context": self.last_decision_context,
            "metrics_table_str": "", "market_intel_str": "",
            "analyst_output": {}, "risk_output": {},
            "target_weights": {}, "llm_calls": [],
        }

        logger.info(f"\n{'=' * 55}\n[V2] {date} | ¥{equity:,.0f}{pnl} | "
                    f"持仓{[_n(s) for s in pos_snap]}")

        tw = {}
        try:
            final = self._run_graph(state)
            tw = final.get("target_weights", {})
            for c in final.get("llm_calls", []):
                self._total_llm_calls += 1
                self._total_llm_tokens += c.get("tokens", 0)
        except Exception as e:
            logger.error(f"[V2 ERROR] {e}")
            tw = self._fallback(all_metrics)

        if tw:
            self._execute(date, tw, bars, equity)
            self.last_decision_context = (
                f"上期({str(date)[:10]}): {json.dumps({_n(s): f'{w:.0%}' for s, w in tw.items()}, ensure_ascii=False)}"
                f", 资产¥{equity:,.0f}{pnl}")
            self._last_equity = equity

    def _fallback(self, metrics: Dict) -> Dict[str, float]:
        valid = {s: m for s, m in metrics.items() if m.get("efficiency", 0) > 0}
        if not valid:
            return {}
        ranked = sorted(valid, key=lambda s: valid[s]["efficiency"], reverse=True)
        sel = ranked[:self.top_n]
        w = round(0.95 / len(sel), 4)
        return {s: w for s in sel}

    def _run_graph(self, state: dict) -> dict:
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

    def _execute(self, date, tw: Dict[str, float], bars, equity: float):
        if not tw:
            return
        cr = self.broker.commission_rate
        logger.info(f"[EXEC] {({_n(s): f'{v:.0%}' for s, v in tw.items()})}")

        for sym in list(self.broker.positions.keys()):
            if tw.get(sym, 0) == 0:
                pos = self.broker.positions.get(sym)
                qty = getattr(pos, "quantity", pos) if pos else 0
                price = float(bars.get(sym, {}).get("close", 0))
                if qty > 0 and price > 0:
                    self.broker.sell(date, sym, price, qty)

        for sym in list(self.broker.positions.keys()):
            w = tw.get(sym, 0)
            if w <= 0:
                continue
            pos = self.broker.positions.get(sym)
            cur = getattr(pos, "quantity", pos) if pos else 0
            price = float(bars.get(sym, {}).get("close", 0))
            if price <= 0:
                continue
            tgt = int(equity * w / price / self.lot_size) * self.lot_size
            if cur > tgt:
                sell_q = ((cur - tgt) // self.lot_size) * self.lot_size
                if sell_q > 0:
                    self.broker.sell(date, sym, price, sell_q)

        cash = self.broker.cash
        for sym, w in sorted(tw.items(), key=lambda x: -x[1]):
            if w <= 0 or sym not in bars:
                continue
            price = float(bars[sym].get("close", 0))
            if price <= 0:
                continue
            pos = self.broker.positions.get(sym)
            cur = getattr(pos, "quantity", pos) if pos else 0
            tgt = int(equity * w / price / self.lot_size) * self.lot_size
            if tgt > cur:
                buy_q = tgt - cur
                cost = buy_q * price * (1 + cr)
                if cost > cash:
                    buy_q = int(cash / (price * (1 + cr)) / self.lot_size) * self.lot_size
                    cost = buy_q * price * (1 + cr)
                if buy_q > 0 and cost <= cash:
                    self.broker.buy(date, sym, price, buy_q)
                    cash -= cost
