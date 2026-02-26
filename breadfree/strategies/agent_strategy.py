import json
import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from .base_strategy import BreadFreeStrategy
from ..utils.llm_client import async_hunyuan_chat
from ..utils.logger import get_logger

logger = get_logger(__name__, mode="all")

# --- State Definition ---
class AgentState(TypedDict):
    date: str
    market_data: str
    account_status: str
    analyst_view: str
    risk_view: str
    final_decision: str # JSON string

# --- Prompts ---
ANALYST_PROMPT = """
你是一个激进的A股市场分析师，专注于政策敏感型市场。你的任务是基于共享内存中的最新数据（OHLCV、均线、真实新闻舆情、散户情绪指标）判断当前趋势。必须考虑A股T+1交易制度、涨跌停板限制和政策影响。

【事件驱动分析要求】
请对提供的【新闻舆情】进行事件分类和影响评估：
1. **Policy_Support (政策扶持)**: 如降息、行业补贴 -> 长期利好
2. **Earnings_Surprise (业绩超预期)**: 如财报预增 -> 短期爆发
3. **M_and_A (并购重组)**: 资产注入 -> 高波动
4. **Scandal (丑闻/立案)**: 违规、减持 -> **一票否决（建议卖出）**
5. **Noise (噪音)**: 无实质影响 -> 忽略

【反射检查要求】
1. 确认数据完整性：检查最近5个交易日数据是否完整
2. 评估预期差：对比【新闻事件】与【今日股价表现】。例如：利好发布但股价下跌=利好出尽；利空发布但股价不跌=底部确认。
3. 考察散户情绪：结合交易量异常波动和热门板块热度指标

输出必须包含：
1. **事件分析**：
   - 核心事件类型：(从上述5类中选择)
   - 预期差判断：(符合预期/低于预期/超预期)
2. **市场趋势**（上涨/下跌/震荡）及多维度理由：
   - 技术面：价格与均线关系
   - 资金面：成交量变化
3. **交易建议**（强烈买入/买入/观望/卖出/强烈卖出）及依据
4. **简短理由**（包含事件驱动逻辑，字数≤50字）
"""

RISK_MANAGER_PROMPT = """
你是一个保守的A股风控官，核心目标是保护本金并防止回撤。你的任务是基于共享内存中的当前账户资金、持仓、市场分析师观点，结合A股特性给出仓位管理建议。

【反射检查要求】
1. 验证分析师数据：确认分析师报告是否包含政策面分析（缺失则要求重分析）
2. 评估A股特殊风险：计算T+1交易下可能的流动性风险，考虑涨跌停板导致的无法及时止损
3. 检查波动率：对比当前波动率与近30日均值（波动率>30%则风险等级上调）

输出必须包含：
1. 风险评估（高/中/低）及详细理由：
   - 高风险：政策突变+波动率>30%+涨跌停板限制
   - 中风险：政策影响+波动率20-30%
   - 低风险：政策平稳+波动率<20%
2. 建议仓位比例（0.0-1.0，必须考虑T+1和涨跌停）：
   - 例：若分析师建议"强烈买入"但风险评估"高"，则建议≤0.3
3. 简短理由（明确说明T+1/涨跌停对仓位的影响，字数≤40字）
"""

FUND_MANAGER_PROMPT = """
你是一个基金经理，拥有最终决策权。必须基于共享内存中的所有信息（市场分析、风控建议、账户状态）做出交易指令，优先遵循风控官意见。

【关键规则】
1. 冲突处理：当分析师建议"买入"但风控建议"卖出"时，强制采用风控建议
2. A股特殊约束：所有决策必须考虑T+1交易和涨跌停板
3. 学习机制：每次决策后必须记录依据，用于后续优化

【重要】输出必须是纯JSON格式，不要使用<think>标签，不要有任何额外说明文本，直接输出JSON：
{
    "action": "BUY" | "SELL" | "HOLD",
    "quantity_pct": 0.0 - 1.0,
    "reason": "综合分析：包含市场趋势（技术/政策/情绪）、风险评估等级、A股特性影响（如'涨跌停限制导致仓位≤0.3'）",
    "learning_note": "决策依据与结果记录（例：'2023-10-05政策利好未被计入，下次需增加政策监测'）"
}

注意：不要使用<think>标签进行思考，直接输出纯JSON对象即可。
"""

# --- 从 config 读取当前 provider 下各节点模型（未配置则用该 provider 的 model）---
_DEFAULT_AGENT_MODELS = {
    "nvidia": {
        "market_analyst": "qwen/qwq-32b",
        "risk_manager": "minimaxai/minimax-m2.1",
        "fund_manager": "deepseek-ai/deepseek-v3.2",
    },
    "volcano": {
        "market_analyst": "ep-20251208192433-wsbrk",
        "risk_manager": "ep-20251208192433-wsbrk",
        "fund_manager": "ep-20251208192433-wsbrk",
    },
}


def _get_agent_model(node_name: str) -> str | None:
    """从 config.yaml 的 llm.providers[active].agent_models 读取节点模型，未配置则用该 provider 的 model 或内置默认。"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if not os.path.exists(config_path):
        return (_DEFAULT_AGENT_MODELS.get("volcano") or {}).get(node_name)
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        llm = cfg.get("llm") or {}
        active = (llm.get("active") or os.environ.get("LLM_PROVIDER") or "volcano").lower()
        providers = llm.get("providers") or {}
        spec = providers.get(active) or {}
        agent_models = spec.get("agent_models") or {}
        return agent_models.get(node_name) or spec.get("model") or (_DEFAULT_AGENT_MODELS.get(active) or {}).get(node_name)
    except Exception:
        return (_DEFAULT_AGENT_MODELS.get("volcano") or {}).get(node_name)


# --- Node Functions ---

async def market_analyst_node(state: AgentState):
    logger.info(f"Market Analyst Node Input: {state}")
    query = f"日期: {state['date']}\n市场数据:\n{state['market_data']}"
    model = _get_agent_model("market_analyst")
    response, _ = await async_hunyuan_chat(
        query=query,
        prompt=ANALYST_PROMPT,
        model=model,
    )
    return {"analyst_view": response}


async def risk_manager_node(state: AgentState):
    logger.info(f"Risk Manager Node Input: {state}")
    query = f"日期: {state['date']}\n账户状态:\n{state['account_status']}\n市场分析师观点:\n{state['analyst_view']}"
    model = _get_agent_model("risk_manager")
    response, _ = await async_hunyuan_chat(
        query=query,
        prompt=RISK_MANAGER_PROMPT,
        model=model,
    )
    return {"risk_view": response}


async def fund_manager_node(state: AgentState):
    logger.info(f"Fund Manager Node Input: {state}")
    query = f"""
    日期: {state['date']}
    市场分析师观点:
    {state['analyst_view']}

    风控官观点:
    {state['risk_view']}

    请做出决策。
    """
    model = _get_agent_model("fund_manager")
    response, _ = await async_hunyuan_chat(
        query=query,
        prompt=FUND_MANAGER_PROMPT,
        model=model,
    )
    
    # Import parse function
    from ..utils.llm_client import parse_llm_response
    
    # Clean up response to handle <think> tags and extract JSON
    cleaned_response = response.strip()
    
    # Remove <think> tags if present (for models like MiniMax, DeepSeek-R1)
    import re
    cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL).strip()
    
    # Remove markdown code blocks
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    
    cleaned_response = cleaned_response.strip()
    
    # If still not valid JSON, try to parse it
    try:
        json.loads(cleaned_response)
    except json.JSONDecodeError:
        # Use parse_llm_response with fallback
        fallback = {
            "action": "HOLD",
            "quantity_pct": 0.0,
            "reason": "Failed to parse LLM response"
        }
        parsed = parse_llm_response(cleaned_response, fallback)
        cleaned_response = json.dumps(parsed, ensure_ascii=False)
    
    return {"final_decision": cleaned_response}

# --- Graph Construction ---
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("fund_manager", fund_manager_node)

    workflow.set_entry_point("market_analyst")
    workflow.add_edge("market_analyst", "risk_manager")
    workflow.add_edge("risk_manager", "fund_manager")
    workflow.add_edge("fund_manager", END)

    return workflow.compile()

# --- Strategy Implementation ---
class AgentStrategy(BreadFreeStrategy):
    def __init__(self, broker, lot_size=100):
        super().__init__(broker, lot_size)
        self.app = build_graph()
        self.symbol = None
        self.news_data = []  # 新闻缓存
        self.volume_history = {}  # {symbol: [volume, ...]} 用于散户情绪

    def set_symbols(self, symbols):
        if len(symbols) > 1:
            print("Warning: AgentStrategy currently supports analysis for only one symbol. Using the first one.")
        self.symbol = symbols[0]
        super().set_symbols(symbols)
        for s in symbols:
            if s not in self.volume_history:
                self.volume_history[s] = []
        self.load_news()

    def load_news(self):
        """从 JSON 缓存加载新闻；无缓存时保持空，get_news_context 内会尝试实时拉一页补充"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, "data", "cache")
            file_path = os.path.join(cache_dir, f"news_{self.symbol}.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.news_data = json.load(f)
                print(f"Loaded {len(self.news_data)} news items for {self.symbol}")
            else:
                self.news_data = []
        except Exception as e:
            print(f"Error loading news: {e}")
            self.news_data = []

    def _fetch_news_once(self):
        """无缓存时拉取一页新闻并合并到 self.news_data（仅补一次）"""
        if self.news_data:
            return
        try:
            from ..data.news_fetcher import NewsFetcher
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, "data", "cache")
            fetcher = NewsFetcher(data_dir=cache_dir)
            df = fetcher.get_stock_news(self.symbol)
            if df.empty:
                return
            # 统一为 get_news_context 使用的字段名
            if "发布时间" not in df.columns and "date" in df.columns:
                df["发布时间"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            for _, row in df.iterrows():
                pub = row.get("发布时间", row.get("date", ""))
                if hasattr(pub, "strftime"):
                    pub = pd.Timestamp(pub).strftime("%Y-%m-%d %H:%M:%S")
                self.news_data.append({
                    "发布时间": str(pub),
                    "新闻标题": str(row.get("新闻标题", row.get("title", ""))),
                    "新闻内容": str(row.get("新闻内容", row.get("content", "")))[:500],
                })
            if self.news_data:
                print(f"Fetched {len(self.news_data)} news items for {self.symbol} (on-the-fly)")
        except Exception as e:
            print(f"On-the-fly news fetch failed: {e}")

    def get_news_context(self, current_date_str, days=2):
        """过去 N 天的新闻摘要；无缓存时会尝试实时拉一页再过滤"""
        self._fetch_news_once()
        if not self.news_data:
            return "暂无相关新闻数据（可预先运行 news_fetcher 拉取并保存到 data/cache/news_<代码>.json）"
        try:
            current_date = pd.to_datetime(current_date_str)
            start_date = current_date - timedelta(days=days)
            relevant_news = []
            for item in self.news_data:
                t = item.get("发布时间")
                if not t:
                    continue
                news_time = pd.to_datetime(t, errors="coerce")
                if pd.isna(news_time):
                    continue
                if start_date <= news_time <= current_date + timedelta(days=1):
                    content = (item.get("新闻内容") or "")[:200]
                    relevant_news.append(f"- [{t}] {item.get('新闻标题', '')} {content} ...")
            if not relevant_news:
                # 若无当日段内新闻，返回最近几条作为“近期舆情”参考
                recent = []
                for item in self.news_data[:5]:
                    t = item.get("发布时间", "")
                    content = (item.get("新闻内容") or "")[:200]
                    recent.append(f"- [{t}] {item.get('新闻标题', '')} {content} ...")
                return "该时段无新闻，近期舆情参考:\n" + "\n".join(recent) if recent else "该时段暂无新闻"
            return "\n".join(relevant_news[:10])
        except Exception as e:
            print(f"Error filtering news: {e}")
            return "新闻数据处理出错"

    def on_bar(self, date, bars):
        if self.symbol not in bars:
            return
        bar_data = bars[self.symbol]
        if self.symbol not in self.history:
            self.history[self.symbol] = []
        self.history[self.symbol].append(bar_data["close"])
        if self.symbol not in self.volume_history:
            self.volume_history[self.symbol] = []
        self.volume_history[self.symbol].append(bar_data.get("volume", 0))
        try:
            asyncio.run(self._run_graph(date, bar_data))
        except Exception as e:
            print(f"Error running LangGraph strategy: {e}")

    async def _run_graph(self, date, bar_data):
        # 1. Prepare Data
        # Calculate Indicators
        hist = self.history[self.symbol]
        ma5 = pd.Series(hist).rolling(window=5).mean().iloc[-1] if len(hist) >= 5 else bar_data['close']
        ma20 = pd.Series(hist).rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else bar_data['close']
        
        # Calculate Volatility (Standard Deviation of last 30 days returns)
        volatility = 0.0
        if len(hist) >= 30:
            returns = pd.Series(hist).pct_change().dropna()
            volatility = returns.tail(30).std() * 100 # Percentage

        # Get recent price history (last 5 days)
        recent_prices = hist[-5:] if len(hist) >= 5 else hist
        recent_prices_str = ", ".join([f"{p:.2f}" for p in recent_prices])

        news_context = self.get_news_context(str(date))

        # 散户情绪：基于量比（当日量/近5日均量）与近期涨跌
        vol_hist = self.volume_history.get(self.symbol) or []
        cur_vol = bar_data.get("volume", 0) or 1
        avg_vol_5 = (sum(vol_hist[-5:]) / 5) if len(vol_hist) >= 5 else cur_vol
        vol_ratio = cur_vol / avg_vol_5 if avg_vol_5 else 1.0
        if vol_ratio > 1.2:
            vol_desc = "交投活跃、情绪升温"
        elif vol_ratio < 0.8:
            vol_desc = "观望、情绪谨慎"
        else:
            vol_desc = "成交平稳、情绪稳定"
        if len(hist) >= 5:
            ret_5 = (hist[-1] - hist[-5]) / hist[-5] if hist[-5] else 0
            trend = "偏多" if ret_5 > 0.02 else "偏空" if ret_5 < -0.02 else "中性"
            sentiment = f"量比{vol_ratio:.2f}，{vol_desc}；近5日涨跌{trend}。"
        else:
            sentiment = f"量比{vol_ratio:.2f}，{vol_desc}。"

        market_data_str = f"""
        Date: {date}
        Close: {bar_data['close']:.2f}
        Open: {bar_data['open']:.2f}
        High: {bar_data['high']:.2f}
        Low: {bar_data['low']:.2f}
        Volume: {bar_data['volume']}
        
        Technical Indicators:
        MA5: {ma5:.2f}
        MA20: {ma20:.2f}
        Volatility (30-day): {volatility:.2f}%
        
        Recent 5 Days Close Prices: [{recent_prices_str}]
        
        Shared Memory Context:
        News Events:
        {news_context}
        
        Retail Sentiment: {sentiment}
        """

        account_status_str = f"""
        Cash: {self.broker.cash:.2f}
        Positions: {self.broker.positions}
        """

        initial_state = AgentState(
            date=str(date),
            market_data=market_data_str,
            account_status=account_status_str,
            analyst_view="",
            risk_view="",
            final_decision=""
        )

        # 2. Run Graph
        result = await self.app.ainvoke(initial_state)
        
        # 3. Execute Decision
        try:
            decision = json.loads(result["final_decision"])
            action = decision.get("action", "HOLD").upper()
            quantity_pct = float(decision.get("quantity_pct", 0.0))
            reason = decision.get("reason", "")

            print(f"[{date}] Agent Decision: {action} | Reason: {reason}")

            close_price = bar_data['close']

            if action == "BUY":
                available_cash = self.broker.cash
                target_cash = available_cash * quantity_pct
                if target_cash > 0 and close_price > 0:
                    # 先按目标资金计算最多可买入的股数（未考虑整手）
                    max_shares = int(target_cash / (close_price * (1 + self.broker.commission_rate)))
                    quantity = (max_shares // self.lot_size) * self.lot_size

                    # 边界处理：若按目标仓位计算结果为0，但账户可用现金足以购买至少一手，则按一手下单并记录警告
                    lot_cost = close_price * self.lot_size * (1 + self.broker.commission_rate)
                    if quantity == 0:
                        if available_cash >= lot_cost and quantity_pct > 0:
                            quantity = self.lot_size
                            print(f"[{date}] Warning: target_cash insufficient for a lot, falling back to 1 lot (lot_size={self.lot_size}).")
                        else:
                            # 无法买入任何整手，输出调试信息
                            print(f"[{date}] Info: target_cash={target_cash:.2f}, lot_cost={lot_cost:.2f}, available_cash={available_cash:.2f} -> no shares to buy.")

                    if quantity > 0:
                        self.broker.buy(date, self.symbol, close_price, quantity)
                elif close_price <= 0:
                    print(f"[{date}] Warning: Invalid close price for {self.symbol}: {close_price}. Skipping buy.")
            
            elif action == "SELL":
                if self.symbol in self.broker.positions and close_price > 0:
                    pos = self.broker.positions[self.symbol]
                    quantity = int(pos.quantity * quantity_pct)
                    quantity = (quantity // self.lot_size) * self.lot_size

                    # 边界处理：当计算结果为0但持仓小于整手时，允许清仓卖出（全部持仓），或当quantity_pct>0且pos.quantity>=lot_size但四舍五入为0时，提示原因
                    if quantity == 0:
                        if pos.quantity > 0 and pos.quantity < self.lot_size:
                            quantity = pos.quantity
                            print(f"[{date}] Info: holding less than one lot ({pos.quantity}), selling entire position.")
                        else:
                            print(f"[{date}] Info: computed sell quantity is 0 (pos={pos.quantity}, quantity_pct={quantity_pct}). Skipping sell.")

                    if quantity > 0:
                        self.broker.sell(date, self.symbol, close_price, quantity)
                elif close_price <= 0:
                    print(f"[{date}] Warning: Invalid close price for {self.symbol}: {close_price}. Skipping sell.")

        except json.JSONDecodeError:
            print(f"[{date}] Failed to parse JSON decision: {result['final_decision']}")
        except Exception as e:
            print(f"[{date}] Error executing decision: {e}")
