"""新闻数据管理页"""
import streamlit as st
import pandas as pd
import json, os
from breadfree.web.page_modules.utils import get_pool, sym_name


def render():
    st.header("📰 新闻数据管理")

    pool = get_pool()
    symbols = list(pool.keys())

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("选择标的", [sym_name(s) for s in symbols])
    with col2:
        max_pages = st.slider("抓取页数", 1, 20, 5)

    code = selected.split("-")[0]

    tab1, tab2 = st.tabs(["📋 已缓存新闻", "🔄 抓取新闻"])

    with tab1:
        _show_cached_news(code)

    with tab2:
        if st.button("🚀 开始抓取", type="primary"):
            _fetch_news(code, max_pages)


def _show_cached_news(code: str):
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
    cache_file = os.path.join(cache_dir, f"news_{code}.json")

    if not os.path.exists(cache_file):
        st.info(f"{sym_name(code)} 暂无缓存新闻，请先抓取")
        return

    with open(cache_file, "r", encoding="utf-8") as f:
        news = json.load(f)

    st.success(f"共 {len(news)} 条新闻")

    df = pd.DataFrame(news)
    if df.empty:
        return

    time_col = "发布时间" if "发布时间" in df.columns else "date"
    title_col = "新闻标题" if "新闻标题" in df.columns else "title"
    content_col = "新闻内容" if "新闻内容" in df.columns else "content"
    source_col = "文章来源" if "文章来源" in df.columns else "mediaName"

    # 搜索过滤
    keyword = st.text_input("🔍 关键词搜索")
    if keyword:
        mask = df[title_col].str.contains(keyword, na=False) | df[content_col].str.contains(keyword, na=False)
        df = df[mask]
        st.caption(f"匹配 {len(df)} 条")

    for _, row in df.head(50).iterrows():
        pub_time = row.get(time_col, "")
        title = row.get(title_col, "")
        source = row.get(source_col, "")
        content = str(row.get(content_col, ""))[:300]

        with st.container():
            st.markdown(f"**{title}**")
            st.caption(f"📅 {pub_time}  |  📰 {source}")
            st.text(content + "..." if len(content) >= 300 else content)
            st.divider()


def _fetch_news(code: str, max_pages: int):
    from breadfree.data.news_fetcher import NewsFetcher

    fetcher = NewsFetcher()
    progress = st.progress(0, text=f"正在抓取 {sym_name(code)} 的新闻...")

    all_news = []
    for page in range(1, max_pages + 1):
        progress.progress(page / max_pages, text=f"抓取第 {page}/{max_pages} 页...")
        df = fetcher._fetch_page(code, page_index=page)
        if df.empty:
            break
        all_news.extend(df.to_dict(orient="records"))

    progress.progress(1.0, text="抓取完成!")

    if not all_news:
        st.warning("未抓取到新闻")
        return

    final_df = pd.DataFrame(all_news)
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df.sort_values(by="date", ascending=False, inplace=True)
    final_df["date"] = final_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    final_df.rename(columns={
        "date": "发布时间", "mediaName": "文章来源",
        "title": "新闻标题", "content": "新闻内容", "url": "新闻链接"
    }, inplace=True)

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"news_{code}.json")
    final_df.to_json(cache_file, orient="records", force_ascii=False, indent=4)

    st.success(f"✅ 已保存 {len(final_df)} 条新闻到缓存")
    st.rerun()
