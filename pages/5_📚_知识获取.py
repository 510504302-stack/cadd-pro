"""
📚 知识获取 - PubMed/PMC文献检索与AI知识提取 (多模型支持)
"""
import streamlit as st
import pandas as pd, os, sys, re
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header
from utils.pubmed_utils import set_entrez_email, search_pmc, search_pubmed, fetch_article_details, fetch_pubmed_summary, extract_article_text, parse_tsv_to_dataframe

st.set_page_config(page_title="知识获取 | CADD-Pro", page_icon="📚", layout="wide")
inject_css()
render_sidebar()

render_page_header("📚 知识获取", "PubMed/PMC文献智能检索，多模型AI驱动的毒副作用信息提取与结构化输出")

# ═══════════════════════════════════════════════════════════════
# Multi-Provider LLM Configuration
# ═══════════════════════════════════════════════════════════════
LLM_PROVIDERS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "key_patterns": [r"^sk-proj-", r"^sk-svcacct-", r"^sk-[a-zA-Z0-9]{48,}$"],
        "key_hint": "sk-proj-... / sk-...",
        "icon": "🟢",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "key_patterns": [r"^sk-[a-zA-Z0-9]{32}$"],
        "key_hint": "sk-... (32位)",
        "icon": "🔵",
    },
    "Moonshot (Kimi)": {
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "key_patterns": [r"^sk-"],
        "key_hint": "sk-...",
        "icon": "🌙",
    },
    "智谱 GLM": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4", "glm-4-flash", "glm-4v", "glm-3-turbo"],
        "key_patterns": [r"\.[a-zA-Z0-9]"],
        "key_hint": "xxx.xxxxxxxxxxx (含.)",
        "icon": "🧠",
    },
    "通义千问 (Qwen)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-longcontext"],
        "key_patterns": [r"^sk-"],
        "key_hint": "sk-...",
        "icon": "☁️",
    },
    "SiliconFlow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "models": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct", "Pro/meta-llama/Meta-Llama-3.1-405B-Instruct"],
        "key_patterns": [r"^sk-"],
        "key_hint": "sk-...",
        "icon": "⚡",
    },
}


def detect_provider(api_key: str) -> str:
    """Auto-detect LLM provider from API key format."""
    if not api_key:
        return "OpenAI"
    for name, cfg in LLM_PROVIDERS.items():
        for pattern in cfg["key_patterns"]:
            if re.match(pattern, api_key):
                return name
    return "OpenAI"


# ── Preset searches ─────────────────────────────────────────
preset = {"自定义搜索": "", "临床毒理学+化学物": '"Clinical Toxicology" AND "Chemical"',
          "药物诱导肝毒性": '"Drug-Induced Liver Injury" AND "mechanism"',
          "化合物心脏毒性": '"Cardiotoxicity" AND "compound"',
          "环境毒理学": '"Environmental Toxicology" AND "chemical exposure"',
          "药物相互作用": '"Drug-Drug Interaction" AND "adverse effect"'}

# ── Settings Panel (检索设置) ───────────────────────────────
with st.expander("⚙️ 检索设置", expanded=True):
    sc1, sc2, sc3 = st.columns(3)
    entrez_email = sc1.text_input("📧 NCBI Email", "user@example.com")
    set_entrez_email(entrez_email)

    # Track previous DB to detect switch
    prev_db = st.session_state.get('db_choice', None)
    db_choice = sc2.radio("🗄️ 数据库", ["PubMed Central (PMC)", "PubMed"], horizontal=True)

    # If DB changed, clear old search results
    if prev_db and prev_db != db_choice and 'search_ids' in st.session_state:
        st.warning(f"⚠️ 数据库已从 {prev_db} 切换为 {db_choice}，请重新搜索。")
        st.session_state.pop('search_ids', None)
        st.session_state.pop('db_choice', None)
        st.session_state.pop('article_loaded', None)
        st.rerun()

    st.session_state['db_choice'] = db_choice
    max_results = sc3.slider("最大结果数", 1, 20, 10)

    st.markdown("---")
    pc = st.selectbox("📌 预设主题", list(preset.keys()))
    keyword = st.text_input("🔎 关键词", value=preset[pc] if pc != "自定义搜索" else "", placeholder='"Drug Toxicity" AND "ML"')
    search_btn = st.button("🔍 搜索文献", use_container_width=True, type="primary")

# ── Step 1: Search ──────────────────────────────────────────
if search_btn and keyword:
    with st.spinner("搜索中..."):
        ids = search_pmc(keyword, max_results) if db_choice == "PubMed Central (PMC)" else search_pubmed(keyword, max_results)
    if ids:
        st.success(f"✅ 找到 {len(ids)} 篇文献")
        st.session_state['search_ids'] = ids
        st.session_state['db_choice'] = db_choice
        st.session_state['article_loaded'] = False
        st.code(", ".join(ids))
        if db_choice == "PubMed":
            sums = fetch_pubmed_summary(ids)
            if sums: st.dataframe(pd.DataFrame(sums), use_container_width=True)
    else:
        st.warning("未找到相关文献。")

# ── Step 2: Article Viewer ──────────────────────────────────
if 'search_ids' in st.session_state and st.session_state.get('search_ids'):
    st.divider()
    st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">📄 文献详情</p>', unsafe_allow_html=True)

    ids = st.session_state['search_ids']
    db = st.session_state.get('db_choice', 'PubMed')
    sid = st.selectbox(f"选择文献 ({db})", ids, key="article_select")

    if sid:
        with st.spinner("获取文献中..."):
            if db == "PubMed Central (PMC)":
                ad = fetch_article_details(sid)
                if ad:
                    title, abstract, full_text = extract_article_text(ad)
                    st.session_state['article_title'] = title
                    st.session_state['article_abstract'] = abstract
                    st.session_state['article_fulltext'] = full_text
                    st.session_state['article_loaded'] = True

                    st.markdown(f"### {title}" if title else "### 文献详情")
                    if abstract:
                        with st.expander("📋 摘要", expanded=True):
                            st.info(abstract)
                    if full_text:
                        with st.expander("📄 全文内容", expanded=False):
                            st.text_area("全文", full_text, height=400, label_visibility="collapsed", key="fulltext_display")
                    else:
                        st.warning("⚠️ 无法提取全文内容，仅可分析摘要。")
                else:
                    st.error(f"获取文献 {sid} 失败。")
            else:
                sums = fetch_pubmed_summary([sid])
                if sums:
                    s = sums[0]
                    st.markdown(f"### {s['title']}")
                    st.markdown(f"**来源:** {s['source']} | **日期:** {s['pubdate']} | **作者:** {s['authors']}")
                    abstract = "PubMed 仅提供摘要，请切换到 PMC 获取全文。"
                    st.session_state['article_title'] = s['title']
                    st.session_state['article_abstract'] = abstract
                    st.session_state['article_fulltext'] = ""
                    st.session_state['article_loaded'] = True

        # ── Step 3: AI 解读 ──
        if st.session_state.get('article_loaded'):
            st.divider()
            st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">🤖 AI 智能解读</p>', unsafe_allow_html=True)

            # Persist AI key in session state
            if "ai_api_key" not in st.session_state:
                st.session_state.ai_api_key = ""
            if "ai_ready" not in st.session_state:
                st.session_state.ai_ready = False

            with st.expander("🔑 AI 模型配置", expanded=not st.session_state.ai_ready):
                sc4, sc5 = st.columns([3, 1])
                with sc4:
                    api_key_local = st.text_input(
                        "API Key",
                        type="password",
                        value=st.session_state.ai_api_key,
                        placeholder="输入大模型 API Key（自动识别平台）",
                        help="支持 OpenAI / DeepSeek / Moonshot / 智谱GLM / 通义千问 / SiliconFlow",
                        key="ai_key_input"
                    )
                    if api_key_local:
                        st.session_state.ai_api_key = api_key_local

                detected = detect_provider(api_key_local)
                if "selected_provider" not in st.session_state:
                    st.session_state.selected_provider = detected
                elif api_key_local:
                    st.session_state.selected_provider = detected

                provider_names = list(LLM_PROVIDERS.keys())
                prov_idx = provider_names.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_names else 0

                with sc5:
                    sel_prov = st.selectbox(
                        "🏢 平台",
                        provider_names,
                        index=prov_idx,
                        format_func=lambda x: f"{LLM_PROVIDERS[x]['icon']} {x}",
                        key="ai_provider_input"
                    )
                    st.session_state.selected_provider = sel_prov

                cfg_local = LLM_PROVIDERS[sel_prov]

                if api_key_local:
                    st.success(f"✅ 已识别为 **{sel_prov}** · `{cfg_local['base_url']}`")
                    st.session_state.ai_ready = True
                else:
                    st.warning("⚠️ 请先输入 API Key 才能使用 AI 解读功能。")
                    st.session_state.ai_ready = False

                ai_model_local = st.selectbox(
                    "🧠 模型",
                    cfg_local["models"],
                    key="ai_model_input"
                )

            # ── Validate & show AI button ──
            if not st.session_state.ai_ready:
                st.info("👆 请展开上方「🔑 AI 模型配置」输入 API Key 后即可使用 AI 解读。")
            else:
                col_mode, col_btn = st.columns([3, 1])
                with col_mode:
                    ai_mode = st.radio(
                        "分析模式",
                        ["📝 自由提问", "📊 结构化提取 (TSV)", "🧪 毒性总结"],
                        horizontal=True,
                        key="ai_mode_input"
                    )
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    ai_btn = st.button("🚀 AI 解读", use_container_width=True, type="primary", key="ai_btn_input")

                if ai_btn:
                    abstract_text = st.session_state.get('article_abstract', '')
                    if not abstract_text:
                        st.warning("没有可分析的文献内容。")
                    else:
                        from openai import OpenAI
                        client = OpenAI(api_key=st.session_state.ai_api_key, base_url=cfg_local["base_url"])

                        with st.spinner(f"🤖 {sel_prov} 分析中..."):
                            if ai_mode == "📝 自由提问":
                                q = f"请从以下文献中提取与毒副作用相关的化合物信息，包括名称、类型和毒副作用描述：\n{abstract_text}"
                            elif ai_mode == "📊 结构化提取 (TSV)":
                                q = f"""请从文献中提取毒副作用相关的化合物信息，TSV格式输出（化合物\\t类型\\t毒副作用），仅输出表格内容，英文回复：
文献：{abstract_text}"""
                            else:
                                q = f"请总结以下文献中的化合物毒性信息（中文回答，按严重程度分为高/中/低三类）：\n{abstract_text}"

                            try:
                                resp = client.chat.completions.create(
                                    model=ai_model_local,
                                    messages=[{"role": "user", "content": q}],
                                    temperature=0.3,
                                )
                                result_text = resp.choices[0].message.content

                                st.markdown("---")
                                st.markdown("#### 💬 AI 解读结果")
                                st.write(result_text)

                                if ai_mode == "📊 结构化提取 (TSV)":
                                    df = parse_tsv_to_dataframe(result_text)
                                    if not df.empty:
                                        st.markdown("#### 📋 结构化表格")
                                        st.dataframe(df, use_container_width=True)
                                        st.download_button(
                                            "📥 下载 TSV",
                                            df.to_csv(index=False).encode('utf-8'),
                                            "toxicity_extraction.tsv",
                                            "text/tab-separated-values",
                                            use_container_width=True,
                                            key="dl_tsv_final"
                                        )
                            except Exception as e:
                                st.error(f"AI 分析出错: {e}")
                                st.caption(f"💡 请确认 Key 有效且模型 `{ai_model_local}` 在 {sel_prov} 可用。")

render_footer()
