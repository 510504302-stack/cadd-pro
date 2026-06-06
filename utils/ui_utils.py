"""
UI utilities — theme-adaptive design system for Streamlit.
"""
import streamlit as st

DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* ── Font ── */
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* ── Typography (structural only, no color overrides) ── */
    h1 { font-size: 1.75rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important; }
    h2 { font-size: 1.35rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
    .stCaption { font-size: 0.85rem; }

    /* ── Sidebar (kept dark as design choice) ── */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] strong, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: #f8fafc !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.05) !important;
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] * { color: #cbd5e1 !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        border-radius: 14px !important; padding: 20px 24px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        transition: box-shadow 0.2s, transform 0.2s !important;
    }
    [data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important; transform: translateY(-1px); }
    [data-testid="stMetric"] label { font-size: 0.78rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }

    /* ── Data Tables ── */
    [data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; }
    [data-testid="stDataFrame"] table { border-collapse: separate !important; border-spacing: 0 !important; }
    [data-testid="stDataFrame"] thead th {
        font-weight: 700 !important; font-size: 0.8rem !important;
        text-transform: uppercase !important; letter-spacing: 0.05em !important;
        padding: 14px 16px !important;
    }
    [data-testid="stDataFrame"] tbody td { padding: 10px 16px !important; font-size: 0.88rem !important; }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        border-radius: 14px !important; font-weight: 600 !important;
        font-size: 0.95rem !important; padding: 14px 18px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-radius: 14px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px !important; padding: 10px 20px !important; font-weight: 500 !important; border: none !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 600 !important; }

    /* ── Buttons ── */
    .stButton > button { border-radius: 12px !important; font-weight: 600 !important; font-size: 0.9rem !important; padding: 10px 24px !important; }
    .stButton > button[kind="primary"]:hover { transform: translateY(-1px); }

    /* ── Inputs ── */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea > div > div > textarea { border-radius: 10px !important; }

    /* ── Alerts ── */
    .stAlert { border-radius: 14px !important; }

    /* ── Progress ── */
    .stProgress > div > div > div { border-radius: 8px !important; }

    /* ── Page Link Cards ── */
    div[data-testid="stPageLink-NavLink"] { border-radius: 14px !important; padding: 20px !important; transition: transform 0.2s, box-shadow 0.2s !important; }
    div[data-testid="stPageLink-NavLink"]:hover { transform: translateY(-2px); }
    div[data-testid="stPageLink-NavLink"] p { font-weight: 600 !important; }

    /* ── Code ── */
    code, .stCode { border-radius: 10px !important; font-size: 0.85rem !important; }

    /* ── Footer ── */
    .app-footer { text-align: center; padding: 2.5rem 1rem 1.5rem; margin-top: 4rem; font-size: 0.8rem; opacity: 0.6; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.4); border-radius: 3px; }
</style>
"""


def inject_css():
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style="text-align:center; padding:18px 8px 8px;">
                <div style="font-size:2rem; margin-bottom:4px;">🧬</div>
                <div style="font-size:1.1rem; font-weight:700; color:#f8fafc;">CADD-Pro</div>
                <div style="font-size:0.7rem; color:#64748b; margin-top:2px; letter-spacing:0.04em; text-transform:uppercase;">Drug Design Platform</div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()
        if st.button("🏠 返回首页", use_container_width=True):
            st.switch_page("app.py")
        st.divider()
        _render_sidebar_links()


def _render_sidebar_links():
    defaults = [
        {"label": "📊 数据展示", "page": "pages/1_📊_数据展示.py", "enabled": True},
        {"label": "🧠 模型训练", "page": "pages/2_🧠_模型训练.py", "enabled": True},
        {"label": "🔮 活性预测", "page": "pages/3_🔮_活性预测.py", "enabled": True},
        {"label": "📁 项目管理", "page": "pages/4_📁_项目管理.py", "enabled": False},
        {"label": "📚 知识获取", "page": "pages/5_📚_知识获取.py", "enabled": False},
        {"label": "🔍 相似性搜索", "page": "pages/6_🔍_相似性搜索.py", "enabled": False},
    ]
    if "sidebar_links" not in st.session_state:
        st.session_state.sidebar_links = defaults
    st.caption("快捷导航")
    for item in st.session_state.sidebar_links:
        if item["enabled"]:
            st.page_link(item["page"], label=item["label"])
    st.divider()
    with st.expander("⚙️ 自定义侧边栏", expanded=False):
        st.caption("勾选要显示的快捷链接：")
        for i, item in enumerate(st.session_state.sidebar_links):
            new_val = st.checkbox(item["label"], value=item["enabled"], key=f"sb_cfg_{i}")
            st.session_state.sidebar_links[i]["enabled"] = new_val


def render_page_header(title: str, description: str = ""):
    st.markdown(f'<h1 style="margin-bottom:0.3rem;">{title}</h1><p style="opacity:0.6; font-size:0.95rem; margin:0;">{description}</p>', unsafe_allow_html=True)
    st.divider()


def render_footer():
    st.markdown('<div class="app-footer">🧬 CADD-Pro &nbsp;·&nbsp; Powered by Streamlit · RDKit · Scikit-learn · SHAP</div>', unsafe_allow_html=True)
