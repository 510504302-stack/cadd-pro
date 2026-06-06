"""
CADD-Pro — Computer-Aided Drug Design Intelligent Platform
"""
import streamlit as st
import sys, os, glob, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header

st.set_page_config(
    page_title="CADD-Pro | 智能药物设计平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar()

# ── Hero ──
render_page_header("🧬 计算机辅助药物设计平台",
                    "分子指纹计算 · 机器学习建模 · 活性预测 · 文献挖掘")

# ── Stats Row ──
csv_count = len(glob.glob("./data/*.csv"))
project_count = len(glob.glob("./projects/*")) if os.path.exists("./projects") else 0
total_compounds = 0
for f in glob.glob("./data/*.csv"):
    try:
        df = pd.read_csv(f)
        total_compounds += len(df)
    except Exception:
        pass

st.markdown('<p style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">平台概览</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 数据集", csv_count)
c2.metric("🧪 化合物", f"{total_compounds:,}")
c3.metric("📁 训练项目", project_count)
c4.metric("🤖 ML 算法", "4+")

# ── Navigation Grid ──
st.markdown('<p style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin:2rem 0 0.6rem;">功能导航</p>', unsafe_allow_html=True)

pages = [
    ("📊", "数据展示", "交互式数据集探索、统计图表与相关性分析", "pages/1_📊_数据展示.py"),
    ("🧠", "模型训练", "RF · XGBoost · SVM · LR 多算法建模与 CV 验证", "pages/2_🧠_模型训练.py"),
    ("🔮", "活性预测", "SMILES 输入预测活性 + SHAP 可解释性分析", "pages/3_🔮_活性预测.py"),
    ("📁", "项目管理", "查看历史项目、模型对比与结果导出", "pages/4_📁_项目管理.py"),
    ("📚", "知识获取", "PubMed / PMC 文献检索 + AI 知识提取", "pages/5_📚_知识获取.py"),
    ("🔍", "相似性搜索", "Morgan 指纹 Tanimoto 分子相似性搜索", "pages/6_🔍_相似性搜索.py"),
]
cols = st.columns(3)
for i, (icon, name, desc, path) in enumerate(pages):
    with cols[i % 3]:
        with st.container(border=True):
            st.page_link(path, label=f"{icon}  {name}", help=desc)
            st.caption(desc)

# ── Quick Start ──
st.markdown('<p style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; margin:2.5rem 0 0.6rem;">三步上手</p>', unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)
g1.info("**1️⃣ 探索数据**\n\n进入 **数据展示**，查看数据集分布与统计", icon="📊")
g2.info("**2️⃣ 训练模型**\n\n在 **模型训练** 中选择算法，一键训练", icon="🧠")
g3.info("**3️⃣ 预测分析**\n\n在 **活性预测** 中输入 SMILES 获取结果", icon="🔮")

render_footer()
