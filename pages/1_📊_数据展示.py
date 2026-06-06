"""
📊 数据展示 — 交互式数据集探索
"""
import streamlit as st
import pandas as pd, numpy as np, glob, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header
from utils.viz_utils import plot_histogram, plot_correlation_heatmap, plot_missing_values, plot_class_distribution

st.set_page_config(page_title="数据展示 | CADD-Pro", page_icon="📊", layout="wide")
inject_css()
render_sidebar()

render_page_header("📊 数据展示", "交互式探索数据集概况，统计图表、相关性分析与缺失值检测")

# ── Settings Panel ──
with st.expander("⚙️ 数据选项", expanded=True):
    csv_files = glob.glob("./data/*.csv")
    if not csv_files:
        st.error("未找到数据集！"); st.stop()

    sc1, sc2, sc3 = st.columns([2, 2, 1])
    dataset_names = [os.path.basename(f) for f in csv_files]
    dataset_choice = sc1.selectbox("📂 数据集", dataset_names)
    selected_file = csv_files[dataset_names.index(dataset_choice)]

    @st.cache_data
    def load_data(fp): return pd.read_csv(fp)
    data = load_data(selected_file)

    sc2.markdown(f"""
        <div style="background:#f8fafc; border-radius:10px; padding:12px 16px; margin-top:4px;">
            <span style="font-weight:700; color:#0f172a;">{len(data):,}</span>
            <span style="color:#94a3b8; margin:0 8px;">行</span>
            <span style="font-weight:700; color:#0f172a;">{len(data.columns)}</span>
            <span style="color:#94a3b8; margin:0 8px;">列</span>
            <span style="color:#94a3b8;">· {data.memory_usage(deep=True).sum()/1024:.0f} KB</span>
        </div>
    """, unsafe_allow_html=True)

    sc3.markdown('<p style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em; margin:8px 0 4px;">显示选项</p>', unsafe_allow_html=True)
    show_raw = sc3.checkbox("原始数据", True)
    show_stats = sc3.checkbox("统计", True)
    show_charts = sc3.checkbox("图表", True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["📋 数据预览", "📈 统计与分布", "🔗 相关性与缺失值"])

with tab1:
    if show_raw:
        # ── Integrated column metadata strip ──
        col_count = len(data.columns)
        type_counts = data.dtypes.value_counts()
        type_badges = ' '.join([f'<span style="background:#eef2ff;color:#4f46e5;padding:2px 8px;border-radius:6px;font-size:0.75rem;font-weight:600;">{str(k)} ×{v}</span>' for k, v in type_counts.items()])
        null_total = data.isnull().sum().sum()
        null_pct = (null_total / (len(data) * col_count) * 100) if col_count else 0

        st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f8fafc,#f1f5f9); border:1px solid #e8ecf1; border-radius:14px; padding:16px 20px; margin-bottom:12px;
                        display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:1.3rem;">📋</span>
                    <span style="font-weight:700; color:#0f172a;">{len(data):,}</span>
                    <span style="color:#94a3b8; font-size:0.85rem;">行</span>
                    <span style="font-weight:700; color:#0f172a; margin-left:4px;">{col_count}</span>
                    <span style="color:#94a3b8; font-size:0.85rem;">列</span>
                </div>
                <div style="width:1px; height:24px; background:#e2e8f0;"></div>
                <div style="display:flex;align-items:center;gap:6px;">{type_badges}</div>
                <div style="width:1px; height:24px; background:#e2e8f0;"></div>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:0.85rem; color:#64748b;">缺失值</span>
                    <span style="font-weight:700; color:{'#ef4444' if null_total > 0 else '#22c55e'};">{null_total:,}</span>
                    <span style="color:#94a3b8; font-size:0.8rem;">({null_pct:.1f}%)</span>
                </div>
                <div style="flex:1;"></div>
                <span style="font-size:0.78rem; color:#94a3b8;">{data.memory_usage(deep=True).sum()/1024:.0f} KB</span>
            </div>
        """, unsafe_allow_html=True)

        # ── Data table ──
        st.dataframe(data, use_container_width=True, height=480,
                     column_config={col: st.column_config.Column(label=f"{col}  [{str(dt)}]", help=f"类型: {dt} | 非空: {data[col].count()} | 唯一值: {data[col].nunique()}")
                                   for col, dt in zip(data.columns, data.dtypes)})

        # Download
        csv_bytes = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出 CSV", csv_bytes, dataset_choice, "text/csv")

with tab2:
    if show_stats:
        st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">描述性统计</p>', unsafe_allow_html=True)
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.dataframe(data[num_cols].describe(), use_container_width=True)

    if show_charts:
        st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin:1.5rem 0 0.6rem;">数值特征分布</p>', unsafe_allow_html=True)
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            for i in range(0, min(6, len(num_cols)), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(num_cols):
                        with cols[j]:
                            fig = plot_histogram(data[num_cols[idx]], title=num_cols[idx][:30])
                            st.plotly_chart(fig, use_container_width=True)

        # Label distribution
        st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin:1.2rem 0 0.6rem;">标签分布</p>', unsafe_allow_html=True)
        pot = [c for c in data.columns if 2 <= data[c].nunique() <= 10]
        if pot:
            sel = st.selectbox("选择标签列", pot, key="label_sel")
            fig = plot_class_distribution(data[sel], title=f"'{sel}' 类别分布")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    ccorr, cmiss = st.columns(2)
    with ccorr:
        st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">特征相关性热图</p>', unsafe_allow_html=True)
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            fig = plot_correlation_heatmap(data[num_cols])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要至少 2 个数值型特征。")
    with cmiss:
        st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">缺失值分析</p>', unsafe_allow_html=True)
        fig = plot_missing_values(data)
        st.plotly_chart(fig, use_container_width=True)

render_footer()
