"""
📁 项目管理 - 查看、对比和管理历史训练项目
"""
import streamlit as st
import pandas as pd, glob, os, sys, json, shutil, zipfile, io
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header

st.set_page_config(page_title="项目管理 | CADD-Pro", page_icon="📁", layout="wide")
inject_css()
render_sidebar()

render_page_header("📁 项目管理", "查看、对比和管理历史训练项目与模型评估结果")

projects = glob.glob('./projects/*')
if not projects:
    st.info("暂无项目，请先进行模型训练。")
    st.stop()

project_names = sorted([os.path.basename(p) for p in projects], reverse=True)

# ── Settings in main content ────────────────────────────────
with st.expander("⚙️ 操作选项", expanded=True):
    sc1, sc2 = st.columns(2)
    compare_projects = sc1.multiselect("🔄 选择项目对比（最多4个）", project_names, max_selections=4)
    export_project = sc2.selectbox("📥 导出项目", [""] + project_names)

# ── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 项目列表", "🔄 模型对比", "🗑️ 管理"])

with tab1:
    for pname in project_names:
        pd_ = os.path.join("./projects", pname)
        ip = os.path.join(pd_, "info.json")
        info = {}
        if os.path.exists(ip):
            try:
                with open(ip) as f: info = json.load(f)
            except: pass

        with st.expander(f"📁 {pname}", expanded=len(project_names) <= 3):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**模型:** {info.get('model_name','?')}  |  **数据:** {info.get('dataset','?')}")
                st.write(f"**标签:** {info.get('label_column','?')}")
                m = info.get('metrics', {})
                if m:
                    st.write(f"Acc: {m.get('accuracy','?'):.4f}" if isinstance(m.get('accuracy'), float) else "")
                    st.write(f"ROC AUC: {m.get('roc_auc','?'):.4f}" if isinstance(m.get('roc_auc'), float) else "")
            with c2:
                for fn in ["roc_curve.png", "confusion_matrix.png"]:
                    p = os.path.join(pd_, fn)
                    if os.path.exists(p): st.image(p, width=280)

with tab2:
    if len(compare_projects) >= 2:
        comp = []
        for pn in compare_projects:
            ip = os.path.join("./projects", pn, "info.json")
            if os.path.exists(ip):
                with open(ip) as f: info = json.load(f)
                m = info.get('metrics', {})
                comp.append({'项目': pn[:25], '模型': info.get('model_name','?'), 'Accuracy': m.get('accuracy'), 'Precision': m.get('precision'), 'Recall': m.get('recall'), 'F1': m.get('f1'), 'ROC AUC': m.get('roc_auc'), 'MCC': m.get('mcc')})
        if comp:
            dfc = pd.DataFrame(comp)
            st.dataframe(dfc.style.format({c: '{:.4f}' for c in dfc.columns if c not in ['项目','模型']}).background_gradient(cmap='Blues', subset=[c for c in dfc.columns if c not in ['项目','模型']]), use_container_width=True)

            # Radar
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'MCC']
            fig = go.Figure()
            colors = ['#2563eb','#7c3aed','#059669','#d97706']
            for i, row in enumerate(comp):
                vals = [row.get(m) or 0 for m in metric_names]
                fig.add_trace(go.Scatterpolar(r=vals, theta=metric_names, fill='toself', name=row['项目'], line_color=colors[i%4], opacity=0.7))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("请选择至少2个项目进行对比。")

with tab3:
    st.warning("删除操作不可恢复！")
    del_project = st.selectbox("选择要删除的项目", [""] + project_names)
    if del_project:
        confirm = st.text_input(f"输入 `{del_project}` 确认删除")
        if confirm == del_project and st.button("🗑️ 确认删除", type="primary"):
            shutil.rmtree(os.path.join("./projects", del_project))
            st.success("已删除！"); st.rerun()

    if export_project:
        pd_ = os.path.join("./projects", export_project)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(pd_):
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp, os.path.relpath(fp, pd_))
        st.download_button(f"📥 下载 {export_project}.zip", buf.getvalue(), f"{export_project}.zip", "application/zip", use_container_width=True)

render_footer()
