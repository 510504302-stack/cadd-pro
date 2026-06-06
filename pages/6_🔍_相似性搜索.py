"""
🔍 分子相似性搜索 - 基于Morgan指纹的Tanimoto相似性搜索
"""
import streamlit as st
import pandas as pd, numpy as np, glob, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header
from utils.chem_utils import mol_to_fp, is_valid_smiles, find_similar_molecules, draw_molecule, show_molecule
from utils.viz_utils import plot_similarity_heatmap

st.set_page_config(page_title="相似性搜索 | CADD-Pro", page_icon="🔍", layout="wide")
inject_css()
render_sidebar()

render_page_header("🔍 分子相似性搜索", "基于Morgan指纹的Tanimoto相似性计算，快速发现结构类似物")

# ── Settings in main content ────────────────────────────────
with st.expander("⚙️ 搜索设置", expanded=True):
    csv_files = glob.glob("./data/*.csv")
    if not csv_files:
        st.error("未找到数据集！"); st.stop()
    dataset_names = [os.path.basename(f) for f in csv_files]
    sc1, sc2, sc3 = st.columns(3)
    dataset_choice = sc1.selectbox("📂 参考数据库", dataset_names)

    @st.cache_data
    def lr(fp):
        df = pd.read_csv(fp)
        for c in ['smiles','SMILES','Smiles','canonical_smiles']:
            if c in df.columns: return df, c
        st.error("未找到SMILES列！"); st.stop()
    ref_df, smiles_col = lr(csv_files[dataset_names.index(dataset_choice)])
    sc2.metric("参考库大小", f"{len(ref_df):,}")
    top_k = sc3.slider("Top-K", 5, 50, 15, 5)

    uf = st.file_uploader("或上传自定义CSV（需含SMILES列）", type=['csv'])
    if uf:
        cdf = pd.read_csv(uf)
        for c in ['smiles','SMILES','Smiles']:
            if c in cdf.columns: ref_df, smiles_col = cdf, c; break

# ── Search ──────────────────────────────────────────────────
st.subheader("🎯 查询分子")
mode = st.radio("输入方式", ["✏️ SMILES", "📋 从库中选择", "📁 批量"], horizontal=True)
queries = []

if mode == "✏️ SMILES":
    smi = st.text_input("SMILES", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    if smi and is_valid_smiles(smi):
        img = draw_molecule(smi, (350, 250))
        if img:
            st.image(img, width=300)
        else:
            show_molecule(smi, 300, 250, "分子结构")
        queries = [smi]
elif mode == "📋 从库中选择":
    sample = ref_df[smiles_col].dropna().head(100).tolist()
    sel = st.selectbox("选择分子", [""] + sample)
    if sel: queries = [sel]
else:
    txt = st.text_area("每行一个SMILES", height=120, placeholder="CCO\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    if txt.strip(): queries = [s.strip() for s in txt.strip().split('\n') if s.strip() and is_valid_smiles(s.strip())]

if queries and st.button("🔍 搜索", use_container_width=True, type="primary"):
    ref_smis = ref_df[smiles_col].dropna().tolist()
    for qi, qs in enumerate(queries):
        if len(queries) > 1: st.subheader(f"查询 {qi+1}: `{qs[:50]}...`")
        with st.spinner("搜索中..."):
            results = find_similar_molecules(qs, ref_smis, top_k)
        if results:
            rdata = []
            for smi, sim in results:
                row = ref_df[ref_df[smiles_col] == smi]
                extra = {}
                if not row.empty:
                    for col in row.columns:
                        if col != smiles_col and row.iloc[0][col] is not None:
                            extra[col] = row.iloc[0][col]
                rdata.append({'SMILES': smi, 'Tanimoto': round(sim, 4), **extra})
            rdf = pd.DataFrame(rdata)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(rdf.style.background_gradient(cmap='Greens', subset=['Tanimoto']).format({'Tanimoto': '{:.4f}'}), use_container_width=True, height=400)
            with c2:
                fig = plot_similarity_heatmap([v for _, v in results], [s[:40] for s, _ in results])
                st.plotly_chart(fig, use_container_width=True)
            # Gallery
            gcols = st.columns(min(5, len(results)))
            for i, (smi, sim) in enumerate(results[:5]):
                with gcols[i]:
                    img = draw_molecule(smi, (160, 130))
                    if img:
                        st.image(img, caption=f"Sim:{sim:.3f}", use_container_width=True)
                    else:
                        show_molecule(smi, 140, 120, f"Sim:{sim:.3f}")
            st.download_button("📥 下载CSV", rdf.to_csv(index=False).encode('utf-8'), f"similarity_{qi+1}.csv", "text/csv", use_container_width=True)
        else:
            st.warning("未找到相似分子。")
        if len(queries) > 1: st.divider()

render_footer()
