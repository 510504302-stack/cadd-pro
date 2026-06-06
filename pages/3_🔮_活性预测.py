"""
🔮 活性预测 - 分子活性预测与SHAP可解释性分析
"""
import streamlit as st
import pandas as pd, numpy as np, glob, os, sys, joblib, json
import matplotlib.pyplot as plt, shap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header
from utils.chem_utils import mol_to_fp, is_valid_smiles, batch_predict, draw_molecule

st.set_page_config(page_title="活性预测 | CADD-Pro", page_icon="🔮", layout="wide")
inject_css()
render_sidebar()

render_page_header("🔮 活性预测", "输入分子SMILES进行活性预测，支持SHAP可解释性分析与批量预测")

# ── Settings in main content ────────────────────────────────
with st.expander("⚙️ 预测设置", expanded=True):
    projects = glob.glob('./projects/*')
    if not projects:
        st.warning("未找到已训练项目，请先训练模型。")
        st.stop()
    project_names = [os.path.basename(p) for p in projects]
    sc1, sc2 = st.columns([2, 3])
    project_name = sc1.selectbox("📁 选择项目", project_names)
    selected_dir = os.path.join("./projects", project_name)

    info_path = os.path.join(selected_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f: pi = json.load(f)
        sc2.markdown(f"**模型:** {pi.get('model_name','?')} | **数据:** {pi.get('dataset','?')} | **Acc:** {pi.get('metrics',{}).get('accuracy','?'):.3f}" if isinstance(pi.get('metrics',{}).get('accuracy'), float) else "")

    model_path = os.path.join(selected_dir, "model.pkl")
    if not os.path.exists(model_path):
        st.error("未找到模型文件！"); st.stop()
    @st.cache_resource
    def lm(p): return joblib.load(p)
    model = lm(model_path)

    pred_mode = st.radio("预测模式", ["🎯 单分子预测", "📋 批量预测"], horizontal=True)

# ── Prediction ──────────────────────────────────────────────
if pred_mode == "🎯 单分子预测":
    c1, c2 = st.columns([1, 1])
    with c1:
        smi = st.text_input("SMILES", "C1C=CC(C)=C(CC2C=C(CCC)C=C2)C=1", placeholder="输入SMILES...")
        btn = st.button("🔮 预测", use_container_width=True, type="primary")
    with c2:
        if smi and is_valid_smiles(smi):
            img = draw_molecule(smi, (350, 250))
            if img: st.image(img, caption="分子结构", use_container_width=True)

    if btn and smi and is_valid_smiles(smi):
        with st.spinner("预测中..."):
            fp = mol_to_fp(smi)
            if fp is not None:
                pred = model.predict([fp])[0]
                proba = model.predict_proba([fp])[0]
                r1, r2, r3 = st.columns(3)
                r1.metric("预测类别", "Active" if pred == 1 else "Inactive")
                r2.metric("Class 0 概率", f"{proba[0]:.4f}")
                r3.metric("Class 1 概率", f"{proba[1]:.4f}")
                # ── SHAP Explanation ──
                st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin:1.2rem 0 0.5rem;">🔍 SHAP 特征解释</p>', unsafe_allow_html=True)

                try:
                    fp_2d = fp.reshape(1, -1)
                    feature_names = [f"FP_{i}" for i in range(len(fp))]

                    # Determine model type
                    model_type = type(model).__name__

                    if model_type in ("RandomForestClassifier", "XGBClassifier"):
                        # Tree-based: use TreeExplainer
                        explainer = shap.TreeExplainer(model)
                        sv_raw = explainer.shap_values(fp_2d)

                        # Handle different SHAP output formats
                        if isinstance(sv_raw, list):
                            # Old SHAP format: list of arrays per class
                            sv = sv_raw[1] if len(sv_raw) > 1 else sv_raw[0]
                            ev = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
                            # New SHAP format: (samples, features, classes)
                            sv = sv_raw[:, :, 1] if sv_raw.shape[2] > 1 else sv_raw[:, :, 0]
                            ev = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(np.atleast_1d(explainer.expected_value)) > 1 else np.atleast_1d(explainer.expected_value)[0]
                        else:
                            sv = sv_raw
                            ev = np.atleast_1d(explainer.expected_value)[0]

                        # Top-N feature importance bar chart (always works)
                        top_n = min(20, len(fp))
                        top_idx = np.argsort(np.abs(sv[0]))[-top_n:][::-1]

                        # Bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#ef4444' if sv[0][i] > 0 else '#3b82f6' for i in top_idx]
                        ax.barh(range(top_n), sv[0][top_idx], color=colors, edgecolor='white', linewidth=0.5)
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
                        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
                        ax.set_title(f"Top {top_n} Feature Contributions  |  Base Value: {ev:.4f}", fontsize=13, fontweight='bold')
                        ax.axvline(x=0, color='#94a3b8', linewidth=0.8, linestyle='--')
                        ax.invert_yaxis()
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption("🔴 红色 = 推向阳性 (Active)  |  🔵 蓝色 = 推向阴性 (Inactive)")

                    elif model_type == "SVC":
                        # SVM: use KernelExplainer with a few background samples
                        st.info("SVM 模型使用 KernelExplainer 计算 SHAP（需要背景样本，较慢）...")
                        # Sample from input.csv for background
                        input_csv = os.path.join(selected_dir, "input.csv")
                        if os.path.exists(input_csv):
                            bg_data = pd.read_csv(input_csv).iloc[:, :-1].sample(min(50, len(pd.read_csv(input_csv))), random_state=42).values
                        else:
                            bg_data = np.zeros((10, len(fp)))

                        explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                        sv_raw = explainer.shap_values(fp_2d, nsamples=100)
                        # Handle multi-class output formats
                        if isinstance(sv_raw, list) and len(sv_raw) > 1:
                            sv_cls = sv_raw[1]
                        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
                            sv_cls = sv_raw[:, :, 1] if sv_raw.shape[2] > 1 else sv_raw[:, :, 0]
                        else:
                            sv_cls = sv_raw[0] if isinstance(sv_raw, list) else sv_raw

                        top_n = min(20, len(fp))
                        top_idx = np.argsort(np.abs(sv_cls[0]))[-top_n:][::-1]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#ef4444' if sv_cls[0][i] > 0 else '#3b82f6' for i in top_idx]
                        ax.barh(range(top_n), sv_cls[0][top_idx], color=colors, edgecolor='white', linewidth=0.5)
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
                        ax.set_xlabel("SHAP Value", fontsize=11)
                        ax.set_title(f"Top {top_n} Feature Contributions ({model_type})", fontsize=13, fontweight='bold')
                        ax.axvline(x=0, color='#94a3b8', linewidth=0.8, linestyle='--')
                        ax.invert_yaxis()
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    elif model_type == "LogisticRegression":
                        # LR: use LinearExplainer
                        explainer = shap.LinearExplainer(model, fp_2d, feature_dependence="independent")
                        sv_lr = explainer.shap_values(fp_2d)

                        top_n = min(20, len(fp))
                        top_idx = np.argsort(np.abs(sv_lr[0]))[-top_n:][::-1]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#ef4444' if sv_lr[0][i] > 0 else '#3b82f6' for i in top_idx]
                        ax.barh(range(top_n), sv_lr[0][top_idx], color=colors, edgecolor='white', linewidth=0.5)
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
                        ax.set_xlabel("SHAP Value", fontsize=11)
                        ax.set_title(f"Top {top_n} Feature Contributions ({model_type})", fontsize=13, fontweight='bold')
                        ax.axvline(x=0, color='#94a3b8', linewidth=0.8, linestyle='--')
                        ax.invert_yaxis()
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    else:
                        # Generic fallback: use model's feature_importances_ or coef_
                        if hasattr(model, "feature_importances_"):
                            importance = model.feature_importances_
                        elif hasattr(model, "coef_"):
                            importance = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
                        else:
                            importance = np.ones(len(fp))

                        top_n = min(20, len(fp))
                        top_idx = np.argsort(importance)[-top_n:][::-1]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(range(top_n), importance[top_idx], color='#4f46e5', edgecolor='white', linewidth=0.5)
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
                        ax.set_xlabel("Importance", fontsize=11)
                        ax.set_title(f"Top {top_n} Feature Importance ({model_type})", fontsize=13, fontweight='bold')
                        ax.invert_yaxis()
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                except Exception as e:
                    st.warning(f"SHAP 分析暂不可用: {str(e)[:100]}")
                    # Fallback to model's built-in importance
                    st.info("使用模型内置特征重要性作为替代：")
                    try:
                        if hasattr(model, "feature_importances_"):
                            imp = model.feature_importances_
                        elif hasattr(model, "coef_"):
                            imp = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
                        else:
                            imp = None
                        if imp is not None:
                            top_n = min(20, len(fp))
                            top_idx = np.argsort(imp)[-top_n:][::-1]
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.barh(range(top_n), imp[top_idx], color='#4f46e5', edgecolor='white')
                            ax.set_yticks(range(top_n))
                            ax.set_yticklabels([f"FP_{i}" for i in top_idx], fontsize=9)
                            ax.set_title("Feature Importance (fallback)", fontsize=13, fontweight='bold')
                            ax.invert_yaxis()
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    except:
                        st.caption("无法生成特征重要性图表。")
else:
    st.subheader("📋 批量预测")
    mode = st.radio("输入方式", ["📝 手动输入", "📄 CSV上传"], horizontal=True)
    smiles_list = []
    if mode == "📝 手动输入":
        txt = st.text_area("每行一个SMILES", height=150, placeholder="CCO\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
        if txt.strip(): smiles_list = [s.strip() for s in txt.strip().split('\n') if s.strip()]
    else:
        uf = st.file_uploader("上传CSV（需含SMILES列）", type=['csv'])
        if uf:
            dfu = pd.read_csv(uf)
            for c in ['smiles', 'SMILES', 'Smiles']:
                if c in dfu.columns:
                    smiles_list = dfu[c].dropna().tolist(); break
    if smiles_list:
        st.caption(f"{len(smiles_list)} 个分子")
        if st.button("🔮 批量预测", use_container_width=True, type="primary"):
            with st.spinner("预测中..."):
                res = batch_predict(model, smiles_list)
                st.dataframe(res, use_container_width=True)
                st.download_button("📥 下载CSV", res.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv", use_container_width=True)

render_footer()
