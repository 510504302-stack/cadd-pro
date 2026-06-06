"""
🧠 模型训练 — 多算法机器学习建模与评估
"""
import streamlit as st
import pandas as pd, numpy as np, glob, os, sys, joblib, json, random, string
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ui_utils import inject_css, render_sidebar, render_footer, render_page_header
from utils.chem_utils import mol_to_fp, save_fingerprint_data
from utils.ml_utils import train_model, generate_evaluation_plots, MODEL_CONFIGS, preprocess_data

st.set_page_config(page_title="模型训练 | CADD-Pro", page_icon="🧠", layout="wide")
inject_css()
render_sidebar()
render_page_header("🧠 模型训练", "多算法机器学习建模，交叉验证、超参数调优与全面性能评估")

def create_project_directory():
    name = datetime.now().strftime("%Y-%m-%d-%H-%M") + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    d = os.path.join("./projects", name); os.makedirs(d, exist_ok=True); return d

# ── Config Panel ──
with st.expander("⚙️ 训练配置", expanded=True):
    sc1, sc2, sc3 = st.columns(3)
    csv_files = glob.glob("./data/*.csv")
    if not csv_files: st.error("未找到数据集！"); st.stop()
    dataset_names = [os.path.basename(f) for f in csv_files]
    dataset_choice = sc1.selectbox("📂 数据集", dataset_names)

    @st.cache_data
    def ld(fp): return pd.read_csv(fp)
    data = ld(csv_files[dataset_names.index(dataset_choice)])

    label_cols = [c for c in data.columns if data[c].nunique() <= 20] or data.columns.tolist()
    label_column = sc2.selectbox("🏷️ 标签列", label_cols)
    model_name = sc3.selectbox("🤖 算法", list(MODEL_CONFIGS.keys()))

    st.caption("超参数")
    param_config = MODEL_CONFIGS[model_name]["params"]
    params = {}
    pc = st.columns(4)
    for pi, (pn, pconf) in enumerate(param_config.items()):
        with pc[pi % 4]:
            if pconf["type"] == "slider":
                params[pn] = st.slider(pn, pconf["min"], pconf["max"], pconf["default"], pconf.get("step", 1) if isinstance(pconf.get("step"), (int, float)) else 1.0)
            elif pconf["type"] == "select":
                params[pn] = st.selectbox(pn, pconf["options"], index=pconf["options"].index(pconf["default"]) if pconf["default"] in pconf["options"] else 0)

    ac1, ac2 = st.columns(2)
    test_size = ac1.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
    cv_folds = ac2.slider("CV 折数", 2, 10, 5)
    train_btn = st.button("🚀 开始训练", use_container_width=True, type="primary")

# ── Data Preview ──
with st.expander("📋 数据预览", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", len(data))
    c2.metric("特征数", len(data.columns) - 1)
    c3.metric("类别数", data[label_column].nunique())

# ── Training ──
if train_btn:
    bar = st.progress(0, "准备中...")
    project_dir = create_project_directory()
    bar.progress(15, "计算指纹...")
    fp_file = save_fingerprint_data(data, project_dir, label_column)
    if fp_file is None: st.error("无法生成指纹"); st.stop()
    bar.progress(35, f"训练 {model_name}...")
    model, metrics = train_model(fp_file, project_dir, model_name, params, test_size, cv_folds)
    if model is None: st.error("训练失败"); st.stop()
    bar.progress(65, "生成图表...")
    dp = preprocess_data(fp_file)
    generate_evaluation_plots(metrics, project_dir, dp.columns[:-1].tolist())
    info = {'model_name': model_name, 'params': params, 'dataset': dataset_choice, 'label_column': label_column,
            'metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items() if k not in ['fpr', 'tpr', 'confusion_matrix', 'feature_importance']},
            'timestamp': datetime.now().isoformat()}
    with open(os.path.join(project_dir, "info.json"), 'w') as f: json.dump(info, f, indent=2, default=str)
    bar.progress(100, "完成！")
    st.success(f"模型已保存: `{project_dir}`")

    st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin:1.5rem 0 0.5rem;">性能指标</p>', unsafe_allow_html=True)
    mc = st.columns(6)
    for i, (k, v) in enumerate([('Accuracy', metrics.get('accuracy',0)), ('Precision', metrics.get('precision',0)), ('Recall', metrics.get('recall',0)), ('F1', metrics.get('f1',0)), ('ROC AUC', metrics.get('roc_auc',0) or 0), ('MCC', metrics.get('mcc',0))]):
        mc[i].metric(k, f"{v:.3f}")
    if metrics.get('cv_mean'): st.info(f"🔁 {cv_folds}-折 CV Accuracy: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

    st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin:1.5rem 0 0.5rem;">评估图表</p>', unsafe_allow_html=True)
    pt1, pt2, pt3, pt4 = st.tabs(["ROC 曲线", "混淆矩阵", "特征重要性", "指标总览"])
    for tab, fn in [(pt1, "roc_curve.png"), (pt2, "confusion_matrix.png"), (pt3, "feature_importance.png"), (pt4, "metrics_summary.png")]:
        with tab:
            p = os.path.join(project_dir, fn)
            if os.path.exists(p): st.image(p, use_container_width=True)
            else: st.info("图表不可用")

    st.markdown('<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin:1.5rem 0 0.5rem;">分类报告</p>', unsafe_allow_html=True)
    X = dp.iloc[:, :-1]; y = dp.iloc[:, -1]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    rpt = classification_report(yv, model.predict(Xv), output_dict=True)
    st.dataframe(pd.DataFrame(rpt).transpose().style.format("{:.3f}"), use_container_width=True)

render_footer()
