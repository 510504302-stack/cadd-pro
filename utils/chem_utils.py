"""
Chemical utility functions: fingerprint generation, similarity, molecule processing.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
import streamlit as st
from typing import Optional, List, Tuple


def mol_to_fp(smiles: str, radius: int = 2, fp_size: int = 2048) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fp = fpgen.GetFingerprint(mol)
    arr = np.zeros((fp_size,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(
        DataStructs.CreateFromBitString("".join(str(int(b)) for b in fp1)),
        DataStructs.CreateFromBitString("".join(str(int(b)) for b in fp2))
    )


def find_similar_molecules(
    query_smiles: str,
    reference_smiles_list: List[str],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """Find top-k most similar molecules from a reference list."""
    query_fp = mol_to_fp(query_smiles)
    if query_fp is None:
        return []

    similarities = []
    for smi in reference_smiles_list:
        ref_fp = mol_to_fp(smi)
        if ref_fp is not None:
            sim = DataStructs.TanimotoSimilarity(
                DataStructs.CreateFromBitString("".join(str(int(b)) for b in query_fp)),
                DataStructs.CreateFromBitString("".join(str(int(b)) for b in ref_fp))
            )
            similarities.append((smi, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def draw_molecule(smiles: str, size: Tuple[int, int] = (400, 300)):
    """Draw a molecule structure as an image."""
    try:
        from rdkit.Chem import Draw
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size)


def show_molecule(smiles: str, width: int = 350, height: int = 250, caption: str = "分子结构"):
    """Render a molecule using JavaScript SmilesDrawer (works without RDKit Draw)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning("无效的 SMILES")
        return

    # Generate canonical SMILES for consistent rendering
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/smiles-drawer@2.0.3/dist/smiles-drawer.min.js"></script>
        <style>
            body {{ margin: 0; display: flex; justify-content: center; align-items: center; background: transparent; }}
            canvas {{ display: block; }}
        </style>
    </head>
    <body>
        <canvas id="mol-canvas"></canvas>
        <script>
            (function() {{
                let smiles = "{canonical_smi}";
                let options = {{
                    width: {width},
                    height: {height},
                    bondThickness: 2.0,
                    bondLength: 20.0,
                    shortBondLength: 0.85,
                    bondSpacing: 0.18 * 20.0,
                    atomVisualization: 'default',
                    isomeric: true,
                    debug: false,
                    terminalCarbons: true,
                    explicitHydrogens: false,
                    overlapSensitivity: 0.42,
                    overlapResolutionIterations: 1,
                    compactDrawing: true,
                    fontFamily: 'Arial, sans-serif',
                    fontSizeLarge: 11,
                    fontSizeSmall: 8,
                    padding: 20.0,
                }};
                let canvas = document.getElementById('mol-canvas');
                let drawer = new SmilesDrawer.Drawer(options);
                SmilesDrawer.parse(smiles, function(tree) {{
                    drawer.draw(tree, canvas, options);
                }}, function(err) {{
                    console.error(err);
                }});
            }})();
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=height + 20, scrolling=False)


def save_fingerprint_data(data: pd.DataFrame, project_dir: str, label_column: str) -> Optional[str]:
    """Save fingerprint features and labels to CSV for model training."""
    # Detect SMILES column
    columns_name = None
    for candidate in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles']:
        if candidate in data.columns:
            columns_name = candidate
            break

    if columns_name is None:
        st.error('无法找到 SMILES 列！请确保数据包含 "smiles" 或 "SMILES" 列。')
        return None

    fingerprints = []
    valid_indices = []
    for idx, smi in enumerate(data[columns_name]):
        fp = mol_to_fp(smi)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)

    if not fingerprints:
        st.error("没有有效的SMILES可以转换。")
        return None

    fingerprint_df = pd.DataFrame(fingerprints)
    fingerprint_df['label'] = data.iloc[valid_indices][label_column].values
    output_file = os.path.join(project_dir, "input.csv")
    fingerprint_df.to_csv(output_file, index=False)
    return output_file


def batch_predict(model, smiles_list: List[str]) -> pd.DataFrame:
    """Predict activity for a batch of SMILES strings."""
    results = []
    for smi in smiles_list:
        fp = mol_to_fp(smi)
        if fp is not None:
            pred = model.predict([fp])[0]
            proba = model.predict_proba([fp])[0]
            results.append({
                'SMILES': smi,
                'Prediction': int(pred),
                'Probability_Class0': float(proba[0]),
                'Probability_Class1': float(proba[1]),
                'Valid': True
            })
        else:
            results.append({
                'SMILES': smi,
                'Prediction': None,
                'Probability_Class0': None,
                'Probability_Class1': None,
                'Valid': False
            })
    return pd.DataFrame(results)
