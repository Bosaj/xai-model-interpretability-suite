# XAI Model Interpretability Suite 🔬

[![CI & Observability](https://img.shields.io/badge/CI%2FCD-Passing-success?logo=githubactions&logoColor=white)](https://github.com/Bosaj/xai-model-interpretability-suite/actions)
[![SLSA Attestation](https://img.shields.io/badge/SLSA%20Level%203-Attested-blue?logo=githubactions&logoColor=white)](https://github.com/Bosaj/xai-model-interpretability-suite/attestations)
[![GHCR Container](https://img.shields.io/badge/GHCR-ghcr.io%2Fbosaj%2Fxai-model-interpretability-suite-brightgreen?logo=docker&logoColor=white)](https://github.com/Bosaj?tab=packages)
[![Project Roadmap](https://img.shields.io/badge/Project%20Roadmap-%2336-8A2BE2?logo=github&logoColor=white)](https://github.com/users/Bosaj/projects/36)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Space-yellow?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/bosaj/xai-model-interpretability-suite)
[![Live Web App](https://img.shields.io/badge/Live%20Web%20App-XAI%20Explorer-brightgreen?logo=googlechrome&logoColor=white)](https://bosaj-xai-model-interpretability-suite.static.hf.space)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20XAI%20Explorer-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

An end-to-end Explainable AI (XAI) exploration and evaluation suite covering academic practical works (TP1–TP5) on modern model interpretability techniques, complete with an interactive **Public Web Explorer Dashboard**: [https://bosaj-xai-model-interpretability-suite.static.hf.space](https://bosaj-xai-model-interpretability-suite.static.hf.space).

---

## 🌟 Interactive Streamlit Dashboard

Run the standalone interactive exploration dashboard locally or deploy to Streamlit Cloud:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit Explorer
streamlit run streamlit_app.py
```

### Dashboard Modules:
- **📊 Overview & Dataset Explorer**: Live inspection of Abalone physical measurement distributions, correlation matrices, and model benchmarks.
- **🌊 SHAP Waterfall & Beeswarm (TP4)**: Interactive Shapley value decomposition for individual tabular predictions with baseline reference.
- **📈 Partial Dependence Plots (PDP & ICE) (TP2)**: One-way and two-way feature marginal effect curves with interactive target selection.
- **🍋 LIME Tabular Explanations (TP3)**: Local linear surrogate models with Ridge regression and perturbation neighborhood analysis.
- **👁️ Computer Vision Saliency & Heatmaps (TP5)**: Upload custom images or inspect synthetic diagnostic benchmarks with Grad-CAM and Sobel edge saliency overlays.

---

## 📚 Academic TPs Curriculum Coverage

| Module | Core Topics | Methodologies & Libraries |
| :--- | :--- | :--- |
| **TP1** | Feature Importance & Global Surrogates | Permutation Importance, Decision Tree Surrogates, `scikit-learn` |
| **TP2** | Partial Dependence & Marginal Effects | One-way PDP, ICE individual curves, `scikit-learn.inspection` |
| **TP3** | Local Interpretable Model-agnostic Explanations | Perturbation sampling, exponential kernel weights, Ridge surrogate |
| **TP4** | Shapley Additive Explanations (SHAP) | Game-theoretic attributions, TreeExplainer, Waterfall plots, `shap` |
| **TP5** | Computer Vision Interpretability | Grad-CAM activation mapping, Saliency gradients, Heatmap blending |

---

## 🚀 Quick Start

### Option A: Local Virtual Environment
```bash
git clone https://github.com/Bosaj/xai-model-interpretability-suite.git
cd xai-model-interpretability-suite
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Option B: Docker Container
```bash
docker build -t xai-model-interpretability-suite .
docker run -p 8501:8501 xai-model-interpretability-suite
```

---

## 👤 Author

**Oussama EL HADJI (Bosaj)**
- GitHub: [@Bosaj](https://github.com/Bosaj)
- Organization: **ENIAD — École Nationale d'Intelligence Artificielle et du Digital**