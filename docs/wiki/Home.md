# Welcome to the ENIAD Explainable AI (XAI) Suite Documentation Wiki 📖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Institution: ENIAD Berkane](https://img.shields.io/badge/Institution-ENIAD%20Berkane-FF6B00?style=flat-square)](https://github.com/Bosaj/xai-model-interpretability-suite)
[![Project Board](https://img.shields.io/badge/Project_Board-Project_36-blue?style=flat-square&logo=github)](https://github.com/users/Bosaj/projects/36)
[![Curated List](https://img.shields.io/badge/Curated_List-ENIAD_Academic_Projects-gold?style=flat-square&logo=github)](https://github.com/stars/Bosaj/lists/eniad-academic-projects)

Welcome to the official technical documentation and engineering reference for **ENIAD Explainable AI (XAI) Suite**.

---

## 🎯 Academic & Technical Mission

Engineering Suite for Machine Learning Model Interpretability: PDP, ALE, KernelSHAP, TreeSHAP, LIME and Grad-CAM Saliency Maps.

Developed within the **State Engineering Degree in Artificial Intelligence & Digital Systems** at the **École Nationale d'Intelligence Artificielle et du Digital (ENIAD)**, Berkane, Morocco.

---

## 📚 Wiki Documentation Chapters

| Chapter | Description | Primary Topics |
| :--- | :--- | :--- |
| **[[Architecture]]** | Deep architectural design & component topology | Mermaid diagrams, component interactions, runtime environment |
| **[[Getting-Started]]** | Developer setup & workstation configuration | Prerequisites, toolchain setup, execution commands |
| **[[Curriculum-Guide]]** | Detailed syllabus and laboratory breakdown | Lab objectives, expected outputs, deliverables |

---

## 🏛️ System Architecture Snapshot

```mermaid
graph TD
    A[Raw Dataset / Input Features] --> B[Black-Box ML Model]
    B --> C{Explainability Engine}
    C -->|Global Interpretability| D[PDP & ALE Plots]
    C -->|Local Attribution| E[TreeSHAP & KernelSHAP]
    C -->|Perturbation Analysis| F[LIME Local Surrogate]
    C -->|Computer Vision| G[Grad-CAM Saliency Maps]
    D --> H[Auditing & Fairness Diagnostics]
    E --> H
    F --> H
    G --> H
    style A fill:#00D9FF,stroke:#333,stroke-width:1px,color:#000
    style B fill:#FF6B00,stroke:#333,stroke-width:1px,color:#fff
    style C fill:#3C873A,stroke:#333,stroke-width:1px,color:#fff
    style H fill:#7928CA,stroke:#333,stroke-width:1px,color:#fff

```

---

## 🔬 Key Methodologies Detailed

### 1. Partial Dependence Plots (PDP)
Partial dependence plots show the marginal effect one or two features have on the predicted outcome of a machine learning model. A partial dependence plot can show whether the relationship between the target and a feature is linear, monotonic or more complex.

### 2. Accumulated Local Effects (ALE)
ALE plots describe how features influence the prediction of a machine learning model on average. ALE plots are faster and unbiased alternatives to partial dependence plots (PDPs) when features are correlated.

### 3. SHAP (SHapley Additive exPlanations)
SHAP explains the prediction of an instance $x$ by computing the contribution of each feature to the prediction using coalitional game theory.

### 4. LIME (Local Interpretable Model-agnostic Explanations)
LIME explains individual predictions by approximating black-box models locally with an interpretable surrogate model (such as a sparse linear regression or decision tree).


---

*Maintained with ❤️ by [Oussama EL HADJI](https://github.com/Bosaj) • ENIAD Berkane*
