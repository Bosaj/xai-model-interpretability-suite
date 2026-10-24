# Academic Curriculum & Laboratory Guide 📂

This guide presents the complete educational structure, competency mapping, and lab sequence for **ENIAD Explainable AI (XAI) Suite**.

---

## 📑 Curriculum Matrix

| Laboratory / Directory | Topic | Key Techniques & Deliverables |
|---|---|---|
| [`Cours/`](Cours/) | Theoretical XAI Foundations | Interpretability taxonomy, algorithmic fairness, fidelity vs. interpretability |
| [`TP1/`](TP1/) | Partial Dependence Plots (PDP) | Marginal effect analysis, 1D and 2D interaction visualizers |
| [`TP2/`](TP2/) | Accumulated Local Effects (ALE) | Handling correlated features, unbiased localized attribution |
| [`TP3/`](TP3/) | SHAP (Shapley Values) | Cooperative game theory, TreeSHAP, KernelSHAP, force & summary plots |
| [`TP4/`](TP4/) | LIME Local Surrogates | Sparse linear explainers, image and tabular perturbation sampling |
| [`TP5/`](TP5/) | Deep Learning Saliency | Grad-CAM, gradient-weighted class activation mapping |
| [`ELHADJI Oussama/`](ELHADJI%20Oussama/) | Research Project Report | Detailed experimental validation, comparative evaluation matrix |


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

