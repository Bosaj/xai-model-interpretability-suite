# 🔍 ENIAD Explainable AI (XAI) Suite

Comprehensive model interpretability, auditing, and feature attribution laboratory suite developed for the **Explainable AI** engineering module (Semestre 9) at **ENIAD** (*École Nationale d'Intelligence Artificielle et du Digital*).

---

## 🎯 Architectural Overview
This repository contains end-to-end algorithmic implementations and empirical evaluations of state-of-the-art interpretability techniques:
- **Accumulated Local Effects (ALE)**: 1D & 2D manual and automated ALE computations isolating feature effects without extrapolation bias.
- **Partial Dependence Plots (PDP)**: Global marginal feature analysis and non-additive interaction discovery.
- **SHAP (Shapley Additive exPlanations)**: KernelSHAP comparison, cooperative game theory attribution, and local/global explanations.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Tabular and visual explanation models for deep convolutional classifiers.
- **What-If & Counterfactual Analysis**: Sensitivity probing across decision boundaries.

## 📊 Benchmarked Datasets
- German Credit Risk Assessment (`credit.csv`)
- FIFA Player Attribute Valuation (`fifa.csv`)
- Agricultural Wheat Seed Geometry (`wheat_seeds.csv`)
- King County Housing & Abalone Regressions (`kc_house_data.csv`, `abalone.data`)

## 🛠️ Tech Stack
- **Python 3.11+**, **Jupyter Notebooks**
- **Scikit-Learn**, **SHAP**, **LIME**, **Matplotlib**, **Seaborn**, **NumPy**, **Pandas**

---
*Author: Oussama EL HADJI (@Bosaj)*
