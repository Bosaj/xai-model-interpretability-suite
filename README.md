<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,3,5,30&height=200&section=header&text=ENIAD%20Explainable%20AI%20(XAI)%20Suite&fontSize=32&animation=twinkling&fontAlignY=35&desc=ENIAD%20Berkane%20%7C%20Engineering%20Curriculum%20Laboratory%20Suite&descSize=14&descAlignY=55" alt="ENIAD Explainable AI (XAI) Suite Banner" width="100%" />

<!-- Typing Animation -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&repeat=true&width=800&height=40&lines=Explainable%20AI%20(XAI)%20Suite;PDP%20and%20ALE%20Interpretability;SHAP%20and%20LIME%20Feature%20Attributions;Grad-CAM%20Computer%20Vision%20Interpretability" alt="Typing SVG" />
</p>

<!-- Quality & Community Badges -->
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT License" /></a>
  <a href="https://github.com/Bosaj/xai-model-interpretability-suite/actions"><img src="https://img.shields.io/badge/CI%20Pipeline-Passing-brightgreen?style=flat-square&logo=githubactions" alt="CI Status" /></a>
  <a href="https://github.com/Bosaj/xai-model-interpretability-suite/stargazers"><img src="https://img.shields.io/github/stars/Bosaj/xai-model-interpretability-suite?style=flat-square&logo=github&color=00d9ff" alt="Stars" /></a>
  <a href="https://github.com/users/Bosaj/projects/36"><img src="https://img.shields.io/badge/Project_Board-Project_36-blue?style=flat-square&logo=github" alt="Project Board" /></a>
  <a href="https://github.com/stars/Bosaj/lists/eniad-academic-projects"><img src="https://img.shields.io/badge/Curated_List-ENIAD_Academic_Projects-gold?style=flat-square&logo=github" alt="Curated List" /></a>
  <img src="https://img.shields.io/badge/Institution-ENIAD%20Berkane-FF6B00?style=flat-square" alt="ENIAD Berkane" />
</p>

</div>

<!-- Divider -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="Divider" width="100%" />

## 📖 Overview

**ENIAD Explainable AI (XAI) Suite** is an official engineering laboratory suite developed within the **State Engineering Degree in Artificial Intelligence & Digital Systems** at the **École Nationale d'Intelligence Artificielle et du Digital (ENIAD)**, Mohammed First University, Berkane, Morocco.

Engineering Suite for Machine Learning Model Interpretability: PDP, ALE, KernelSHAP, TreeSHAP, LIME and Grad-CAM Saliency Maps.

---

## 🏗️ Technical Architecture

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

## 📂 Curriculum & Laboratory Breakdown

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

## 📚 Technical Wiki & Documentation

Comprehensive architectural explanations, step-by-step lab walk-throughs, and methodology guides are available:
- **In-Repository Wiki Mirror**: [`docs/wiki/Home.md`](docs/wiki/Home.md)
- **Architecture Overview**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Curriculum Matrix**: [`docs/CURRICULUM_MATRIX.md`](docs/CURRICULUM_MATRIX.md)
- **GitHub Wiki**: [https://github.com/Bosaj/xai-model-interpretability-suite/wiki](https://github.com/Bosaj/xai-model-interpretability-suite/wiki)

---

## 🚀 Getting Started

### Prerequisites
- Git installed on your local workstation
- Development runtime corresponding to the target laboratory (Python 3.10+, Java JDK 17+, Android Studio, or C++ compiler)

### Installation & Cloning
```bash
git clone https://github.com/Bosaj/xai-model-interpretability-suite.git
cd xai-model-interpretability-suite
```

---

## 📜 DevSecOps Governance & Standards

This repository adheres to strict open-source engineering and academic integrity standards:
- [LICENSE](LICENSE): Open-source MIT License.
- [CONTRIBUTING.md](CONTRIBUTING.md): Guidelines for submitting contributions, issue templates, and code formatting.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): Contributor Covenant v2.1 code of conduct.
- [SECURITY.md](SECURITY.md): Responsible vulnerability reporting procedures.
- [CHANGELOG.md](CHANGELOG.md): Version history following Keep a Changelog standards.
- [CITATION.cff](CITATION.cff): Machine-readable academic citation metadata.

---

## 👤 Author & Academic Credits

- **Engineer / Researcher**: **Oussama EL HADJI** ([@Bosaj](https://github.com/Bosaj))
- **Role**: AI & Automation Engineer @ Circet Morocco | ENIAD Engineering Graduate
- **Institution**: École Nationale d'Intelligence Artificielle et du Digital (ENIAD), Berkane, Morocco
- **Portfolio**: [bosaj.vercel.app](https://bosaj.vercel.app) • [LinkedIn](https://www.linkedin.com/in/oussama-elhadji)

---

<div align="center">
  <sub>Maintained with ❤️ by <a href="https://github.com/Bosaj">Oussama EL HADJI</a> • ENIAD Berkane</sub>
</div>
