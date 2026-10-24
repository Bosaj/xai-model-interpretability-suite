# Architecture & Technical Design 🏗️

This chapter outlines the engineering architecture, data pipelines, and modular subsystems of **ENIAD Explainable AI (XAI) Suite**.

---

## 🧩 Architectural Blueprint

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

## ⚙️ Design Principles

1. **Modularity**: Each laboratory exercise is isolated and self-contained with minimal external side-effects.
2. **Reproducibility**: Clear seed parameters, deterministic executions, and explicit environment manifests.
3. **Academic Rigor**: High adherence to theoretical foundations combined with production-grade engineering practices.
