# Explainable AI (XAI) - Devoir 2025-2026

**Auteur:** ELHADJI Oussama  
**Encadré par:** Pr. Mm S. MHAMMEDI  
**Date:** Janvier 2026

## Description
Ce dépôt contient les travaux pratiques et le rapport complet pour le module d'Intelligence Artificielle Explicable (XAI). Le projet explore et implémente plusieurs méthodes d'interprétabilité des modèles de Machine Learning, allant des méthodes globales (PDP, ALE) aux méthodes locales (SHAP, LIME) et contre-factuelles.

## Structure du Projet

```
Devoir XAI/
├── Dataset/
├    ├── credit.csv            # Données (German Credit)
├    ├── fifa.csv              # Données (FIFA Man of the Match)
├    ├── wheat_seeds.csv       # Données (Wheat Seeds)
├    ├── datasett.py         
├── utils/
├    ├── lime.py
├    ├── whatif.py          # Scripts utilitaires et helper functions
├── outputs/                # Graphiques et visualisations générés
├── Exercice_1.ipynb        # PDP et ALE (German Credit)
├── Exercice_2.ipynb        # SHAP (FIFA Man of the Match)
├── Exercice_3.ipynb   # LIME (SVM Multiclass sur Wheat Seeds)
├── Exercice_4.ipynb        # Contre-factuels (What-If sur Iris)
├── main.tex                # Code source LaTeX du rapport
└── main.pdf                # Rapport final compilé
```

## Contenu Détaillé

### Exercice 1 : Méthodes Globales (PDP & ALE)
Analyse approfondie du dataset **German Credit** utilisant un modèle Random Forest.
- **Objectifs :** Comprendre les effets marginaux des variables (Global Interpretability).
- **Méthodes :** Partial Dependence Plots (PDP) et Accumulated Local Effects (ALE).
- **Résultats :** Mise en évidence de biais (genre, emploi) et d'interactions complexes masqués par les analyses classiques.

### Exercice 2 : Valeurs de Shapley (SHAP)
Application de la théorie des jeux pour expliquer les prédictions individuelles sur le dataset **FIFA**.
- **Objectifs :** Attribuer une contribution juste à chaque feature pour une prédiction donnée.
- **Méthodes :** Implémentation manuelle de KernelSHAP.
- **Résultats :** Identification des facteurs clés (Possession, Tirs) influençant l'élection de l'homme du match.

### Exercice 3 : LIME (Local Interpretable Model-agnostic Explanations)
Interprétation locale d'un modèle "boîte noire" complexe (**SVM à noyau RBF**) sur le dataset **Wheat Seeds**.
- **Objectifs :** Approximer localement un modèle complexe par un modèle simple (Arbre de décision).
- **Méthodes :** Implémentation de LIME from scratch (échantillonnage, pondération exponentielle).
- **Résultats :** Visualisation des frontières de décision et analyse de la fidélité de l'explication locale.

### Exercice 4 : Explications Contre-factuelles (What-If)
Analyse de scénarios alternatifs sur le dataset **Iris**.
- **Objectifs :** Déterminer le changement minimal nécessaire pour modifier la classe prédite.
- **Méthodes :** Recherche du voisin le plus proche d'une classe différente (Gower Distance).
- **Résultats :** Génération d'explications contrastives et analyse de leur minimalité et actionnabilité.

## Installation et Prérequis

Le projet nécessite **Python 3.x**. Les dépendances principales sont :

- `pandas` & `numpy` : Manipulation de données
- `matplotlib` & `seaborn` : Visualisation
- `scikit-learn` : Modélisation (Random Forest, SVM)
- `ConfigSpace` : Gestion des configurations (utilisé par `Dataset`)
- `gower` : Calcul de distances pour données mixtes (Exercice 4)

Installation recommandée :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ConfigSpace gower
```

## Note sur les Données
Le dataset `wheat_seeds.csv` nécessaire pour l'Exercice 3 a été restauré dans le dossier `Dataset/`. Assurez-vous que tous les fichiers CSV sont présents avant l'exécution.

## Rapport
Le fichier `main.pdf` contient le compte-rendu détaillé, incluant toutes les figures, les interprétations des résultats et les réponses aux questions théoriques.
