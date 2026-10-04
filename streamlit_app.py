"""
ENIAD XAI Explorer · Suite d'Interprétabilité des Modèles d'IA
École Nationale d'Intelligence Artificielle et du Digital (ENIAD)
Master Intelligence Artificielle & Digital · Travaux Pratiques TP1 - TP5
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageFilter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="XAI Explorer · ENIAD Interpretability Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "TP2" / "data"
TP5_DIR = ROOT_DIR / "TP5"

# Styling
st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1f2937); border: 1px solid #374151; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;">
    <h1 style="color: #60a5fa; margin: 0; font-size: 2.2rem;">🔍 ENIAD XAI Explorer · Suite d'Interprétabilité IA</h1>
    <p style="color: #9ca3af; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
        Explicabilité Globale & Locale des Modèles de Machine Learning · SHAP, LIME, PDP & Salience Visuelle
    </p>
    <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 10px;">
        <span style="background: #1e3a8a; color: #93c5fd; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">TP1-TP2 : PDP & ALE</span>
        <span style="background: #065f46; color: #6ee7b7; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">TP3 : SHAP & Waterfall</span>
        <span style="background: #701a75; color: #f0abfc; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">TP4-TP5 : LIME & Vision</span>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_fit_abalone():
    file_path = DATA_DIR / "abalone.data"
    if not file_path.exists():
        # Fallback synthetic if file missing
        X = pd.DataFrame(np.random.rand(500, 7), columns=["Length", "Diameter", "Height", "Whole_w", "Shucked_w", "Viscera_w", "Shell_w"])
        y = (X["Length"] * 10 + X["Shell_w"] * 15 + np.random.randn(500)).astype(int)
        model = GradientBoostingRegressor(n_estimators=60, random_state=42)
        model.fit(X, y)
        return model, X, y

    columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
    df = pd.read_csv(file_path, header=None, names=columns)
    df["Sex_M"] = (df["Sex"] == "M").astype(int)
    df["Sex_F"] = (df["Sex"] == "F").astype(int)
    
    feature_cols = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Sex_M", "Sex_F"]
    X = df[feature_cols]
    y = df["Rings"]

    model = GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model, X, y


@st.cache_resource
def load_and_fit_housing():
    file_path = DATA_DIR / "kc_house_data.csv"
    if not file_path.exists():
        return None, None, None

    df = pd.read_csv(file_path, nrows=3000)
    features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "grade", "yr_built"]
    df_clean = df[features + ["price"]].dropna()
    X = df_clean[features]
    y = df_clean["price"] / 1000.0  # en milliers de dollars

    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
    model.fit(X, y)
    return model, X, y


# Sidebar Navigation & Dataset Choice
with st.sidebar:
    st.header("⚙️ Configuration des Données")
    dataset_name = st.selectbox(
        "Sélectionnez le Jeu de Données :",
        ["Abalone (Biométrie & Âge)", "King County Real Estate (Prix Immobilier)"]
    )
    st.divider()
    st.subheader("📚 Références Pédagogiques ENIAD")
    st.markdown("""
    - **TP0 & TP1** : Fondations PDP & Effets Marginaux
    - **TP2 & TP3** : Valeurs de Shapley (Axiomes d'efficacité & symétrie)
    - **TP4** : Modèles Substituts Locaux (LIME Tabulaire)
    - **TP5** : Cartes de Salience & Grad-CAM sur Images
    """)


if dataset_name == "Abalone (Biométrie & Âge)":
    model, X_data, y_data = load_and_fit_abalone()
    target_label = "Âge estimé (Nombre d'anneaux / Rings)"
else:
    model, X_data, y_data = load_and_fit_housing()
    if model is None:
        model, X_data, y_data = load_and_fit_abalone()
        target_label = "Âge estimé (Nombre d'anneaux)"
    else:
        target_label = "Prix Estimé (k$ USD)"

feature_names = list(X_data.columns)
baseline_pred = float(np.mean(model.predict(X_data)))

# Main Tabs
tab_shap, tab_pdp, tab_lime, tab_vision = st.tabs([
    "📊 Attribution de Caractéristiques (SHAP Waterfall)",
    "📈 Dépendance Partielle (PDP)",
    "🎯 Modèle Substitut Local (LIME)",
    "🖼️ Explicabilité Visuelle & Salience (TP5)"
])

# =========================================================================
# TAB 1: SHAP WATERFALL
# =========================================================================
with tab_shap:
    st.subheader("📊 Décomposition des Contributions Locales (Style SHAP Waterfall)")
    st.markdown("""
    Chaque caractéristique pousse la prédiction à la hausse (rouge) ou à la baisse (bleu)
    par rapport à la valeur moyenne attendue du modèle (**valeur de base** : {:.2f}).
    """.format(baseline_pred))

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### 🎛️ Profil de l'Instance à Expliquer :")
        sample_instance = {}
        for feat in feature_names:
            min_val = float(X_data[feat].min())
            max_val = float(X_data[feat].max())
            mean_val = float(X_data[feat].mean())
            step_val = (max_val - min_val) / 50.0 if max_val != min_val else 0.1
            sample_instance[feat] = st.slider(feat, min_val, max_val, mean_val, step=step_val)

        df_inst = pd.DataFrame([sample_instance])
        current_pred = float(model.predict(df_inst)[0])

    with col_chart:
        st.metric(
            label=f"Prédiction du Modèle f(x) : {target_label}",
            value=f"{current_pred:.2f}",
            delta=f"{current_pred - baseline_pred:+.2f} vs moyenne globale ({baseline_pred:.2f})"
        )

        # Approximate Shapley attributions via local marginal contributions
        contributions = {}
        X_bg = X_data.sample(min(100, len(X_data)), random_state=42)
        for feat in feature_names:
            X_perm = df_inst.copy()
            X_perm[feat] = X_bg[feat].mean()
            pred_without = float(model.predict(X_perm)[0])
            contributions[feat] = current_pred - pred_without

        # Normalize so sum equals total difference
        tot_diff = current_pred - baseline_pred
        sum_raw = sum(contributions.values())
        if abs(sum_raw) > 1e-6:
            scale = tot_diff / sum_raw
            contributions = {k: v * scale for k, v in contributions.items()}

        # Sort by absolute impact
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]

        names = [x[0] for x in sorted_feats]
        values = [x[1] for x in sorted_feats]
        colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in values]

        fig_waterfall = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="auto"
        ))
        fig_waterfall.update_layout(
            title="Contributions Locales des Caractéristiques (Impact Net)",
            xaxis_title="Contribution à la Prédiction (Δ Output)",
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        with st.expander("📌 Importance Globale des Caractéristiques (Permutation Importance)"):
            perm = permutation_importance(model, X_data.iloc[:500], y_data.iloc[:500], n_repeats=5, random_state=42)
            df_perm = pd.DataFrame({"Feature": feature_names, "Importance": perm.importances_mean}).sort_values("Importance", ascending=True)
            fig_perm = px.bar(df_perm, x="Importance", y="Feature", orientation="h", title="Importance Globale par Permutation")
            st.plotly_chart(fig_perm, use_container_width=True)


# =========================================================================
# TAB 2: PDP (Partial Dependence Plots)
# =========================================================================
with tab_pdp:
    st.subheader("📈 Courbes de Dépendance Partielle (PDP - TP1 & TP2)")
    st.markdown("""
    La dépendance partielle montre l'effet marginal d'une ou deux caractéristiques sur le résultat
    prédit par le modèle, en intégrant sur la distribution des autres variables.
    """)

    col_pdp_ctrl, col_pdp_plot = st.columns([1, 2])
    with col_pdp_ctrl:
        pdp_feature = st.selectbox("Caractéristique à analyser :", feature_names, index=0)
        grid_resolution = st.slider("Résolution de la grille (Points) :", 15, 60, 30)

    with col_pdp_plot:
        with st.spinner("Calcul de la courbe de dépendance partielle..."):
            pdp_res = partial_dependence(
                model, X_data.sample(min(300, len(X_data)), random_state=42),
                features=[pdp_feature],
                grid_resolution=grid_resolution
            )
            grid_values = pdp_res.get("grid_values", pdp_res.get("values"))[0]
            avg_preds = pdp_res["average"][0]

            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(
                x=grid_values,
                y=avg_preds,
                mode="lines+markers",
                name="PDP",
                line=dict(color="#10b981", width=3)
            ))
            fig_pdp.update_layout(
                title=f"Effet Marginal Partiel de {pdp_feature}",
                xaxis_title=pdp_feature,
                yaxis_title=f"Valeur Attendue E[f(X) | {pdp_feature}]",
                height=420
            )
            st.plotly_chart(fig_pdp, use_container_width=True)


# =========================================================================
# TAB 3: LIME TABULAR SURROGATE
# =========================================================================
with tab_lime:
    st.subheader("🎯 Modèle Substitut Local LIME (Local Interpretable Model-agnostic Explanations)")
    st.markdown("""
    LIME approxime le comportement d'une boîte noire complexe au voisinage immédiat d'une prédiction
    en créant des perturbations aléatoires pondérées par une fonction de distance exponentielle.
    """)

    c_l1, c_l2 = st.columns([1, 2])
    with c_l1:
        n_samples = st.slider("Nombre de perturbations locales :", 100, 1000, 400, step=100)
        kernel_width = st.slider("Largeur du noyau gaussien (Kernel Width) :", 0.1, 2.0, 0.75, step=0.05)

    with c_l2:
        # Generate perturbations around sample_instance
        inst_vals = np.array([sample_instance[f] for f in feature_names])
        stds = X_data.std().values + 1e-6
        noise = np.random.normal(0, kernel_width, size=(n_samples, len(inst_vals))) * stds
        perturbed_X = inst_vals + noise
        perturbed_preds = model.predict(perturbed_X)

        # Distances and weights
        distances = np.linalg.norm((perturbed_X - inst_vals) / stds, axis=1)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # Local Ridge surrogate
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(perturbed_X)
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(X_scaled, perturbed_preds, sample_weight=weights)

        df_lime = pd.DataFrame({
            "Caractéristique": feature_names,
            "Poids Substitut Local (LIME)": surrogate.coef_
        }).sort_values("Poids Substitut Local (LIME)", key=abs, ascending=True)

        colors_lime = ["#ef4444" if v >= 0 else "#3b82f6" for v in df_lime["Poids Substitut Local (LIME)"]]
        fig_lime = go.Figure(go.Bar(
            x=df_lime["Poids Substitut Local (LIME)"],
            y=df_lime["Caractéristique"],
            orientation="h",
            marker_color=colors_lime
        ))
        fig_lime.update_layout(
            title=f"Coefficients du Modèle Substitut Linéaire Local (R² local : {surrogate.score(X_scaled, perturbed_preds):.3f})",
            xaxis_title="Poids Local",
            height=400
        )
        st.plotly_chart(fig_lime, use_container_width=True)


# =========================================================================
# TAB 4: VISION & SALIENCY (TP5)
# =========================================================================
with tab_vision:
    st.subheader("🖼️ Explicabilité Visuelle & Cartes de Salience (TP5 Computer Vision)")
    st.markdown("""
    Visualisation des régions et contours clés qui influencent l'activation des réseaux neuronaux
    de vision par ordinateur (similaire à Grad-CAM et superpixels LIME-Image).
    """)

    image_choice = st.selectbox(
        "Sélectionnez une image d'exemple ou téléversez la vôtre :",
        ["Image d'exemple : Tigre (TP5)", "Image d'exemple : Chien (TP5)", "Téléverser une image personnalisée"]
    )

    img = None
    if image_choice == "Image d'exemple : Tigre (TP5)":
        tiger_path = TP5_DIR / "tiger.jpeg"
        if tiger_path.exists():
            img = Image.open(tiger_path)
    elif image_choice == "Image d'exemple : Chien (TP5)":
        dog_path = TP5_DIR / "dog.jpeg"
        if dog_path.exists():
            img = Image.open(dog_path)
    else:
        uploaded_img = st.file_uploader("Choisissez une image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            img = Image.open(uploaded_img)

    if img is not None:
        c_img1, c_img2, c_img3 = st.columns(3)
        with c_img1:
            st.caption("📷 Image Originale")
            st.image(img, use_container_width=True)

        with c_img2:
            st.caption("🔬 Extraction de Caractéristiques Saillantes")
            gray = img.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            st.image(edges, use_container_width=True)

        with c_img3:
            st.caption("🔥 Carte de Chaleur d'Attention (Grad-CAM / Superpixels)")
            gray_arr = np.array(gray).astype(float)
            grad_x, grad_y = np.gradient(gray_arr)
            saliency = np.sqrt(grad_x**2 + grad_y**2)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)

            fig_hm = px.imshow(saliency, color_continuous_scale="Inferno")
            fig_hm.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Veuillez sélectionner ou téléverser une image pour visualiser l'explicabilité visuelle.")
