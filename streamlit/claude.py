import streamlit as st
import requests
import base64
import io
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
CANVAS_SIZE = 280

st.set_page_config(page_title="Digit Recognition", page_icon="🔢", layout="wide")

# Initialisation du state
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def image_to_base64(img_array: np.ndarray) -> str:
    """Convertit un numpy array en base64"""
    if len(img_array.shape) == 3:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
    else:
        img_pil = Image.fromarray(img_array.astype(np.uint8))

    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return img_base64


def call_prediction_api(image_base64: str):
    """Appelle l'API de prédiction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", json={"image": image_base64}, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None


def submit_feedback(prediction_id: str, is_correct: bool, true_digit: int = None):
    """Soumet le feedback à l'API"""
    try:
        payload = {"prediction_id": prediction_id, "is_correct": is_correct}
        if true_digit is not None:
            payload["true_digit"] = true_digit

        response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Erreur lors de l'envoi du feedback: {str(e)}")
        return None


def get_stats():
    """Récupère les statistiques du modèle"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


# Interface utilisateur
st.title("🔢 Reconnaissance de Chiffres")
st.markdown("Dessinez un chiffre et l'IA essaiera de le reconnaître !")

# Sidebar avec statistiques
with st.sidebar:
    st.header("📊 Statistiques")
    stats = get_stats()
    if stats:
        st.metric("Total prédictions", stats["total_predictions"])
        st.metric("Avec feedback", stats["with_feedback"])
        st.metric("Précision", f"{stats['accuracy']:.1%}")
    else:
        st.info("Impossible de charger les statistiques")

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Dessinez un chiffre")

    # Canvas pour dessiner
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Transparent
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Boutons d'action
    col1a, col1b = st.columns(2)
    with col1a:
        predict_button = st.button(
            "🔮 Prédire", type="primary", use_container_width=True
        )
    with col1b:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.rerun()

with col2:
    st.subheader("🤖 Prédiction")

    # Zone de prédiction
    prediction_container = st.container()

    # Zone de feedback
    feedback_container = st.container()

# Logique de prédiction
if predict_button and canvas_result.image_data is not None:
    # Vérifier que l'image n'est pas vide
    img_array = canvas_result.image_data
    if np.any(img_array[:, :, 3] > 0):  # Vérifier le canal alpha
        with st.spinner("Prédiction en cours..."):
            # Conversion en base64
            img_base64 = image_to_base64(img_array)

            # Appel API
            prediction_result = call_prediction_api(img_base64)

            if prediction_result:
                st.session_state.current_prediction = prediction_result
                st.session_state.feedback_submitted = False

                # Ajouter à l'historique
                st.session_state.prediction_history.append(
                    {**prediction_result, "image": img_array}
                )

                st.rerun()
    else:
        st.warning("Veuillez dessiner quelque chose avant de prédire !")

# Affichage de la prédiction
if st.session_state.current_prediction:
    pred = st.session_state.current_prediction

    with prediction_container:
        st.success(f"**Prédiction: {pred['predicted_digit']}**")
        st.info(f"Confiance: {pred['confidence']:.1%}")
        st.caption(f"ID: {pred['prediction_id']}")

    # Interface de feedback
    if not st.session_state.feedback_submitted:
        with feedback_container:
            st.subheader("👍 Feedback")
            st.write("Cette prédiction est-elle correcte ?")

            col2a, col2b = st.columns(2)

            with col2a:
                if st.button("✅ Correct", use_container_width=True):
                    feedback_result = submit_feedback(pred["prediction_id"], True)
                    if feedback_result:
                        st.session_state.feedback_submitted = True
                        st.success("Merci pour votre feedback !")
                        st.rerun()

            with col2b:
                if st.button("❌ Incorrect", use_container_width=True):
                    st.session_state.show_correction = True
                    st.rerun()

            # Interface de correction
            if st.session_state.get("show_correction", False):
                st.write("**Quel était le bon chiffre ?**")

                # Boutons pour chaque chiffre
                digit_cols = st.columns(5)
                for i in range(10):
                    col_idx = i % 5
                    if i == 5:
                        digit_cols = st.columns(5)

                    with digit_cols[col_idx]:
                        if st.button(f"{i}", key=f"digit_{i}"):
                            feedback_result = submit_feedback(
                                pred["prediction_id"], False, i
                            )
                            if feedback_result:
                                st.session_state.feedback_submitted = True
                                st.session_state.show_correction = False
                                st.success(f"Merci ! Corrigé vers {i}")
                                st.rerun()
    else:
        with feedback_container:
            st.success("✅ Feedback envoyé, merci !")

# Historique des prédictions
if st.session_state.prediction_history:
    st.subheader("📝 Historique")

    # Afficher les 5 dernières prédictions
    recent_predictions = st.session_state.prediction_history[-5:]

    for i, pred in enumerate(reversed(recent_predictions)):
        with st.expander(
            f"Prédiction {len(recent_predictions) - i}: {pred['predicted_digit']} ({pred['confidence']:.1%})"
        ):
            col_hist1, col_hist2 = st.columns([1, 2])

            with col_hist1:
                st.image(pred["image"], caption="Image dessinée", width=100)

            with col_hist2:
                st.write(f"**Prédiction:** {pred['predicted_digit']}")
                st.write(f"**Confiance:** {pred['confidence']:.1%}")
                st.write(f"**ID:** {pred['prediction_id']}")
                st.write(f"**Timestamp:** {pred['timestamp']}")

# Instructions
st.markdown("---")
st.markdown("""
### 📋 Instructions
1. **Draw** a digit from 0 to 9 in the drawing area
2. **Click** "Predict" to get the model's prediction
3. **Give your feedback**: correct or incorrect
4. **If incorrect**, select the correct digit to improve the model

Your feedback helps improve the model's accuracy! 🚀
""")
