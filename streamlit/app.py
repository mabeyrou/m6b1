import streamlit as st
from streamlit_drawable_canvas import st_canvas
from loguru import logger
from PIL import Image
import numpy as np
import base64
import io

from api_client import predict, feedback


logger.remove()
logger.add(
    "logs/local_streamlit.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="TRACE",
    enqueue=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

DRAWING_MODES = ("freedraw", "point", "line", "rect", "circle", "transform")
CANVAS_DIMENSION = 192


def image_to_base64(img_array: np.ndarray) -> str:
    img_pil = Image.fromarray(img_array)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return img_base64


def main():
    # Initialisation du state
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("✏️ Draw a digit")

        canvas_result = st_canvas(
            stroke_width=25,
            stroke_color="#000",
            background_color="#fff",
            height=CANVAS_DIMENSION,
            width=CANVAS_DIMENSION,
            drawing_mode="freedraw",
            key="canvas",
        )

        is_empty = not canvas_result.json_data["objects"]
        has_image_data = canvas_result.image_data is not None

        predict_button = st.button(
            label="🔮 Predict",
            type="primary",
            disabled=is_empty,
            use_container_width=True,
        )

    with col2:
        st.subheader("🤖 Prediction")
        prediction_container = st.container()
        feedback_container = st.container()

    if predict_button and has_image_data and not is_empty:
        img_array = canvas_result.image_data.astype(np.uint8)
        img_base64 = image_to_base64(img_array=img_array)

        with st.spinner("Predicting..."):
            prediction_result = predict(img_base64)

            if prediction_result:
                st.session_state.current_prediction = prediction_result
                st.session_state.feedback_submitted = False

                st.session_state.prediction_history.append(
                    {**prediction_result, "image": img_array}
                )
            else:
                st.write("Something went wrong")

    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction

        with prediction_container:
            st.success(f"**Prédiction: {pred['predicted_digit']}**")
            st.info(f"Confiance: {pred['confidence']:.1%}")
            st.caption(f"ID: {pred['digit_uuid']}")

        if not st.session_state.feedback_submitted:
            with feedback_container:
                st.subheader("👍 Feedback")
                st.write("Is this prediction correct?")

                col2a, col2b = st.columns(2)

                with col2a:
                    if st.button("✅ Correct", use_container_width=True):
                        feedback_result = feedback(
                            {
                                "digit_uuid": pred["digit_uuid"],
                                "is_correct": True,
                                "true_digit": pred["predicted_digit"],
                            }
                        )
                        if feedback_result:
                            st.session_state.feedback_submitted = True
                            st.success("Thank you for your feedback!")
                            st.rerun()

                with col2b:
                    if st.button("❌ Incorrect", use_container_width=True):
                        st.session_state.show_correction = True
                        st.rerun()

                # Interface de correction
                if st.session_state.get("show_correction", False):
                    st.write("**What was the correct digit?**")

                    # Boutons pour chaque chiffre
                    digit_cols = st.columns(5)
                    for i in range(10):
                        col_idx = i % 5
                        if i == 5:
                            digit_cols = st.columns(5)

                        with digit_cols[col_idx]:
                            if st.button(f"{i}", key=f"digit_{i}"):
                                logger.debug("before")
                                feedback_result = feedback(
                                    {
                                        "digit_uuid": pred["digit_uuid"],
                                        "is_correct": False,
                                        "true_digit": i,
                                    }
                                )
                                logger.debug(feedback_result)
                                if feedback_result:
                                    st.session_state.feedback_submitted = True
                                    st.session_state.show_correction = False
                                    st.success(f"Thank you! Corrected to {i}")
                                    st.rerun()
        else:
            with feedback_container:
                st.success("✅ Feedback sent, thanks !")

    # Historique des prédictions
    if st.session_state.prediction_history:
        st.subheader("📝 History")

        # Afficher les 5 dernières prédictions
        recent_predictions = st.session_state.prediction_history[-5:]

        for i, pred in enumerate(reversed(recent_predictions)):
            with st.expander(
                f"Prédiction {len(recent_predictions) - i}: {pred['predicted_digit']} ({pred['confidence']:.1%})"
            ):
                col_hist1, col_hist2 = st.columns([1, 2])

                with col_hist1:
                    st.image(pred["image"], caption="Drawn digit", width=100)

                with col_hist2:
                    st.write(f"**Prediction:** {pred['predicted_digit']}")
                    st.write(f"**Confidence:** {pred['confidence']:.1%}")
                    st.write(f"**UUID:** {pred['digit_uuid']}")
                    # st.write(f"**Timestamp:** {pred['timestamp']}")

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


if __name__ == "__main__":
    st.set_page_config(page_title="Digit Recognition", page_icon="🔢")
    st.title("🔢 Digits Recognition")
    st.markdown("Draw a digit and the AI will try to recognize it!")
    main()
