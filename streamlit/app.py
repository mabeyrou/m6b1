import streamlit as st
from streamlit_drawable_canvas import st_canvas
from loguru import logger
from PIL import Image
import numpy as np
import base64
import io

from api_client import predict


logger.remove()
logger.add(
    "./logs/dev_streamlit.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="TRACE",
    enqueue=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

DRAWING_MODES = ("freedraw", "point", "line", "rect", "circle", "transform")
CANVAS_DIMENSION = 96
EXPECTED_DIMENSION = 28


def main():
    drawing_mode = st.sidebar.selectbox("Drawing tool:", DRAWING_MODES)

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=CANVAS_DIMENSION,
        width=CANVAS_DIMENSION,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        key="canvas",
    )

    is_empty = not canvas_result.json_data["objects"]
    predict_button = st.button(type="primary", label="Predict", disabled=is_empty)

    if canvas_result.image_data is not None and not is_empty:
        img_array = canvas_result.image_data.astype(np.uint8)
        img_pil = Image.fromarray(img_array)
        resized_img = img_pil.resize((EXPECTED_DIMENSION, EXPECTED_DIMENSION))

        buffer = io.BytesIO()
        resized_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        if predict_button:
            prediction = predict(img_base64)
            logger.debug(prediction)

            if prediction["success"]:
                st.write(f"The model thinks it's a {prediction['prediction']}")
            else:
                st.write("Something went wrong")


if __name__ == "__main__":
    st.set_page_config(page_title="test", page_icon=":pencil2:")
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    main()
