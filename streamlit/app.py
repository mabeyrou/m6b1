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


def main():
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
    predict_button = st.button(type="primary", label="Predict", disabled=is_empty)

    if canvas_result.image_data is not None and not is_empty:
        img_array = canvas_result.image_data.astype(np.uint8)
        img_pil = Image.fromarray(img_array)

        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()

        if predict_button:
            prediction = predict(img_base64)

            if prediction["success"]:
                st.write(f"The model thinks it's a {prediction['prediction']}")
            else:
                st.write("Something went wrong")


if __name__ == "__main__":
    st.set_page_config(page_title="test", page_icon=":pencil2:")
    st.title("Drawable Canvas Demo")
    main()
