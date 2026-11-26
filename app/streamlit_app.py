
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("Brain Tumor Classification")
st.caption("MRI Image Classification using Fine-tuned ResNet50")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("Analyze Image", type="primary"):

        with st.spinner("Analyzing image..."):

            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

                response.raise_for_status()
                result = response.json()

                st.divider()
                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Class",
                        result["prediction"]
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f'{result["confidence"] * 100:.2f}%'
                    )

                st.subheader("Class Probabilities")

                for class_name, probability in result["probabilities"].items():

                    st.write(
                        f"{class_name}: {probability * 100:.2f}%"
                    )

                    st.progress(probability)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Check that the API is running.")

            except requests.exceptions.RequestException as error:
                st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Research and educational purposes only. "
    "Not intended for clinical diagnosis."
)
