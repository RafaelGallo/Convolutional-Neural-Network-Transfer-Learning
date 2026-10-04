"""Streamlit app to serve the Fashion-MNIST CNN / Transfer Learning models."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

MODEL_FILES = {
    "CNN Baseline": "cnn_baseline_fashion_mnist.keras",
    "MobileNetV2 (Transfer Learning)": "mobilenetv2_fashion_mnist.keras",
    "VGG16 (Transfer Learning)": "vgg16_fashion_mnist.keras",
    "VGG19 (Transfer Learning)": "vgg19_fashion_mnist.keras",
    "ResNet50 (Transfer Learning)": "resnet50_fashion_mnist.keras",
    "EfficientNetB0 (Transfer Learning)": "efficientnetb0_fashion_mnist.keras",
}


@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    return keras.models.load_model(MODELS_DIR / filename)


@st.cache_data(show_spinner=False)
def load_metrics():
    path = MODELS_DIR / "results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def preprocess_image(image: Image.Image, invert: bool) -> np.ndarray:
    """Convert an uploaded image to the (1, 28, 28, 1) float32 [0, 1] tensor
    expected by the saved models (same format used for Fashion-MNIST)."""
    gray = ImageOps.grayscale(image)
    gray = gray.resize((28, 28), Image.LANCZOS)
    if invert:
        gray = ImageOps.invert(gray)
    array = np.asarray(gray, dtype="float32") / 255.0
    return array.reshape(1, 28, 28, 1)


def main():
    st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👕", layout="centered")
    st.title("👕 Fashion-MNIST — CNN & Transfer Learning")
    st.write(
        "Envie uma foto de uma peça de roupa/calçado (fundo simples, "
        "de preferência centralizada) e escolha um modelo treinado para classificá-la."
    )

    metrics = load_metrics()

    st.sidebar.header("Modelo")
    model_label = st.sidebar.selectbox("Escolha o modelo", list(MODEL_FILES.keys()))
    model_key = model_label.replace(" (Transfer Learning)", "")
    if model_key in metrics:
        m = metrics[model_key]
        st.sidebar.metric("Acurácia (teste)", f"{m['accuracy']:.2%}")
        st.sidebar.metric("Macro F1", f"{m['macro_f1']:.4f}")
        st.sidebar.caption(f"Parâmetros: {m['params']:,}")

    invert = st.sidebar.checkbox(
        "Inverter cores",
        value=False,
        help="Marque se sua foto tem a peça escura sobre fundo claro "
        "(o dataset original tem peças claras sobre fundo escuro).",
    )

    uploaded_file = st.file_uploader("Imagem da peça", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Faça upload de uma imagem para ver a previsão.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(image, caption="Imagem enviada", use_column_width=True)

    x = preprocess_image(image, invert)
    col2.image(x.reshape(28, 28), caption="Entrada pré-processada (28x28)", use_column_width=True, clamp=True)

    with st.spinner(f"Carregando {model_label} e classificando..."):
        model = load_model(MODEL_FILES[model_label])
        probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    st.success(f"Previsão: **{CLASS_NAMES[pred_idx]}** ({probs[pred_idx]:.1%} de confiança)")

    st.subheader("Probabilidade por classe")
    st.bar_chart({"probabilidade": dict(zip(CLASS_NAMES, probs.tolist()))})


if __name__ == "__main__":
    main()
