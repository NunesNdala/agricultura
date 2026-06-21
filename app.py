import streamlit as st
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="Agricultura Inteligente",
    page_icon="🌱",
    layout="wide"
)

MODELOS_DIR = "modelos"
EFFICIENTNET_PATH = os.path.join(MODELOS_DIR, "plantvillage_efficientnet_final.keras")
CLASS_NAMES_PATH = os.path.join(MODELOS_DIR, "class_names.json")
MINNEAPPLE_YOLO_PATH = os.path.join(MODELOS_DIR, "minneapple_yolo_best.pt")
WEED_YOLO_PATH = os.path.join(MODELOS_DIR, "weed_yolo_best.pt")

IMG_SIZE = (224, 224)

# Tradução das classes do PlantVillage (cultura — condição)
TRADUCAO_CLASSES = {
    "Apple___Apple_scab": "Macieira — Sarna da macieira",
    "Apple___Black_rot": "Macieira — Podridão negra",
    "Apple___Cedar_apple_rust": "Macieira — Ferrugem do cedro",
    "Apple___healthy": "Macieira — Saudável",
    "Blueberry___healthy": "Mirtilo — Saudável",
    "Cherry_(including_sour)___Powdery_mildew": "Cerejeira — Oídio",
    "Cherry_(including_sour)___healthy": "Cerejeira — Saudável",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Milho — Mancha-cinzenta da folha",
    "Corn_(maize)___Common_rust_": "Milho — Ferrugem comum",
    "Corn_(maize)___Northern_Leaf_Blight": "Milho — Helmintosporiose do norte",
    "Corn_(maize)___healthy": "Milho — Saudável",
    "Grape___Black_rot": "Videira — Podridão negra",
    "Grape___Esca_(Black_Measles)": "Videira — Esca (sarampo negro)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Videira — Mancha das folhas (Isariopsis)",
    "Grape___healthy": "Videira — Saudável",
    "Orange___Haunglongbing_(Citrus_greening)": "Laranjeira — Greening dos citrinos",
    "Peach___Bacterial_spot": "Pessegueiro — Mancha bacteriana",
    "Peach___healthy": "Pessegueiro — Saudável",
    "Pepper,_bell___Bacterial_spot": "Pimento — Mancha bacteriana",
    "Pepper,_bell___healthy": "Pimento — Saudável",
    "Potato___Early_blight": "Batateira — Pinta-preta",
    "Potato___Late_blight": "Batateira — Requeima",
    "Potato___healthy": "Batateira — Saudável",
    "Raspberry___healthy": "Framboesa — Saudável",
    "Soybean___healthy": "Soja — Saudável",
    "Squash___Powdery_mildew": "Abóbora — Oídio",
    "Strawberry___Leaf_scorch": "Morangueiro — Queima das folhas",
    "Strawberry___healthy": "Morangueiro — Saudável",
    "Tomato___Bacterial_spot": "Tomateiro — Mancha bacteriana",
    "Tomato___Early_blight": "Tomateiro — Pinta-preta",
    "Tomato___Late_blight": "Tomateiro — Requeima",
    "Tomato___Leaf_Mold": "Tomateiro — Bolor das folhas",
    "Tomato___Septoria_leaf_spot": "Tomateiro — Septoriose",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomateiro — Ácaro-rajado",
    "Tomato___Target_Spot": "Tomateiro — Mancha-alvo",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomateiro — Vírus do enrolamento amarelo da folha",
    "Tomato___Tomato_mosaic_virus": "Tomateiro — Vírus do mosaico",
    "Tomato___healthy": "Tomateiro — Saudável",
}


def traduzir_classe(nome_classe):
    return TRADUCAO_CLASSES.get(nome_classe, nome_classe.replace("___", " — ").replace("_", " "))


# ---------- Carregamento de modelos (cache para não recarregar a cada interação) ----------

@st.cache_resource
def carregar_modelo_doencas():
    import tensorflow as tf
    modelo = tf.keras.models.load_model(EFFICIENTNET_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    return modelo, classes


@st.cache_resource
def carregar_modelo_frutos():
    from ultralytics import YOLO
    return YOLO(MINNEAPPLE_YOLO_PATH)


@st.cache_resource
def carregar_modelo_ervas():
    from ultralytics import YOLO
    return YOLO(WEED_YOLO_PATH)


# ---------- Funções de inferência ----------

def prever_doenca(imagem_pil):
    import tensorflow as tf
    modelo, classes = carregar_modelo_doencas()

    img_resized = imagem_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    if img_array.shape[-1] == 4:  # remover canal alpha, se existir
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)

    preds = modelo.predict(img_array, verbose=0)[0]
    idx_top = np.argsort(preds)[::-1][:3]  # top-3

    resultados = [(classes[i], float(preds[i])) for i in idx_top]
    return resultados


def contar_frutos(imagem_pil, conf=0.35):
    modelo = carregar_modelo_frutos()
    resultado = modelo.predict(source=np.array(imagem_pil), conf=conf, imgsz=960, verbose=False)[0]
    n_frutos = len(resultado.boxes)
    img_anotada = resultado.plot()  # BGR
    img_anotada_rgb = img_anotada[:, :, ::-1]  # BGR -> RGB
    return n_frutos, img_anotada_rgb


def detetar_ervas(imagem_pil, conf=0.4):
    modelo = carregar_modelo_ervas()
    resultado = modelo.predict(source=np.array(imagem_pil), conf=conf, imgsz=640, verbose=False)[0]
    img_anotada = resultado.plot()
    img_anotada_rgb = img_anotada[:, :, ::-1]

    n_crop = sum(1 for c in resultado.boxes.cls if int(c) == 0)
    n_weed = sum(1 for c in resultado.boxes.cls if int(c) == 1)
    return n_crop, n_weed, img_anotada_rgb


# ---------- Interface ----------

st.title("🌱 Visão Computacional Aplicada à Agricultura Inteligente")
st.caption("Sistema de apoio à decisão com 3 modelos de Deep Learning treinados para deteção de doenças, contagem de frutos e deteção de ervas daninhas.")

st.sidebar.header("Escolha o modelo")
opcao = st.sidebar.radio(
    "Funcionalidade:",
    [
        "🍃 Classificação de Doenças (Folhas)",
        "🍎 Contagem de Frutos (Maçãs)",
        "🌿 Deteção de Ervas Daninhas",
    ]
)

uploaded_file = st.file_uploader("Carregar imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    imagem = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagem original")
        st.image(imagem, use_column_width=True)

    if opcao.startswith("🍃"):
        with st.spinner("A classificar a doença..."):
            resultados = prever_doenca(imagem)

        with col2:
            st.subheader("Resultado")
            classe_top, conf_top = resultados[0]
            nome_legivel = traduzir_classe(classe_top)
            st.success(f"**{nome_legivel}**")
            st.metric("Confiança", f"{conf_top * 100:.1f}%")

            st.write("Outras possibilidades:")
            for classe, conf in resultados[1:]:
                nome = traduzir_classe(classe)
                st.write(f"- {nome}: {conf * 100:.1f}%")

        st.caption("⚠️ Modelo EfficientNetB0 treinado no dataset PlantVillage (38 classes, 99% accuracy em teste). Use como apoio à decisão, não como diagnóstico definitivo.")

    elif opcao.startswith("🍎"):
        conf_threshold = st.sidebar.slider("Confiança mínima (conf)", 0.1, 0.9, 0.35, 0.05)
        with st.spinner("A contar maçãs..."):
            n_frutos, img_anotada = contar_frutos(imagem, conf=conf_threshold)

        with col2:
            st.subheader("Resultado")
            st.metric("Maçãs detetadas", n_frutos)
            st.image(img_anotada, use_column_width=True)

        st.caption("⚠️ Modelo YOLOv8n treinado no dataset MinneApple (mAP50: 0.865). Erro de contagem agregado medido em validação: 0.39% com conf=0.35.")

    else:
        conf_threshold = st.sidebar.slider("Confiança mínima (conf)", 0.1, 0.9, 0.4, 0.05)
        with st.spinner("A detetar plantas..."):
            n_crop, n_weed, img_anotada = detetar_ervas(imagem, conf=conf_threshold)

        with col2:
            st.subheader("Resultado")
            c1, c2 = st.columns(2)
            c1.metric("🌾 Cultura (crop)", n_crop)
            c2.metric("🌿 Erva daninha (weed)", n_weed)
            st.image(img_anotada, use_column_width=True)

        st.caption("⚠️ Modelo YOLOv8n treinado em dataset de sésamo + ervas daninhas (mAP50: 0.826).")

else:
    st.info("Carregue uma imagem para começar.")

st.sidebar.markdown("---")
st.sidebar.caption("Projeto académico — Visão Computacional Aplicada à Agricultura Inteligente")
