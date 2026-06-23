import hashlib
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(
    page_title="Agricultura Inteligente",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [BASE_DIR / "modelos", BASE_DIR]


def model_path(filename):
    for directory in MODEL_DIRS:
        path = directory / filename
        if path.exists():
            return str(path)
    return str(MODEL_DIRS[0] / filename)


EFFICIENTNET_PATH = model_path("plantvillage_efficientnet_final.keras")
CLASS_NAMES_PATH = model_path("class_names.json")
MINNEAPPLE_YOLO_PATH = model_path("minneapple_yolo_best.pt")
WEED_YOLO_PATH = model_path("weed_yolo_best.pt")
IMG_SIZE = (224, 224)

TRANSLATION = {
    "Apple___Apple_scab": "Macieira - Sarna da macieira",
    "Apple___Black_rot": "Macieira - Podridao negra",
    "Apple___Cedar_apple_rust": "Macieira - Ferrugem do cedro",
    "Apple___healthy": "Macieira - Saudavel",
    "Blueberry___healthy": "Mirtilo - Saudavel",
    "Cherry_(including_sour)___Powdery_mildew": "Cerejeira - Oidio",
    "Cherry_(including_sour)___healthy": "Cerejeira - Saudavel",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Milho - Mancha-cinzenta da folha",
    "Corn_(maize)___Common_rust_": "Milho - Ferrugem comum",
    "Corn_(maize)___Northern_Leaf_Blight": "Milho - Helmintosporiose do norte",
    "Corn_(maize)___healthy": "Milho - Saudavel",
    "Grape___Black_rot": "Videira - Podridao negra",
    "Grape___Esca_(Black_Measles)": "Videira - Esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Videira - Mancha das folhas",
    "Grape___healthy": "Videira - Saudavel",
    "Orange___Haunglongbing_(Citrus_greening)": "Laranjeira - Greening dos citrinos",
    "Peach___Bacterial_spot": "Pessegueiro - Mancha bacteriana",
    "Peach___healthy": "Pessegueiro - Saudavel",
    "Pepper,_bell___Bacterial_spot": "Pimento - Mancha bacteriana",
    "Pepper,_bell___healthy": "Pimento - Saudavel",
    "Potato___Early_blight": "Batateira - Pinta-preta",
    "Potato___Late_blight": "Batateira - Requeima",
    "Potato___healthy": "Batateira - Saudavel",
    "Raspberry___healthy": "Framboesa - Saudavel",
    "Soybean___healthy": "Soja - Saudavel",
    "Squash___Powdery_mildew": "Abobora - Oidio",
    "Strawberry___Leaf_scorch": "Morangueiro - Queima das folhas",
    "Strawberry___healthy": "Morangueiro - Saudavel",
    "Tomato___Bacterial_spot": "Tomateiro - Mancha bacteriana",
    "Tomato___Early_blight": "Tomateiro - Pinta-preta",
    "Tomato___Late_blight": "Tomateiro - Requeima",
    "Tomato___Leaf_Mold": "Tomateiro - Bolor das folhas",
    "Tomato___Septoria_leaf_spot": "Tomateiro - Septoriose",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomateiro - Acaro-rajado",
    "Tomato___Target_Spot": "Tomateiro - Mancha-alvo",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomateiro - Virus do enrolamento amarelo",
    "Tomato___Tomato_mosaic_virus": "Tomateiro - Virus do mosaico",
    "Tomato___healthy": "Tomateiro - Saudavel",
}


def translate_class(class_name):
    return TRANSLATION.get(class_name, class_name.replace("___", " - ").replace("_", " "))


def image_hash(uploaded_file):
    uploaded_file.seek(0)
    digest = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    uploaded_file.seek(0)
    return digest


@st.cache_resource
def load_disease_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(EFFICIENTNET_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
        classes = json.load(file)
    return model, classes


@st.cache_resource
def load_fruit_model():
    from ultralytics import YOLO

    return YOLO(MINNEAPPLE_YOLO_PATH)


@st.cache_resource
def load_weed_model():
    from ultralytics import YOLO

    return YOLO(WEED_YOLO_PATH)


@st.cache_resource
def load_owlv2():
    import torch
    from transformers import pipeline

    return pipeline(
        model="google/owlv2-base-patch16-ensemble",
        task="zero-shot-object-detection",
        device=0 if torch.cuda.is_available() else -1,
    )


def predict_disease(image):
    model, classes = load_disease_model()
    resized = image.resize(IMG_SIZE)
    array = np.array(resized)
    if array.shape[-1] == 4:
        array = array[..., :3]
    array = np.expand_dims(array, axis=0)
    predictions = model.predict(array, verbose=0)[0]
    top = np.argsort(predictions)[::-1][:3]
    return [(classes[i], float(predictions[i])) for i in top]


def count_fruits_yolo(image, conf=0.35):
    model = load_fruit_model()
    result = model.predict(source=np.array(image), conf=conf, imgsz=960, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]
    return len(result.boxes), annotated


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["box"]["xmin"], a["box"]["ymin"], a["box"]["xmax"], a["box"]["ymax"]
    bx1, by1, bx2, by2 = b["box"]["xmin"], b["box"]["ymin"], b["box"]["xmax"], b["box"]["ymax"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0


def calibrate_threshold_otsu(detections, min_threshold=0.20, max_threshold=0.70):
    if len(detections) < 5:
        return 0.45

    scores = np.array([item["score"] for item in detections])
    best_threshold = 0.45
    best_variance = -1

    for threshold in np.arange(min_threshold, max_threshold, 0.01):
        above = scores[scores >= threshold]
        below = scores[scores < threshold]

        if len(above) < 2 or len(below) < 2:
            continue

        weight_above = len(above) / len(scores)
        weight_below = len(below) / len(scores)
        variance = weight_above * weight_below * (above.mean() - below.mean()) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return round(float(best_threshold), 2)


def count_fruits_owlv2(image, tile_size=500, overlap=0.35):
    detector = load_owlv2()
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    queries = ["apple", "red apple", "ripe apple", "apple fruit"]

    step = max(1, int(tile_size * (1 - overlap)))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    xs = list(range(0, max(1, width - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))

    all_detections = []
    total_tiles = len(ys) * len(xs)
    done = 0
    progress = st.progress(0, text="A analisar imagem em janelas...")

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = image.crop((x0, y0, x1, y1))
            results = detector(tile, candidate_labels=queries)

            for result in results:
                if result["score"] < 0.10:
                    continue
                box = result["box"]
                all_detections.append({
                    "score": float(result["score"]),
                    "label": result["label"],
                    "box": {
                        "xmin": box["xmin"] + x0,
                        "ymin": box["ymin"] + y0,
                        "xmax": box["xmax"] + x0,
                        "ymax": box["ymax"] + y0,
                    },
                })

            done += 1
            progress.progress(done / total_tiles, text=f"A processar... {done}/{total_tiles} tiles")

    progress.empty()

    threshold = calibrate_threshold_otsu(all_detections)
    detections = [item for item in all_detections if item["score"] >= threshold]

    kept = []
    for detection in sorted(detections, key=lambda item: -item["score"]):
        if all(iou(detection, previous) < 0.55 for previous in kept):
            kept.append(detection)

    output = Image.fromarray(image_array.copy())
    draw = ImageDraw.Draw(output)
    for detection in kept:
        box = detection["box"]
        x1, y1 = int(box["xmin"]), int(box["ymin"])
        x2, y2 = int(box["xmax"]), int(box["ymax"])
        draw.rectangle([x1, y1, x2, y2], outline=(0, 210, 0), width=2)
        draw.text((x1, max(y1 - 14, 0)), f"{detection['score']:.2f}", fill=(0, 210, 0))

    return len(kept), np.array(output), len(all_detections), threshold


def detect_weeds(image, conf=0.4):
    model = load_weed_model()
    result = model.predict(source=np.array(image), conf=conf, imgsz=640, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]
    n_crop = sum(1 for klass in result.boxes.cls if int(klass) == 0)
    n_weed = sum(1 for klass in result.boxes.cls if int(klass) == 1)
    return n_crop, n_weed, annotated


def run_analysis(option, image, params):
    if option == "disease":
        return {"kind": option, "results": predict_disease(image)}

    if option == "fruit_yolo":
        count, annotated = count_fruits_yolo(image, conf=params["conf"])
        return {"kind": option, "count": count, "image": annotated}

    if option == "fruit_owlv2":
        count, annotated, raw_count, threshold = count_fruits_owlv2(
            image,
            tile_size=params["tile_size"],
            overlap=0.35,
        )
        return {
            "kind": option,
            "count": count,
            "raw_count": raw_count,
            "image": annotated,
            "threshold": threshold,
            "tile_size": params["tile_size"],
        }

    crop, weed, annotated = detect_weeds(image, conf=params["conf"])
    return {"kind": option, "crop": crop, "weed": weed, "image": annotated}


def show_result(result):
    if result["kind"] == "disease":
        st.subheader("Resultado")
        best_class, best_conf = result["results"][0]
        st.success(f"**{translate_class(best_class)}**")
        st.metric("Confianca", f"{best_conf * 100:.1f}%")
        st.write("Outras possibilidades:")
        for class_name, conf in result["results"][1:]:
            st.write(f"- {translate_class(class_name)}: {conf * 100:.1f}%")
        return

    if result["kind"] == "fruit_yolo":
        st.subheader("Resultado - YOLOv8")
        st.metric("Macas detetadas", result["count"])
        st.image(result["image"], use_container_width=True)
        return

    if result["kind"] == "fruit_owlv2":
        st.subheader("Resultado - OWLv2 (zero-shot)")
        st.metric("Macas detetadas", result["count"])
        st.caption(f"Threshold calibrado automaticamente: {result['threshold']:.2f}")
        st.caption(f"Detecoes recolhidas antes do filtro/NMS: {result['raw_count']}")
        st.image(result["image"], use_container_width=True)
        st.caption(
            f"OWLv2 com calibracao Otsu automatica | Tile: {result['tile_size']}px | Overlap: 35%"
        )
        return

    st.subheader("Resultado")
    c1, c2 = st.columns(2)
    c1.metric("Cultura (crop)", result["crop"])
    c2.metric("Erva daninha (weed)", result["weed"])
    st.image(result["image"], use_container_width=True)


st.title("Visao Computacional Aplicada a Agricultura Inteligente")
st.caption(
    "Sistema de apoio a decisao com modelos de Deep Learning para doencas em folhas, "
    "contagem de frutos e detecao de ervas daninhas."
)

st.sidebar.header("Escolha o modelo")
feature = st.sidebar.radio(
    "Funcionalidade:",
    [
        "Classificacao de Doencas (Folhas)",
        "Contagem de Frutos (Macas)",
        "Detecao de Ervas Daninhas",
    ],
)

params = {}
option = "disease"

if feature.startswith("Contagem"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline de detecao")
    pipeline = st.sidebar.radio(
        "Metodo:",
        ["YOLOv8 (rapido)", "OWLv2 Sliding Window (avancado)"],
        help="YOLOv8 e rapido e treinado no MinneApple. OWLv2 e zero-shot e mais lento.",
    )
    if pipeline.startswith("YOLO"):
        option = "fruit_yolo"
        params["conf"] = st.sidebar.slider("Confianca minima (conf)", 0.1, 0.9, 0.35, 0.05)
    else:
        option = "fruit_owlv2"
        params["tile_size"] = st.sidebar.select_slider("Tamanho do tile (px)", [300, 400, 500, 640], value=500)
        st.sidebar.caption(
            "O modo avancado pode demorar alguns minutos. "
            "O threshold e calibrado automaticamente para cada imagem."
        )
elif feature.startswith("Detecao"):
    option = "weed"
    params["conf"] = st.sidebar.slider("Confianca minima (conf)", 0.1, 0.9, 0.4, 0.05)

uploaded_file = st.file_uploader("Carregar imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Carregue uma imagem para comecar.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    current_key = {
        "file": image_hash(uploaded_file),
        "option": option,
        "params": params,
    }

    run = st.button("Analisar imagem", type="primary", use_container_width=True)

    if run:
        st.session_state["last_key"] = current_key
        st.session_state["last_result"] = None
        with st.spinner("A processar imagem..."):
            st.session_state["last_result"] = run_analysis(option, image, params)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem original")
        st.image(image, use_container_width=True)

    with col2:
        if st.session_state.get("last_key") == current_key and st.session_state.get("last_result"):
            show_result(st.session_state["last_result"])
        else:
            st.info("Clique em Analisar imagem para gerar um novo resultado com os parametros atuais.")

    if option == "fruit_owlv2" and st.session_state.get("last_key") == current_key:
        result = st.session_state.get("last_result")
        if result and result["kind"] == "fruit_owlv2":
            with st.expander("Comparar com YOLOv8 (mesma imagem)"):
                if st.button("Correr comparacao YOLOv8 vs OWLv2"):
                    with st.spinner("A correr YOLOv8 para comparacao..."):
                        yolo_count, yolo_image = count_fruits_yolo(image, conf=0.35)
                    c1, c2 = st.columns(2)
                    c1.metric("YOLOv8", yolo_count)
                    c1.image(yolo_image, use_container_width=True)
                    c2.metric("OWLv2 Sliding Window", result["count"], delta=result["count"] - yolo_count)
                    c2.image(result["image"], use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Projeto academico - Visao Computacional Aplicada a Agricultura Inteligente")
