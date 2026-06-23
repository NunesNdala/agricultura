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


# ---------- Carregamento de modelos ----------

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


@st.cache_resource
def carregar_owlv2():
    from transformers import pipeline
    import torch
    detector = pipeline(
        model="google/owlv2-base-patch16-ensemble",
        task="zero-shot-object-detection",
        device=0 if torch.cuda.is_available() else -1,
    )
    return detector


# ---------- Calibração automática de threshold (método Otsu) ----------

def calibrar_threshold_automatico(todas_detecoes, min_t=0.20, max_t=0.70):
    """
    Escolhe automaticamente o threshold ótimo baseado na distribuição
    dos scores de confiança usando o método de Otsu adaptado.
    Maximiza a variância inter-classe entre deteções fortes e fracas.
    """
    if len(todas_detecoes) < 5:
        return 0.45  # fallback se poucas deteções

    scores = np.array([r['score'] for r in todas_detecoes])
    melhor_threshold = 0.45
    melhor_variancia = -1

    for t in np.arange(min_t, max_t, 0.01):
        acima = scores[scores >= t]
        abaixo = scores[scores < t]
        if len(acima) < 2 or len(abaixo) < 2:
            continue
        w_acima = len(acima) / len(scores)
        w_abaixo = len(abaixo) / len(scores)
        variancia = w_acima * w_abaixo * (acima.mean() - abaixo.mean()) ** 2
        if variancia > melhor_variancia:
            melhor_variancia = variancia
            melhor_threshold = t

    return round(float(melhor_threshold), 2)


# ---------- Funções de inferência ----------

def prever_doenca(imagem_pil):
    import tensorflow as tf
    modelo, classes = carregar_modelo_doencas()
    img_resized = imagem_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    preds = modelo.predict(img_array, verbose=0)[0]
    idx_top = np.argsort(preds)[::-1][:3]
    return [(classes[i], float(preds[i])) for i in idx_top]


def contar_frutos_yolo(imagem_pil, conf=0.35):
    modelo = carregar_modelo_frutos()
    resultado = modelo.predict(source=np.array(imagem_pil), conf=conf, imgsz=960, verbose=False)[0]
    n_frutos = len(resultado.boxes)
    img_anotada = resultado.plot()[:, :, ::-1]
    return n_frutos, img_anotada


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a['box']['xmin'], a['box']['ymin'], a['box']['xmax'], a['box']['ymax']
    bx1, by1, bx2, by2 = b['box']['xmin'], b['box']['ymin'], b['box']['xmax'], b['box']['ymax']
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def contar_frutos_owlv2(imagem_pil, tile_size=500, overlap=0.35):
    detector = carregar_owlv2()
    img_np = np.array(imagem_pil)
    H, W = img_np.shape[:2]
    queries = ["apple", "red apple", "apple on ground"]

    step = int(tile_size * (1 - overlap))
    ys = list(range(0, max(1, H - tile_size + 1), step))
    xs = list(range(0, max(1, W - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(0, W - tile_size))

    # Fase 1: recolher TODAS as deteções com threshold mínimo
    todas = []
    total_tiles = len(ys) * len(xs)
    prog = st.progress(0, text="A analisar imagem em janelas...")

    for ti, y0 in enumerate(ys):
        for tj, x0 in enumerate(xs):
            y1_t = min(y0 + tile_size, H)
            x1_t = min(x0 + tile_size, W)
            tile = imagem_pil.crop((x0, y0, x1_t, y1_t))
            res = detector(tile, candidate_labels=queries)
            for r in res:
                if r['score'] < 0.10:  # só eliminar scores muito baixos
                    continue
                box = r['box']
                todas.append({
                    'score': r['score'],
                    'label': r['label'],
                    'box': {
                        'xmin': box['xmin'] + x0,
                        'ymin': box['ymin'] + y0,
                        'xmax': box['xmax'] + x0,
                        'ymax': box['ymax'] + y0,
                    }
                })
        prog.progress(min((ti + 1) / len(ys), 1.0),
                      text=f"A processar... {(ti+1)*len(xs)}/{total_tiles} tiles")

    prog.empty()

    # Fase 2: calibrar threshold automaticamente (Otsu)
    threshold_auto = calibrar_threshold_automatico(todas)

    # Fase 3: filtrar com threshold calibrado + NMS
    filtradas = [r for r in todas if r['score'] >= threshold_auto]
    sorted_det = sorted(filtradas, key=lambda x: -x['score'])
    kept = []
    for r in sorted_det:
        if all(_iou(r, k) < 0.55 for k in kept):
            kept.append(r)

    # Desenhar caixas com PIL
    from PIL import ImageDraw
    img_pil_out = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(img_pil_out)
    for r in kept:
        box = r['box']
        x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
        draw.text((x1, max(y1 - 12, 0)), f"{r['score']:.2f}", fill=(0, 200, 0))

    return len(kept), np.array(img_pil_out), threshold_auto


def detetar_ervas(imagem_pil, conf=0.4):
    modelo = carregar_modelo_ervas()
    resultado = modelo.predict(source=np.array(imagem_pil), conf=conf, imgsz=640, verbose=False)[0]
    img_anotada = resultado.plot()[:, :, ::-1]
    n_crop = sum(1 for c in resultado.boxes.cls if int(c) == 0)
    n_weed = sum(1 for c in resultado.boxes.cls if int(c) == 1)
    return n_crop, n_weed, img_anotada


# ---------- Interface ----------

st.title("🌱 Visão Computacional Aplicada à Agricultura Inteligente")
st.caption("Sistema de apoio à decisão com modelos de Deep Learning para deteção de doenças, contagem de frutos e deteção de ervas daninhas.")

st.sidebar.header("Escolha o modelo")
opcao = st.sidebar.radio(
    "Funcionalidade:",
    [
        "🍃 Classificação de Doenças (Folhas)",
        "🍎 Contagem de Frutos (Maçãs)",
        "🌿 Deteção de Ervas Daninhas",
    ]
)

if opcao.startswith("🍎"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline de deteção")
    pipeline_frutos = st.sidebar.radio(
        "Método:",
        ["YOLOv8 (rápido)", "OWLv2 Sliding Window (avançado)"],
        help="YOLOv8: treinado no MinneApple, rápido.\nOWLv2: zero-shot com calibração automática, mais lento."
    )
    if pipeline_frutos == "YOLOv8 (rápido)":
        conf_threshold = st.sidebar.slider("Confiança mínima (conf)", 0.1, 0.9, 0.35, 0.05)
    else:
        tile_size = st.sidebar.select_slider("Tamanho do tile (px)", [300, 400, 500, 640], value=500)
        st.sidebar.caption("⚠️ O modo avançado pode demorar 2–4 minutos. O threshold é calibrado automaticamente para cada imagem.")

elif opcao.startswith("🌿"):
    conf_threshold = st.sidebar.slider("Confiança mínima (conf)", 0.1, 0.9, 0.4, 0.05)

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
            st.success(f"**{traduzir_classe(classe_top)}**")
            st.metric("Confiança", f"{conf_top * 100:.1f}%")
            st.write("Outras possibilidades:")
            for classe, conf in resultados[1:]:
                st.write(f"- {traduzir_classe(classe)}: {conf * 100:.1f}%")
        st.caption("⚠️ Modelo EfficientNetB0 treinado no dataset PlantVillage (38 classes, 99% accuracy em teste).")

    elif opcao.startswith("🍎"):
        if pipeline_frutos == "YOLOv8 (rápido)":
            with st.spinner("A contar maçãs com YOLOv8..."):
                n_frutos, img_anotada = contar_frutos_yolo(imagem, conf=conf_threshold)
            with col2:
                st.subheader("Resultado — YOLOv8")
                st.metric("🍎 Maçãs detetadas", n_frutos)
                st.image(img_anotada, use_column_width=True)
            st.caption("⚠️ YOLOv8n treinado no MinneApple (mAP50: 0.865, erro agregado: 0.39%). Limitado a maçãs na árvore.")

        else:
            st.info("🔍 Modo avançado: a imagem será processada em tiles sobrepostos com OWLv2. O threshold será calibrado automaticamente para esta imagem. Aguarde...")
            with st.spinner("A processar com OWLv2 Sliding Window..."):
                n_frutos, img_anotada, threshold_auto = contar_frutos_owlv2(
                    imagem, tile_size=tile_size, overlap=0.35)
            with col2:
                st.subheader("Resultado — OWLv2 (zero-shot)")
                st.metric("🍎 Maçãs detetadas", n_frutos)
                st.image(img_anotada, use_column_width=True)

            st.sidebar.success(f"🎯 Threshold calibrado automaticamente: **{threshold_auto}**")

            with st.expander("📊 Comparar com YOLOv8 (mesma imagem)"):
                if st.button("▶ Correr comparação YOLOv8 vs OWLv2"):
                    with st.spinner("A correr YOLOv8 para comparação..."):
                        n_yolo, img_yolo = contar_frutos_yolo(imagem, conf=0.35)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("YOLOv8", n_yolo)
                        st.image(img_yolo, use_column_width=True)
                    with c2:
                        st.metric("OWLv2 Sliding Window", n_frutos, delta=n_frutos - n_yolo)
                        st.image(img_anotada, use_column_width=True)
                else:
                    st.caption("Clique no botão acima para correr o YOLOv8 e comparar lado a lado.")

            st.caption(f"⚠️ OWLv2 (google/owlv2-base-patch16-ensemble) — zero-shot com calibração Otsu automática. Threshold: {threshold_auto} | Tile: {tile_size}px | Overlap: 35%.")

    else:
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
