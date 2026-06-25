import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import cv2

st.set_page_config(
    page_title="Agricultura Inteligente",
    page_icon="🌱",
    layout="wide"
)

MODELOS_DIR = "modelos"
EFFICIENTNET_PATH = os.path.join(MODELOS_DIR, "plantvillage_efficientnet_final.keras")
CLASS_NAMES_PATH  = os.path.join(MODELOS_DIR, "class_names.json")
MINNEAPPLE_PATH   = os.path.join(MODELOS_DIR, "minneapple_yolo_best.pt")
FRUITS_VEG_PATH   = os.path.join(MODELOS_DIR, "yolo_fruits_and_vegetables_v3.pt")
WEED_PATH         = os.path.join(MODELOS_DIR, "weed_yolo_best.pt")

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

@st.cache_resource
def carregar_modelo_doencas():
    import tensorflow as tf
    modelo = tf.keras.models.load_model(EFFICIENTNET_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        classes = json.load(f)
    return modelo, classes

@st.cache_resource
def carregar_minneapple():
    from ultralytics import YOLO
    return YOLO(MINNEAPPLE_PATH)

@st.cache_resource
def carregar_fruits_veg():
    from ultralytics import YOLO
    return YOLO(FRUITS_VEG_PATH)

@st.cache_resource
def carregar_modelo_ervas():
    from ultralytics import YOLO
    return YOLO(WEED_PATH)

@st.cache_resource
def carregar_owlv2():
    from transformers import pipeline
    import torch
    return pipeline(
        model="google/owlv2-base-patch16-ensemble",
        task="zero-shot-object-detection",
        device=0 if torch.cuda.is_available() else -1,
    )

def calibrar_threshold_automatico(todas_detecoes, min_t=0.45, max_t=0.75):
    """
    Calibração automática melhorada para evitar falsos positivos.
    Aumentamos o min_t de 0.35 para 0.45.
    """
    if len(todas_detecoes) < 5:
        return 0.50
    scores = np.array([r['score'] for r in todas_detecoes], dtype=np.float64)
    hist, bin_edges = np.histogram(scores, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if hist.sum() == 0:
        return 0.50
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_total = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        numerador = (mu_total * omega - mu) ** 2
        denominador = omega * (1.0 - omega)
        variancia = np.where(denominador > 0, numerador / denominador, 0.0)
    k_otimo = int(np.argmax(variancia))
    threshold_otsu = float(bin_centers[k_otimo])
    # Aplicar limites mais rigorosos
    return round(max(min(threshold_otsu, max_t), min_t), 2)

def prever_doenca(imagem_pil):
    import tensorflow as tf
    modelo, classes = carregar_modelo_doencas()
    img = imagem_pil.resize(IMG_SIZE)
    arr = np.array(img)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    preds = modelo.predict(arr, verbose=0)[0]
    idx_top = np.argsort(preds)[::-1][:3]
    return [(classes[i], float(preds[i])) for i in idx_top]

def obter_todos_scores_yolo(imagem_pil, modelo_fn, imgsz=640):
    modelo = modelo_fn()
    res = modelo.predict(source=np.array(imagem_pil), conf=0.01, imgsz=imgsz, verbose=False)[0]
    scores = [float(b.conf) for b in res.boxes]
    return scores

def _iou_yolo(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def contar_com_threshold(imagem_pil, modelo_fn, conf, imgsz=640, fruto_filtro=None, iou_threshold=0.45):
    modelo = modelo_fn()
    img_np = np.array(imagem_pil)
    res = modelo.predict(source=img_np, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy()
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1: break
        ious = np.array([_iou_yolo(boxes[i], boxes[j]) for j in indices[1:]])
        filtered_indices = np.where(ious < iou_threshold)[0]
        indices = indices[filtered_indices + 1]
    final_classes = classes[keep]
    contagens = {}
    total = 0
    for cls_id in final_classes:
        nome = modelo.names[int(cls_id)]
        if fruto_filtro and fruto_filtro != "Todos":
            if nome == fruto_filtro: total += 1
        else: total += 1
        contagens[nome] = contagens.get(nome, 0) + 1
    img_anotada = res.plot()[:, :, ::-1]
    return total, img_anotada, contagens

def calcular_curva_threshold(scores, thresholds):
    contagens = []
    for t in thresholds:
        contagens.append(sum(1 for s in scores if s >= t))
    return contagens

def encontrar_joelho_curva(thresholds, contagens):
    if len(contagens) < 3: return thresholds[0]
    derivadas = np.diff(contagens)
    segunda_derivada = np.diff(derivadas)
    if len(segunda_derivada) == 0: return thresholds[0]
    idx_joelho = int(np.argmax(np.abs(segunda_derivada))) + 1
    return round(float(thresholds[idx_joelho]), 2)

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a['box']['xmin'],a['box']['ymin'],a['box']['xmax'],a['box']['ymax']
    bx1,by1,bx2,by2 = b['box']['xmin'],b['box']['ymin'],b['box']['xmax'],b['box']['ymax']
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0

def contar_owlv2(imagem_pil, tile_size=500, overlap=0.35):
    detector = carregar_owlv2()
    img_np = np.array(imagem_pil)
    H, W = img_np.shape[:2]
    # Queries mais específicas para evitar detetar folhas verdes
    queries = ["ripe red apple", "apple fruit", "round red apple"]
    step = int(tile_size * (1 - overlap))
    ys = list(range(0, max(1, H - tile_size + 1), step))
    xs = list(range(0, max(1, W - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < H: ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W: xs.append(max(0, W - tile_size))
    todas, total_tiles = [], len(ys) * len(xs)
    prog = st.progress(0, text="A analisar imagem em janelas...")
    for ti, y0 in enumerate(ys):
        for tj, x0 in enumerate(xs):
            tile = imagem_pil.crop((x0, y0, min(x0+tile_size,W), min(y0+tile_size,H)))
            for r in detector(tile, candidate_labels=queries):
                if r['score'] < 0.20: continue # Ignorar scores muito baixos logo no início
                box = r['box']
                todas.append({'score': r['score'], 'label': r['label'],
                    'box': {'xmin': box['xmin']+x0, 'ymin': box['ymin']+y0,
                            'xmax': box['xmax']+x0, 'ymax': box['ymax']+y0}})
        prog.progress(min((ti+1)/len(ys), 1.0), text=f"A processar... {(ti+1)*len(xs)}/{total_tiles} tiles")
    prog.empty()
    threshold_auto = calibrar_threshold_automatico(todas)
    kept = []
    for r in sorted([x for x in todas if x['score'] >= threshold_auto], key=lambda x: -x['score']):
        if all(_iou(r, k) < 0.60 for k in kept): # IOU mais rigoroso (0.60)
            kept.append(r)
    from PIL import ImageDraw
    out = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(out)
    for r in kept:
        b = r['box']
        x1,y1,x2,y2 = int(b['xmin']),int(b['ymin']),int(b['xmax']),int(b['ymax'])
        draw.rectangle([x1,y1,x2,y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(y1-12,0)), f"{r['score']:.2f}", fill=(0, 255, 0))
    return len(kept), np.array(out), threshold_auto

def detetar_ervas(imagem_pil, conf=0.4):
    modelo = carregar_modelo_ervas()
    res = modelo.predict(source=np.array(imagem_pil), conf=conf, imgsz=640, verbose=False)[0]
    img_anotada = res.plot()[:, :, ::-1]
    n_crop = sum(1 for c in res.boxes.cls if int(c) == 0)
    n_weed = sum(1 for c in res.boxes.cls if int(c) == 1)
    return n_crop, n_weed, img_anotada

# ---------- Interface ----------

st.title("🌱 Visão Computacional Aplicada à Agricultura Inteligente")
st.caption("Sistema de apoio à decisão com modelos de Deep Learning para deteção de doenças, contagem de frutos e deteção de ervas daninhas.")

st.sidebar.header("Configurações")
opcao = st.sidebar.radio("Funcionalidade:", [
    "🍃 Classificação de Doenças (Folhas)",
    "🍎 Contagem de Frutos",
    "🌿 Deteção de Ervas Daninhas",
])

fruto_filtro = "Todos"
tile_size = 500
conf_threshold = 0.35
pipeline_frutos = ""

if opcao.startswith("🍎"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline de deteção")
    pipeline_frutos = st.sidebar.radio("Método:", [
        "YOLOv8 MinneApple (maçãs, especializado)",
        "YOLOv8 Frutas & Vegetais (63 classes)",
        "OWLv2 Sliding Window (zero-shot)",
    ])
    
    if not pipeline_frutos.startswith("OWLv2"):
        iou_threshold = st.sidebar.slider("Filtro de Sobreposição (IOU)", 0.1, 0.9, 0.45, 0.05)
    
    if pipeline_frutos.startswith("YOLOv8 MinneApple"):
        conf_threshold = st.sidebar.slider("Confiança mínima", 0.05, 0.90, 0.45, 0.05)
        calibrar_auto = st.sidebar.checkbox("🎯 Calibrar threshold automaticamente", value=False)
    elif pipeline_frutos.startswith("YOLOv8 Frutas"):
        conf_threshold = st.sidebar.slider("Confiança mínima", 0.05, 0.90, 0.35, 0.05)
        calibrar_auto = st.sidebar.checkbox("🎯 Calibrar threshold automaticamente", value=True)
        frutos_disponiveis = ["Todos","apple","banana","papaya","pineapple","watermelon",
                              "coconut/cocoanut","orange/orange fruit","grape","tomato",
                              "avocado","lemon","lime","peach","pear","kiwi fruit","melon","cherry"]
        fruto_filtro = st.sidebar.selectbox("Filtrar por fruto:", frutos_disponiveis)
    else:
        tile_size = st.sidebar.select_slider("Tamanho do tile (px)", [300,400,500,640], value=500)
        st.sidebar.warning("⚠️ O OWLv2 é sensível a folhagem. Usamos queries específicas e threshold mínimo de 0.45 para maior precisão.")
        calibrar_auto = True

elif opcao.startswith("🌿"):
    conf_threshold = st.sidebar.slider("Confiança mínima", 0.1, 0.9, 0.4, 0.05)
    calibrar_auto = False

else:
    calibrar_auto = False

uploaded_file = st.file_uploader("Carregar imagem", type=["jpg","jpeg","png"])

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
            st.metric("Confiança", f"{conf_top*100:.1f}%")
            st.write("Outras possibilidades:")
            for classe, conf in resultados[1:]:
                st.write(f"- {traduzir_classe(classe)}: {conf*100:.1f}%")

    elif opcao.startswith("🍎"):
        if not pipeline_frutos.startswith("OWLv2"):
            modelo_fn = carregar_minneapple if pipeline_frutos.startswith("YOLOv8 MinneApple") else carregar_fruits_veg
            imgsz = 960 if pipeline_frutos.startswith("YOLOv8 MinneApple") else 640
            label = "MinneApple" if pipeline_frutos.startswith("YOLOv8 MinneApple") else "Frutas & Vegetais"

            if calibrar_auto:
                with st.spinner("A calibrar..."):
                    scores = obter_todos_scores_yolo(imagem, modelo_fn, imgsz=imgsz)
                if scores:
                    thresholds = np.arange(0.05, 0.91, 0.05)
                    contagens_curva = calcular_curva_threshold(scores, thresholds)
                    conf_threshold = encontrar_joelho_curva(thresholds, contagens_curva)
                    st.sidebar.info(f"🎯 Threshold sugerido: **{conf_threshold}**")

            with st.spinner(f"A detetar com YOLOv8..."):
                n, img_anotada, contagens = contar_com_threshold(
                    imagem, modelo_fn, conf_threshold, imgsz=imgsz, fruto_filtro=fruto_filtro, iou_threshold=iou_threshold)
            with col2:
                st.subheader(f"Resultado — {label}")
                st.metric("🍎 Objetos detetados", n)
                st.image(img_anotada, use_column_width=True)

        else:
            with st.spinner("A processar com OWLv2 (Sliding Window)..."):
                n, img_anotada, threshold_auto = contar_owlv2(imagem, tile_size=tile_size)
            with col2:
                st.subheader("Resultado — OWLv2 (zero-shot)")
                st.metric("🍎 Maçãs detetadas", n)
                st.image(img_anotada, use_column_width=True)
            st.sidebar.success(f"🎯 Threshold automático: **{threshold_auto}**")

    else:
        with st.spinner("A detetar plantas..."):
            n_crop, n_weed, img_anotada = detetar_ervas(imagem, conf=conf_threshold)
        with col2:
            st.subheader("Resultado")
            c1, c2 = st.columns(2)
            c1.metric("🌾 Cultura (crop)", n_crop)
            c2.metric("🌿 Erva daninha (weed)", n_weed)
            st.image(img_anotada, use_column_width=True)

else:
    st.info("Carregue uma imagem para começar.")

st.sidebar.markdown("---")
st.sidebar.caption("Projeto académico — Visão Computacional Aplicada à Agricultura Inteligente")
