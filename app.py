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
FRUITS_VEG_PATH   = os.path.join(MODELOS_DIR, "yolo_fruits_and_vegetables_v3.pt")
WEED_PATH         = os.path.join(MODELOS_DIR, "weed_yolo_best.pt")
INCEPTION_PATH    = os.path.join(MODELOS_DIR, "inception.keras")
RESNET_PATH       = os.path.join(MODELOS_DIR, "resnet.keras")

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
def carregar_fruits_veg():
    from ultralytics import YOLO
    return YOLO(FRUITS_VEG_PATH)

@st.cache_resource
def carregar_modelo_ervas():
    from ultralytics import YOLO
    return YOLO(WEED_PATH)

@st.cache_resource
def carregar_inception():
    import tensorflow as tf
    return tf.keras.models.load_model(INCEPTION_PATH, compile=False)

@st.cache_resource
def carregar_resnet():
    import tensorflow as tf
    return tf.keras.models.load_model(RESNET_PATH, compile=False)

@st.cache_resource
def carregar_owlv2():
    from transformers import pipeline
    import torch
    return pipeline(
        model="google/owlv2-base-patch16-ensemble",
        task="zero-shot-object-detection",
        device=0 if torch.cuda.is_available() else -1,
    )

def prever_doenca(imagem_pil):
    import tensorflow as tf
    modelo, classes = carregar_modelo_doencas()
    img = imagem_pil.resize(IMG_SIZE)
    arr = np.array(img)
    if arr.shape[-1] == 4: arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    preds = modelo.predict(arr, verbose=0)[0]
    idx_top = np.argsort(preds)[::-1][:3]
    return [(classes[i], float(preds[i])) for i in idx_top]

def detetar_frutas_veg(imagem_pil, conf=0.25):
    modelo = carregar_fruits_veg()
    img_np = np.array(imagem_pil)
    res = modelo.predict(source=img_np, conf=conf, imgsz=640, verbose=False)[0]
    contagens = {}
    for box in res.boxes:
        cls_id = int(box.cls)
        nome = modelo.names[cls_id]
        contagens[nome] = contagens.get(nome, 0) + 1
    return res.plot()[:, :, ::-1], contagens

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a['box']['xmin'],a['box']['ymin'],a['box']['xmax'],a['box']['ymax']
    bx1,by1,bx2,by2 = b['box']['xmin'],b['box']['ymin'],b['box']['xmax'],b['box']['ymax']
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0

def contar_owlv2(imagem_pil, tile_size=500, overlap=0.30, min_conf=0.25):
    detector = carregar_owlv2()
    img_np = np.array(imagem_pil)
    H, W = img_np.shape[:2]
    queries = ["ripe red apple", "green leaf", "tree branch"]
    step = int(tile_size * (1 - overlap))
    ys = list(range(0, max(1, H - tile_size + 1), step))
    xs = list(range(0, max(1, W - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < H: ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W: xs.append(max(0, W - tile_size))
    todas = []
    prog = st.progress(0, text="Análise multiescala em curso...")
    for ti, y0 in enumerate(ys):
        for tj, x0 in enumerate(xs):
            tile = imagem_pil.crop((x0, y0, min(x0+tile_size,W), min(y0+tile_size,H)))
            for r in detector(tile, candidate_labels=queries):
                if r['label'] != "ripe red apple" or r['score'] < min_conf:
                    continue
                box = r['box']
                if (box['xmax'] - box['xmin']) < 15 or (box['ymax'] - box['ymin']) < 15:
                    continue
                todas.append({'score': r['score'], 'label': r['label'],
                    'box': {'xmin': box['xmin']+x0, 'ymin': box['ymin']+y0,
                            'xmax': box['xmax']+x0, 'ymax': box['ymax']+y0}})
        prog.progress(min((ti+1)/len(ys), 1.0))
    prog.empty()
    kept = []
    for r in sorted(todas, key=lambda x: -x['score']):
        if all(_iou(r, k) < 0.30 for k in kept):
            kept.append(r)
    from PIL import ImageDraw
    out = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(out)
    for r in kept:
        b = r['box']
        x1,y1,x2,y2 = int(b['xmin']),int(b['ymin']),int(b['xmax']),int(b['ymax'])
        draw.rectangle([x1,y1,x2,y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(y1-15,0)), f"{r['score']:.2f}", fill=(0, 255, 0))
    return len(kept), np.array(out)

def detetar_ervas(imagem_pil, conf=0.4):
    modelo = carregar_modelo_ervas()
    res = modelo.predict(source=np.array(imagem_pil), conf=conf, imgsz=640, verbose=False)[0]
    n_crop = sum(1 for c in res.boxes.cls if int(c) == 0)
    n_weed = sum(1 for c in res.boxes.cls if int(c) == 1)
    return n_crop, n_weed, res.plot()[:, :, ::-1]

def prever_cnn(imagem_pil, modelo):
    img = imagem_pil.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(modelo.predict(arr, verbose=0)[0][0])
    # sigmoid binário: prob > 0.5 → weed, senão → crop
    if prob > 0.5:
        return "🌿 Erva Daninha", prob
    else:
        return "🌾 Cultura", 1.0 - prob

# ---------- Interface ----------

st.title("🌱 Visão Computacional Aplicada à Agricultura Inteligente")
st.caption("Sistema de apoio à decisão para contagem de precisão e diagnóstico fitossanitário.")

st.sidebar.header("Módulos de Análise")
opcao = st.sidebar.radio("Selecione a tarefa:", [
    "🍎 Contagem de Precisão (Maçãs)",
    "🍃 Diagnóstico de Doenças",
    "🌿 Deteção de Ervas Daninhas",
    "🔍 Deteção Geral (Multiclasse)"
])

if opcao == "🍎 Contagem de Precisão (Maçãs)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ajuste de Sensibilidade")
    min_conf = st.sidebar.slider("Limiar de Confiança", 0.10, 0.80, 0.35, 0.05)
    tile_size = st.sidebar.select_slider("Janela de Análise (px)", [50, 100, 200, 300, 400, 500, 640], value=500)
    st.sidebar.success("✅ Algoritmo de contraste ativo.")

elif opcao == "🔍 Deteção Geral (Multiclasse)":
    conf_threshold = st.sidebar.slider("Confiança YOLO", 0.10, 0.90, 0.25, 0.05)

elif opcao == "🌿 Deteção de Ervas Daninhas":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuração")
    conf_weed = st.sidebar.slider("Confiança YOLO", 0.10, 0.90, 0.40, 0.05)
    usar_cnn = st.sidebar.checkbox("Confirmação por CNN (InceptionV3 + ResNet50)", value=True)

uploaded_file = st.file_uploader("Carregar Imagem de Campo", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    imagem = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entrada")
        st.image(imagem, use_column_width=True)

    # 1. Contagem de Maçãs (OWLv2)
    if opcao == "🍎 Contagem de Precisão (Maçãs)":
        with st.spinner("A processar filtragem de folhagem e contagem..."):
            n, img_anotada = contar_owlv2(imagem, tile_size=tile_size, min_conf=min_conf)
        with col2:
            st.subheader("Resultado da Contagem")
            st.metric("🍎 Maçãs Confirmadas", n)
            st.image(img_anotada, use_column_width=True)

    # 2. Doenças
    elif opcao == "🍃 Diagnóstico de Doenças":
        with st.spinner("A analisar tecidos vegetais..."):
            resultados = prever_doenca(imagem)
        with col2:
            st.subheader("Diagnóstico")
            classe_top, conf_top = resultados[0]
            st.success(f"**{traduzir_classe(classe_top)}**")
            st.metric("Confiança", f"{conf_top*100:.1f}%")

    # 3. Deteção Geral
    elif opcao == "🔍 Deteção Geral (Multiclasse)":
        with st.spinner("A detetar objetos e organizar dados..."):
            img_anotada, contagens = detetar_frutas_veg(imagem, conf=conf_threshold)
        with col2:
            st.subheader("Resultados da Deteção")
            st.image(img_anotada, use_column_width=True)
            if contagens:
                import pandas as pd
                st.write("**Dados Quantitativos:**")
                df = pd.DataFrame(list(contagens.items()), columns=['Classe', 'Quantidade'])
                st.table(df)
            else:
                st.warning("Nenhum objeto detetado com o threshold atual.")

    # 4. Ervas Daninhas
    else:
        with st.spinner("A mapear infestantes..."):
            n_crop, n_weed, img_anotada = detetar_ervas(imagem, conf=conf_weed)
        with col2:
            st.subheader("Métricas de Campo — YOLOv8")
            c1, c2 = st.columns(2)
            c1.metric("🌾 Cultura", n_crop)
            c2.metric("🌿 Ervas Daninhas", n_weed)
            st.image(img_anotada, use_column_width=True)

        # Confirmação CNN
        if usar_cnn:
            st.markdown("---")
            st.subheader("Confirmação por Classificação CNN")
            col_inc, col_res = st.columns(2)

            with st.spinner("A executar InceptionV3 e ResNet50..."):
                modelo_inc = carregar_inception()
                modelo_res = carregar_resnet()
                label_inc, conf_inc = prever_cnn(imagem, modelo_inc)
                label_res, conf_res = prever_cnn(imagem, modelo_res)

            with col_inc:
                st.markdown("**InceptionV3**")
                st.metric("Classificação", label_inc)
                st.metric("Confiança", f"{conf_inc*100:.1f}%")

            with col_res:
                st.markdown("**ResNet50**")
                st.metric("Classificação", label_res)
                st.metric("Confiança", f"{conf_res*100:.1f}%")

            # Consenso
            votos_weed = sum(1 for l in [label_inc, label_res] if "Erva" in l)
            if votos_weed >= 2:
                st.error("⚠️ Consenso CNN: **Presença dominante de ervas daninhas** confirmada.")
            elif votos_weed == 1:
                st.warning("⚠️ Consenso CNN: **Resultado misto** — verificar manualmente.")
            else:
                st.success("✅ Consenso CNN: **Cultura predominante** confirmada.")

else:
    st.info("Carregue uma imagem capturada no campo para iniciar a análise.")

st.sidebar.markdown("---")
st.sidebar.caption("Projeto Académico — Agricultura 4.0")
