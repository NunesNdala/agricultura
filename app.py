import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import glob
import matplotlib.pyplot as plt

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
# Modelo treinado no MinneApple (contagem de maçãs) referido no relatório.
# NOTA: a aba de contagem ao vivo usa OWLv2 zero-shot, não este modelo.
# Este ficheiro é usado apenas para gerar as métricas reais desta secção.
MINNEAPPLE_PATH   = os.path.join(MODELOS_DIR, "yolo_minneapple_v2.pt")

# Dados de validação usados para calcular métricas REAIS (não inventadas).
# Estrutura esperada — ver aviso na aba "Métricas de Avaliação" se estiver em falta.
VALIDACAO_DIR          = "dados_validacao"
PLANTVILLAGE_VAL_DIR   = os.path.join(VALIDACAO_DIR, "plantvillage")        # pastas por classe
AGRI_CNN_VAL_DIR       = os.path.join(VALIDACAO_DIR, "agri_data_cnn")      # pastas "crop"/"weed"
AGRI_YOLO_DATA_YAML    = os.path.join(VALIDACAO_DIR, "agri_data_yolo", "data.yaml")
MINNEAPPLE_DATA_YAML   = os.path.join(VALIDACAO_DIR, "minneapple_yolo", "data.yaml")

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
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
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
    queries = ["ripe red apple"]
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
                if r['score'] < min_conf:
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
    if prob > 0.5:
        return "🌿 Erva Daninha", prob
    else:
        return "🌾 Cultura", 1.0 - prob

def _plot_matriz_confusao(y_true, y_pred, labels, titulo, normalizar=True, figsize=(6, 5)):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if normalizar:
        with np.errstate(all="ignore"):
            cm_show = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_show = np.nan_to_num(cm_show)
        fmt = ".2f"
    else:
        cm_show = cm
        fmt = "d"
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_show, cmap="Blues")
    ax.set_title(titulo)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    if len(labels) <= 6:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(cm_show[i, j], fmt), ha="center", va="center",
                         color="white" if cm_show[i, j] > cm_show.max()/2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, cm


def _plot_roc_binaria(y_true, y_score, titulo):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, roc_auc


def _plot_roc_multiclasse(y_true, y_prob, n_classes, titulo):
    """Curva ROC macro-average (one-vs-rest), útil para muitas classes (ex.: 38 doenças)."""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fpr_list, tpr_list = [], []
    all_fpr = np.linspace(0, 1, 200)
    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        fpr_list.append(fpr); tpr_list.append(tpr)
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= max(len(fpr_list), 1)
    macro_auc = auc(all_fpr, mean_tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(all_fpr, mean_tpr, label=f"Macro-average AUC = {macro_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, macro_auc


@st.cache_data(show_spinner=False)
def _avaliar_efficientnet_cache():
    """Corre o EfficientNetB0 sobre o conjunto de validação real e devolve y_true/y_pred/y_prob."""
    import tensorflow as tf
    modelo = tf.keras.models.load_model(EFFICIENTNET_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    ds = tf.keras.utils.image_dataset_from_directory(
        PLANTVILLAGE_VAL_DIR, image_size=IMG_SIZE, batch_size=32,
        shuffle=False, label_mode="int"
    )
    pastas = ds.class_names  # ordem alfabética das subpastas do disco
    # mapear índice da pasta -> índice usado pelo modelo (class_names.json)
    mapa = [classes.index(nome) for nome in pastas]

    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(modelo.predict(x, verbose=0))
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_true = np.array([mapa[i] for i in y_true])
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob, classes


@st.cache_data(show_spinner=False)
def _avaliar_cnn_binaria_cache(caminho_modelo):
    """Avalia InceptionV3 ou ResNet50 (crop=0 / weed=1) no conjunto de validação real."""
    import tensorflow as tf
    modelo = tf.keras.models.load_model(caminho_modelo, compile=False)
    ds = tf.keras.utils.image_dataset_from_directory(
        AGRI_CNN_VAL_DIR, image_size=IMG_SIZE, batch_size=32,
        shuffle=False, label_mode="binary"
    )
    y_true, y_prob = [], []
    for x, y in ds:
        x = tf.cast(x, tf.float32) / 255.0
        y_prob.append(modelo.predict(x, verbose=0).ravel())
        y_true.append(y.numpy().ravel())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    return y_true, y_pred, y_prob


@st.cache_data(show_spinner=False)
def _avaliar_yolo_cache(caminho_modelo, data_yaml):
    """Corre model.val() do Ultralytics: gera matriz de confusão, PR/F1/P/R curves automaticamente."""
    from ultralytics import YOLO
    modelo = YOLO(caminho_modelo)
    resultados = modelo.val(data=data_yaml, split="val", plots=True, verbose=False)
    return {
        "save_dir": str(resultados.save_dir),
        "map50": float(resultados.box.map50),
        "map50_95": float(resultados.box.map),
        "precision": float(resultados.box.mp),
        "recall": float(resultados.box.mr),
    }


def _mostrar_imagens_yolo(save_dir):
    nomes = ["confusion_matrix.png", "confusion_matrix_normalized.png",
             "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]
    cols = st.columns(2)
    i = 0
    for nome in nomes:
        caminho = os.path.join(save_dir, nome)
        if os.path.exists(caminho):
            with cols[i % 2]:
                st.image(caminho, caption=nome, width="stretch")
            i += 1
    if i == 0:
        st.warning("O Ultralytics não gerou imagens de avaliação — verifica a versão do pacote.")


def mostrar_metricas_avaliacao():
    st.subheader("Métricas de Avaliação dos Modelos")
    st.caption(
        "Estas métricas são calculadas em tempo real a partir dos modelos e do conjunto de "
        "validação em `dados_validacao/` — não são valores fixos no código."
    )

    if not os.path.isdir(VALIDACAO_DIR):
        st.error(
            "Pasta `dados_validacao/` não encontrada. Estrutura esperada:\n\n"
            "```\n"
            "dados_validacao/\n"
            "├── plantvillage/            # subpasta por classe (mesmos nomes de class_names.json)\n"
            "├── agri_data_cnn/           # subpastas 'crop' e 'weed'\n"
            "├── agri_data_yolo/data.yaml # apontando para imagens/labels YOLO de validação\n"
            "└── minneapple_yolo/data.yaml\n"
            "```\n"
            "Sem estes dados não é possível calcular matrizes de confusão nem curvas ROC/PR reais."
        )
        return

    aba_doencas, aba_ervas, aba_macas = st.tabs([
        "Diagnóstico de Doenças", "Deteção de Ervas Daninhas", "Contagem de Maçãs"
    ])

    # ---------------- Doenças (EfficientNetB0, 38 classes) ----------------
    with aba_doencas:
        if not os.path.isdir(PLANTVILLAGE_VAL_DIR):
            st.warning(f"Não encontrei `{PLANTVILLAGE_VAL_DIR}`. Coloca aí o conjunto de validação do PlantVillage (pastas por classe).")
        elif st.button("Calcular métricas — EfficientNetB0", key="btn_doencas"):
            with st.spinner("A correr o modelo sobre o conjunto de validação (pode demorar vários minutos)..."):
                y_true, y_pred, y_prob, classes = _avaliar_efficientnet_cache()
            acc = float((y_true == y_pred).mean())
            st.metric("Acurácia real no conjunto de validação", f"{acc:.2%}")

            fig_cm, cm = _plot_matriz_confusao(y_true, y_pred, classes, "Matriz de Confusão (normalizada) — EfficientNetB0", figsize=(9, 8))
            st.pyplot(fig_cm)

            fig_roc, macro_auc = _plot_roc_multiclasse(y_true, y_prob, len(classes), "Curva ROC macro-average (one-vs-rest)")
            st.pyplot(fig_roc)

            from sklearn.metrics import classification_report
            relatorio = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(relatorio).transpose(), width="stretch")
        else:
            st.info("Clica no botão para correr a avaliação (não é feita automaticamente por ser pesada).")

    # ---------------- Ervas daninhas: YOLOv8 + InceptionV3 + ResNet50 ----------------
    with aba_ervas:
        st.markdown("**YOLOv8n — deteção (matriz de confusão + curvas PR/F1 nativas do Ultralytics)**")
        if not os.path.exists(AGRI_YOLO_DATA_YAML):
            st.warning(f"Não encontrei `{AGRI_YOLO_DATA_YAML}`.")
        elif st.button("Calcular métricas — YOLOv8n (ervas daninhas)", key="btn_yolo_ervas"):
            with st.spinner("A correr model.val()..."):
                res = _avaliar_yolo_cache(WEED_PATH, AGRI_YOLO_DATA_YAML)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("mAP50", f"{res['map50']:.1%}")
            c2.metric("mAP50-95", f"{res['map50_95']:.1%}")
            c3.metric("Precisão", f"{res['precision']:.1%}")
            c4.metric("Recall", f"{res['recall']:.1%}")
            _mostrar_imagens_yolo(res["save_dir"])

        st.markdown("---")
        st.markdown("**InceptionV3 e ResNet50 — classificação binária (crop / weed)**")
        if not os.path.isdir(AGRI_CNN_VAL_DIR):
            st.warning(f"Não encontrei `{AGRI_CNN_VAL_DIR}`.")
        elif st.button("Calcular métricas — InceptionV3 e ResNet50", key="btn_cnn_ervas"):
            with st.spinner("A avaliar InceptionV3..."):
                yt_i, yp_i, ys_i = _avaliar_cnn_binaria_cache(INCEPTION_PATH)
            with st.spinner("A avaliar ResNet50..."):
                yt_r, yp_r, ys_r = _avaliar_cnn_binaria_cache(RESNET_PATH)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**InceptionV3**")
                st.metric("Acurácia", f"{(yt_i == yp_i).mean():.2%}")
                fig_cm, _ = _plot_matriz_confusao(yt_i, yp_i, ["crop", "weed"], "Matriz de Confusão — InceptionV3", figsize=(4, 4))
                st.pyplot(fig_cm)
                fig_roc, auc_i = _plot_roc_binaria(yt_i, ys_i, "Curva ROC — InceptionV3")
                st.pyplot(fig_roc)
            with col2:
                st.markdown("**ResNet50**")
                st.metric("Acurácia", f"{(yt_r == yp_r).mean():.2%}")
                fig_cm, _ = _plot_matriz_confusao(yt_r, yp_r, ["crop", "weed"], "Matriz de Confusão — ResNet50", figsize=(4, 4))
                st.pyplot(fig_cm)
                fig_roc, auc_r = _plot_roc_binaria(yt_r, ys_r, "Curva ROC — ResNet50")
                st.pyplot(fig_roc)

    # ---------------- Contagem de maçãs (YOLOv8n treinado no MinneApple) ----------------
    with aba_macas:
        st.caption(
            "⚠️ A aba de contagem ao vivo desta app usa OWLv2 (zero-shot), não este modelo. "
            "Estas métricas avaliam o YOLOv8n treinado no MinneApple, referido no relatório."
        )
        if not os.path.exists(MINNEAPPLE_PATH):
            st.warning(f"Não encontrei `{MINNEAPPLE_PATH}`. Coloca aí os pesos do YOLOv8n treinado no MinneApple.")
        elif not os.path.exists(MINNEAPPLE_DATA_YAML):
            st.warning(f"Não encontrei `{MINNEAPPLE_DATA_YAML}`.")
        elif st.button("Calcular métricas — YOLOv8n (MinneApple)", key="btn_yolo_macas"):
            with st.spinner("A correr model.val()..."):
                res = _avaliar_yolo_cache(MINNEAPPLE_PATH, MINNEAPPLE_DATA_YAML)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("mAP50", f"{res['map50']:.1%}")
            c2.metric("mAP50-95", f"{res['map50_95']:.1%}")
            c3.metric("Precisão", f"{res['precision']:.1%}")
            c4.metric("Recall", f"{res['recall']:.1%}")
            _mostrar_imagens_yolo(res["save_dir"])


# ---------- Interface ----------

st.title("🌱 Visão Computacional Aplicada à Agricultura Inteligente")
st.caption("Sistema de apoio à decisão para contagem de precisão e diagnóstico fitossanitário.")

st.sidebar.header("Módulos de Análise")
opcao = st.sidebar.radio("Selecione a tarefa:", [
    "🍎 Contagem de Precisão (Maçãs)",
    "🍃 Diagnóstico de Doenças",
    "🌿 Deteção de Ervas Daninhas",
    "🔍 Deteção Geral (Multiclasse)",
    "📊 Métricas de Avaliação"
])

if opcao == "🍎 Contagem de Precisão (Maçãs)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ajuste de Sensibilidade")
    min_conf = st.sidebar.slider("Limiar de Confiança", 0.10, 0.80, 0.35, 0.05)
    tile_size = st.sidebar.select_slider("Janela de Análise (px)", [200, 300, 400, 500, 640], value=500)
    st.sidebar.caption("⚠️ Pode demorar 2–4 minutos.")

elif opcao == "🔍 Deteção Geral (Multiclasse)":
    conf_threshold = st.sidebar.slider("Confiança YOLO", 0.10, 0.90, 0.25, 0.05)

elif opcao == "🌿 Deteção de Ervas Daninhas":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuração")
    conf_weed = st.sidebar.slider("Confiança YOLO", 0.10, 0.90, 0.40, 0.05)
    usar_cnn = st.sidebar.checkbox("Confirmação por CNN (InceptionV3 + ResNet50)", value=True)

if opcao == "📊 Métricas de Avaliação":
    mostrar_metricas_avaliacao()
    st.stop()

uploaded_file = st.file_uploader("Carregar Imagem de Campo", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    imagem = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entrada")
        st.image(imagem, width="stretch")

    if opcao == "🍎 Contagem de Precisão (Maçãs)":
        with st.spinner("A processar filtragem de folhagem e contagem..."):
            n, img_anotada = contar_owlv2(imagem, tile_size=tile_size, min_conf=min_conf)
        with col2:
            st.subheader("Resultado da Contagem")
            st.metric("🍎 Maçãs Confirmadas", n)
            st.image(img_anotada, width="stretch")
        st.caption("⚠️ OWLv2 zero-shot. Ajuste o limiar de confiança se a contagem parecer desviada.")

    elif opcao == "🍃 Diagnóstico de Doenças":
        with st.spinner("A analisar tecidos vegetais..."):
            resultados = prever_doenca(imagem)
        with col2:
            st.subheader("Diagnóstico")
            classe_top, conf_top = resultados[0]
            st.success(f"**{traduzir_classe(classe_top)}**")
            st.metric("Confiança", f"{conf_top*100:.1f}%")
            st.write("Outras possibilidades:")
            for classe, conf in resultados[1:]:
                st.write(f"- {traduzir_classe(classe)}: {conf*100:.1f}%")
        st.caption("⚠️ EfficientNetB0 treinado no PlantVillage (38 classes, 99% accuracy).")

    elif opcao == "🔍 Deteção Geral (Multiclasse)":
        with st.spinner("A detetar objetos e organizar dados..."):
            img_anotada, contagens = detetar_frutas_veg(imagem, conf=conf_threshold)
        with col2:
            st.subheader("Resultados da Deteção")
            st.image(img_anotada, width="stretch")
            if contagens:
                import pandas as pd
                st.write("**Dados Quantitativos:**")
                df = pd.DataFrame(list(contagens.items()), columns=['Classe', 'Quantidade'])
                st.table(df)
            else:
                st.warning("Nenhum objeto detetado com o threshold atual.")
        st.caption("⚠️ YOLOv8 treinado no LVIS (63 classes de frutas e vegetais).")

    else:
        with st.spinner("A mapear infestantes..."):
            n_crop, n_weed, img_anotada = detetar_ervas(imagem, conf=conf_weed)
        with col2:
            st.subheader("Métricas de Campo — YOLOv8")
            c1, c2 = st.columns(2)
            c1.metric("🌾 Cultura", n_crop)
            c2.metric("🌿 Ervas Daninhas", n_weed)
            st.image(img_anotada, width="stretch")
        st.caption("⚠️ YOLOv8n treinado em dataset de sésamo + ervas daninhas (mAP50: 0.826).")

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
