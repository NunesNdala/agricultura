

def run_analysis(option, image, params):
    if option == "disease":
        return {"kind": option, "results": predict_disease(image)}

    if option == "fruit_yolo":
        count, annotated = count_fruits_yolo(image, conf=params["conf"])
        return {"kind": option, "count": count, "image": annotated}

    if option == "fruit_owlv2":
        count, annotated, raw_count, filtered_count, threshold = count_fruits_owlv2(
            image,
            tile_size=params["tile_size"],
            overlap=0.35,
        )
        return {
            "kind": option,
            "count": count,
            "raw_count": raw_count,
            "filtered_count": filtered_count,
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
        st.caption(
            f"Detecoes recolhidas: {result['raw_count']} | "
            f"Apos filtro visual: {result['filtered_count']}"
        )
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
        st.session_state["last_error"] = None
        with st.spinner("A processar imagem..."):
            try:
                st.session_state["last_result"] = run_analysis(option, image, params)
            except Exception as exc:
                st.session_state["last_error"] = str(exc)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem original")
        st.image(image, use_container_width=True)

    with col2:
        if st.session_state.get("last_key") == current_key and st.session_state.get("last_result"):
            show_result(st.session_state["last_result"])
        elif st.session_state.get("last_key") == current_key and st.session_state.get("last_error"):
            st.error(st.session_state["last_error"])
            if option == "fruit_yolo":
                st.info("Alternativa: selecione OWLv2 Sliding Window para contar as macas sem depender do YOLOv8.")
        else:
            st.info("Clique em Analisar imagem para gerar um novo resultado com os parametros atuais.")

    if option == "fruit_owlv2" and st.session_state.get("last_key") == current_key:
        result = st.session_state.get("last_result")
        if result and result["kind"] == "fruit_owlv2":
            with st.expander("Comparar com YOLOv8 (mesma imagem)"):
                if st.button("Correr comparacao YOLOv8 vs OWLv2"):
                    with st.spinner("A correr YOLOv8 para comparacao..."):
                        try:
                            yolo_count, yolo_image = count_fruits_yolo(image, conf=0.35)
                        except Exception as exc:
                            st.warning(f"Nao foi possivel correr YOLOv8 neste ambiente: {exc}")
                        else:
                            c1, c2 = st.columns(2)
                            c1.metric("YOLOv8", yolo_count)
                            c1.image(yolo_image, use_container_width=True)
                            c2.metric("OWLv2 Sliding Window", result["count"], delta=result["count"] - yolo_count)
                            c2.image(result["image"], use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Projeto academico - Visao Computacional Aplicada a Agricultura Inteligente")
