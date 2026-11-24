"""Aplicación: detecta y cuenta las plantas de plátano y banano de una foto aérea.

Qué hace: abre una foto (JPG o PNG), la pasa por DETR, dibuja un rectángulo por planta con su
confianza, cuenta las plantas y muestra el tiempo de procesamiento.
Cómo se corre (desde la carpeta del proyecto):  streamlit run Codigo/app.py
Qué escribe: nada.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from reconocer import cargar_modelo, detectar, dibujar, sugerir_mosaicos


@st.cache_resource
def modelo_en_memoria():
    return cargar_modelo()


def main():
    st.set_page_config(page_title="Contador de plantas (UNIBAN)", layout="wide")
    st.title("Contador de plantas de plátano y banano")
    modelo, info = modelo_en_memoria()
    m = info["metricas"]
    st.caption("Modelo: DETR (ResNet-50), una sola clase («planta»; no distingue plátano de banano)."
               + (f" En prueba: AP@0.5 = {m['AP50']:.2f}, error medio de conteo = {m['MAE_conteo']:.1f} plantas por imagen."
                  if m else "") + f" Ejecución en {info['dispositivo']}.")

    archivo = st.file_uploader("Abrir una imagen aérea (vista cenital)", type=["jpg", "jpeg", "png"])
    if archivo is None:
        st.info("Sube una foto de dron tomada desde arriba. Si es muy grande (ortomosaico), "
                "divídela en mosaicos con el control de la izquierda.")
        return
    try:
        img = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")
    except Exception:
        st.error("No se pudo leer la imagen.")
        return

    with st.sidebar:
        st.header("Ajustes")
        umbral = st.slider("Umbral de confianza", 0.10, 0.99, info["umbral"], 0.01,
                           help="Solo se cuentan las plantas con confianza mayor a este valor. "
                                "El valor inicial es el que dio el mejor conteo en validación.")
        mosaicos = st.slider("Mosaicos por lado (N × N)", 1, 6, sugerir_mosaicos(*img.size),
                             key=f"mos_{archivo.name}_{img.size}",
                             help="El modelo mira 640×640 px por vez y detecta máximo 100 plantas por mosaico. "
                                  "Las fotos grandes deben partirse para que las plantas no queden diminutas.")
        st.caption(f"Foto de {img.width}×{img.height} px → {mosaicos * mosaicos} mosaico(s) de "
                   f"≈{img.width // mosaicos}×{img.height // mosaicos} px.")

    dets, ms, por_mosaico = detectar(img, modelo, info, umbral, mosaicos)
    anotada = dibujar(img, dets)

    col_img, col_res = st.columns([3, 2])
    with col_img:
        st.image(anotada)
        buf = io.BytesIO()
        anotada.save(buf, format="PNG")
        st.download_button("Descargar imagen con detecciones", buf.getvalue(), "plantas_detectadas.png", "image/png")
    with col_res:
        st.metric("Plantas detectadas", len(dets))
        st.metric("Tiempo de procesamiento", f"{ms:.0f} ms")
        if dets:
            st.metric("Confianza media", f"{np.mean([d['confianza'] for d in dets]) * 100:.1f} %")

    if dets:
        st.subheader("Detalle por planta")
        st.dataframe({"N.º": list(range(1, len(dets) + 1)),
                      "Confianza": [f"{d['confianza'] * 100:.1f} %" for d in dets],
                      "Ancho (px)": [d["caja"][2] for d in dets],
                      "Alto (px)": [d["caja"][3] for d in dets]}, hide_index=True)
    else:
        st.warning("No se encontraron plantas. Prueba bajar el umbral o cambiar el número de mosaicos.")

    if max(por_mosaico) >= 0.9 * info["consultas"]:
        st.warning(f"Algún mosaico llegó a {max(por_mosaico)} plantas, cerca del máximo de {info['consultas']} "
                   "que puede detectar el modelo. Aumenta el número de mosaicos.")
    if mosaicos > 1:
        st.caption("Con varios mosaicos, una planta cortada por la línea de división puede contarse dos veces "
                   "o no detectarse; el conteo total es aproximado en esas zonas.")


main()
