"""Convierte el checkpoint del cuaderno (detr_plantas_mejor.pt) en el archivo que usa la aplicación.

Qué hace: lee modelos/detr_plantas_mejor.pt (pesos + configuración), le añade las clases, el
umbral de confianza y las métricas de prueba, guarda los pesos en media precisión (~83 MB en vez
de ~166 MB) y escribe modelos/detr_plantas_front.pt.
Cómo se corre (desde la carpeta del proyecto):  python exportar_modelo.py
Requisitos: solo torch. No necesita el cuaderno, ni GPU, ni el dataset.
Qué escribe: modelos/detr_plantas_front.pt (no modifica el archivo de entrada).
"""
import argparse
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent

# Valores tomados de la salida del cuaderno (secciones 7.1 y 7.2). Si vuelven a entrenar el
# modelo, actualícenlos con los números nuevos del cuaderno.
UMBRAL = 0.9
CLASES = ["planta"]
METRICAS = {"AP50": 0.924, "AP50_95": 0.542, "MAE_conteo": 1.27, "error_conteo_total_pct": 3.2}
CLAVES_CFG = ("backbone", "d_model", "nheads", "dim_ffn", "enc_layers", "dec_layers",
              "num_queries", "num_classes", "img_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=str(RAIZ / "modelos" / "detr_plantas_mejor.pt"))
    ap.add_argument("--salida", default=str(RAIZ / "modelos" / "detr_plantas_front.pt"))
    ap.add_argument("--fp32", action="store_true", help="no pasar a media precisión (archivo de ~166 MB)")
    a = ap.parse_args()

    try:
        ck = torch.load(a.entrada, map_location="cpu", weights_only=True)
    except Exception:  # checkpoint de una versión de torch con otros tipos guardados
        ck = torch.load(a.entrada, map_location="cpu", weights_only=False)
    pesos = ck["model"]
    if not a.fp32:
        pesos = {k: (v.half() if v.is_floating_point() else v) for k, v in pesos.items()}
    exp = {"model": pesos,
           "cfg": {k: ck["cfg"][k] for k in CLAVES_CFG},
           "clases": CLASES, "umbral": UMBRAL, "metricas": METRICAS}
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    torch.save(exp, a.salida)

    # comprobación: el archivo se vuelve a abrir como lo hará la aplicación
    chk = torch.load(a.salida, map_location="cpu", weights_only=True)
    n = sum(v.numel() for v in chk["model"].values() if v.is_floating_point())
    print(f"Entrada : {a.entrada} (época {ck.get('epoch', '?')})")
    print(f"Salida  : {a.salida} | {Path(a.salida).stat().st_size / 1e6:.0f} MB | {n / 1e6:.1f} M parámetros")
    print(f"Clases {chk['clases']} | umbral {chk['umbral']} | cfg {chk['cfg']}")


if __name__ == "__main__":
    main()
