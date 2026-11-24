"""Detección y conteo de plantas en una foto aérea (lo usa la aplicación).

Qué hace: carga el modelo DETR entrenado, parte la foto en mosaicos si hace falta, redimensiona
cada mosaico a 640x640 (igual que en el entrenamiento), predice y devuelve las cajas en
coordenadas de la foto original, con su confianza.
Cómo se corre: no se corre; se importa. El modelo se carga una sola vez (cargar_modelo).
Qué escribe: nada.
"""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from detr_modelo import DETR

RAIZ = Path(__file__).resolve().parents[1]
RUTA_MODELO = RAIZ / "modelos" / "detr_plantas_front.pt"
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
COLOR_CAJA = (230, 40, 40)


def cargar_modelo(ruta=RUTA_MODELO):
    """Devuelve (modelo, info). Acepta el archivo exportado (detr_plantas_front.pt) o el
    checkpoint crudo detr_plantas_mejor.pt del cuaderno (en ese caso el umbral es 0.9)."""
    ck = torch.load(ruta, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    modelo = DETR(**cfg)
    modelo.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["model"].items()})
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo).eval()
    info = {"clases": ck.get("clases", ["planta"]), "umbral": float(ck.get("umbral", 0.9)),
            "lado": int(cfg["img_size"]), "consultas": int(cfg["num_queries"]),
            "metricas": ck.get("metricas", {}), "dispositivo": str(dispositivo)}
    return modelo, info


def sugerir_mosaicos(ancho, alto):
    """Heurística: un mosaico por cada ~800 px del lado mayor (en el entrenamiento, las fotos de
    1024-1280 px se partieron en 2x2). Es solo el valor inicial: el usuario puede cambiarlo."""
    return int(min(6, max(1, round(max(ancho, alto) / 800))))


def _tensor(img, lado):
    if img.size != (lado, lado):
        img = img.resize((lado, lado))          # mismo resize que YoloDataset
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (x - MEDIA) / STD


@torch.no_grad()
def detectar(img, modelo, info, umbral=None, mosaicos=1):
    """img: PIL.Image. Devuelve (detecciones, milisegundos, detecciones_por_mosaico).
    Cada detección: caja (x, y, w, h) en píxeles de la foto, clase, confianza y area_px."""
    t0 = time.perf_counter()
    umbral = info["umbral"] if umbral is None else umbral
    img = img.convert("RGB")
    W, H = img.size
    xs = np.linspace(0, W, mosaicos + 1).round().astype(int)
    ys = np.linspace(0, H, mosaicos + 1).round().astype(int)
    cajas_mosaico = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(mosaicos) for i in range(mosaicos)]
    disp = next(modelo.parameters()).device

    dets, por_mosaico = [], []
    for ini in range(0, len(cajas_mosaico), 4):                 # lotes de 4 mosaicos
        lote = cajas_mosaico[ini:ini + 4]
        x = torch.stack([_tensor(img.crop(c), info["lado"]) for c in lote]).to(disp)
        out = modelo(x)
        prob = out["pred_logits"].softmax(-1)[..., :-1].cpu()   # sin la clase «sin objeto»
        conf, clase = prob.max(-1)
        cajas = out["pred_boxes"].cpu()
        for b, (x0, y0, x1, y1) in enumerate(lote):
            tw, th = x1 - x0, y1 - y0
            k = conf[b] > umbral
            por_mosaico.append(int(k.sum()))
            for (cx, cy, w, h), c, cl in zip(cajas[b][k].tolist(), conf[b][k].tolist(), clase[b][k].tolist()):
                bx, by = x0 + (cx - w / 2) * tw, y0 + (cy - h / 2) * th
                dets.append({"caja": (int(bx), int(by), int(w * tw), int(h * th)), "clase": info["clases"][cl],
                             "confianza": float(c), "area_px": int(w * tw * h * th)})
    dets.sort(key=lambda d: (d["caja"][1], d["caja"][0]))
    return dets, (time.perf_counter() - t0) * 1000.0, por_mosaico


def dibujar(img, dets, lado_max=1600, mosaicos=1):
    """Copia de la foto (reducida a lado_max como mucho) con un rectángulo por planta."""
    img = img.convert("RGB")
    s = min(1.0, lado_max / max(img.size))
    if s < 1.0:
        img = img.resize((int(img.width * s), int(img.height * s)))
    d = ImageDraw.Draw(img)
    for p in dets:
        x, y, w, h = [v * s for v in p["caja"]]
        d.rectangle([x, y, x + w, y + h], outline=COLOR_CAJA, width=2)
        if len(dets) <= 80:
            d.text((x + 2, max(y - 11, 0)), f"{p['confianza']:.2f}", fill=COLOR_CAJA)
    return img
