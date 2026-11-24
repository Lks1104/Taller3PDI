"""Arquitectura DETR para inferencia (la misma del cuaderno DETR_Plantas_UNIBAN.ipynb).

Qué hace: define el modelo con los MISMOS nombres de capas del cuaderno, para que los pesos
guardados (detr_plantas_front.pt) encajen. Solo sirve para predecir: no trae pérdida,
emparejamiento húngaro, aumentación ni código de entrenamiento.
Cómo se corre: no se corre; se importa desde reconocer.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):
    def __init__(self, name="resnet50"):
        super().__init__()
        m = getattr(torchvision.models, name)(weights=None, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.num_channels = 2048 if name in ("resnet50", "resnet101") else 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def sine_position_encoding(h, w, d_model=256, temperature=10000.0, device="cpu"):
    npf = d_model // 2
    y = torch.arange(1, h + 1, dtype=torch.float32, device=device)[:, None].expand(h, w)
    x = torch.arange(1, w + 1, dtype=torch.float32, device=device)[None, :].expand(h, w)
    eps, scale = 1e-6, 2 * math.pi
    y = y / (h + eps) * scale
    x = x / (w + eps) * scale
    dim_t = temperature ** (2 * (torch.arange(npf, device=device) // 2) / npf)
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1).flatten(2)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1).flatten(2)
    return torch.cat((py, px), dim=-1).flatten(0, 1)


class EncoderLayer(nn.Module):
    def __init__(s, d, h, ff, p=0.1):
        super().__init__()
        s.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.l1, s.l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.drop, s.d1, s.d2 = nn.Dropout(p), nn.Dropout(p), nn.Dropout(p)

    def forward(s, src, pos):
        q = k = src + pos
        src = s.n1(src + s.d1(s.attn(q, k, src, need_weights=False)[0]))
        return s.n2(src + s.d2(s.l2(s.drop(F.relu(s.l1(src))))))


class DecoderLayer(nn.Module):
    def __init__(s, d, h, ff, p=0.1):
        super().__init__()
        s.self_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.cross_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.l1, s.l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        s.n1, s.n2, s.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.drop, s.d1, s.d2, s.d3 = nn.Dropout(p), nn.Dropout(p), nn.Dropout(p), nn.Dropout(p)

    def forward(s, tgt, memory, pos, query_pos):
        q = k = tgt + query_pos
        tgt = s.n1(tgt + s.d1(s.self_attn(q, k, tgt, need_weights=False)[0]))
        out = s.cross_attn(tgt + query_pos, memory + pos, memory, need_weights=False)[0]
        tgt = s.n2(tgt + s.d2(out))
        return s.n3(tgt + s.d3(s.l2(s.drop(F.relu(s.l1(tgt))))))


class Transformer(nn.Module):
    def __init__(s, d=256, h=8, ff=2048, n_enc=6, n_dec=6, p=0.1):
        super().__init__()
        s.enc = nn.ModuleList([EncoderLayer(d, h, ff, p) for _ in range(n_enc)])
        s.dec = nn.ModuleList([DecoderLayer(d, h, ff, p) for _ in range(n_dec)])
        s.dec_norm = nn.LayerNorm(d)

    def forward(s, src, pos, query_embed):
        B = src.shape[0]
        memory = src
        for layer in s.enc:
            memory = layer(memory, pos)
        qpos = query_embed.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(qpos)
        for layer in s.dec:
            tgt = layer(tgt, memory, pos, qpos)
        return s.dec_norm(tgt)                      # (B, Q, d): solo la última capa


class MLP(nn.Module):
    def __init__(s, i, h, o, n):
        super().__init__()
        dims = [i] + [h] * (n - 1)
        s.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:] + [o])])

    def forward(s, x):
        for k, l in enumerate(s.layers):
            x = F.relu(l(x)) if k < len(s.layers) - 1 else l(x)
        return x


class DETR(nn.Module):
    def __init__(s, num_classes, backbone="resnet50", d_model=256, nheads=8, dim_ffn=2048,
                 enc_layers=6, dec_layers=6, num_queries=100, **_):
        super().__init__()
        s.backbone = Backbone(backbone)
        s.input_proj = nn.Conv2d(s.backbone.num_channels, d_model, kernel_size=1)
        s.transformer = Transformer(d_model, nheads, dim_ffn, enc_layers, dec_layers)
        s.query_embed = nn.Embedding(num_queries, d_model)
        s.class_embed = nn.Linear(d_model, num_classes + 1)     # +1 = «sin objeto»
        s.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(s, x):
        src = s.input_proj(s.backbone(x))
        B, d, h, w = src.shape
        src = src.flatten(2).transpose(1, 2)
        pos = sine_position_encoding(h, w, d, device=x.device).unsqueeze(0).expand(B, -1, -1)
        hs = s.transformer(src, pos, s.query_embed.weight)
        return {"pred_logits": s.class_embed(hs), "pred_boxes": s.bbox_embed(hs).sigmoid()}
