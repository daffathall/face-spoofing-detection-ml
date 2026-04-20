# src/model.py — Support CNN (GeM pooling) + Transformer (direct pooling)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class FaceAntispoofModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, drop_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="",
        )
        feat_dim = self.backbone.num_features

        # Deteksi apakah model ini Transformer (output 2D) atau CNN (output 4D)
        self.is_transformer = any(k in model_name for k in
                                   ["swin", "vit", "deit", "beit", "coat"])

        if not self.is_transformer:
            self.pool = GeMPooling(p=3.0)
        else:
            # Swin output: (B, H*W, C) atau (B, C) tergantung config
            # global_pool="" → kita handle sendiri
            self.pool = nn.AdaptiveAvgPool1d(1)  # untuk sequence output

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)

        if self.is_transformer:
            # Swin dengan global_pool="" → output (B, H*W, C) atau (B, C)
            if feat.dim() == 3:
                # (B, seq_len, C) → average over sequence → (B, C)
                feat = feat.mean(dim=1)
            elif feat.dim() == 4:
                # (B, H, W, C) format Swin → (B, C, H, W) → GeM
                feat = feat.permute(0, 3, 1, 2)
                feat = F.adaptive_avg_pool2d(feat, 1)
            # feat sekarang (B, C) atau (B, C, 1, 1)
            if feat.dim() == 4:
                feat = feat.flatten(1)
        else:
            # CNN: (B, C, H, W) → GeM → (B, C, 1, 1)
            feat = self.pool(feat)

        return self.head(feat)


def build_model(model_name, num_classes, pretrained=True, drop_rate=0.4):
    return FaceAntispoofModel(model_name, num_classes,
                               pretrained=pretrained, drop_rate=drop_rate)
