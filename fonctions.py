# fonctions_cleaned.py : version allégée uniquement pour le dashboard app4.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from cityscapesscripts.helpers.labels import name2label, trainId2label
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from typing import Tuple

# ------------ MÉTRIQUES POUR ÉVALUATION DES PRÉDICTIONS ----------------

def dice_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f == y_pred_f)
    return (2. * intersection) / (len(y_true_f) + len(y_pred_f) + smooth)

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    intersection = np.logical_and(y_true == y_pred, True).sum()
    union = np.logical_or(y_true >= 0, y_pred >= 0).sum()
    return (intersection + smooth) / (union + smooth)

# ------------ PALETTE + REMAPPING POUR MASQUES CITYSCAPES ----------------

CITYSCAPES_COLORS = {
    0: (0, 0, 0),         # void (noir)
    1: (75, 0, 130),      # flat (violet foncé)
    2: (124, 124, 124),   # construction (gris)
    3: (255, 102, 0),     # object (orange)
    4: (34, 139, 34),     # nature (vert forêt)
    5: (0, 139, 139),     # sky (cyan foncé)
    6: (178, 34, 34),     # human (rouge foncé)
    7: (0, 0, 255),       # vehicle (bleu)
}

def colorize_mask(mask):
    if mask.ndim == 3 and mask.shape[2] == 3:
        return mask
    if mask.ndim == 2:
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in CITYSCAPES_COLORS.items():
            color_mask[mask == cls_id] = color
        return color_mask
    raise ValueError("Le masque doit être 2D (classes) ou déjà RGB (3 canaux)")

def rgb_to_class(rgb_mask: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for cls_id, color in CITYSCAPES_COLORS.items():
        match = np.all(rgb_mask == color, axis=-1)
        class_mask[match] = cls_id
    return class_mask
