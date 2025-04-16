# fonctions : version allégée uniquement pour le dashboard app4.py

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

def dice_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 8, smooth: float = 1e-6) -> float:
    """
    Calcule le Dice Coefficient moyen (macro) entre deux masques indexés (valeurs 0 à num_classes-1).
    """
    dice = 0.0
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(np.uint8)
        y_pred_c = (y_pred == c).astype(np.uint8)
        intersection = np.sum(y_true_c * y_pred_c)
        union = np.sum(y_true_c) + np.sum(y_pred_c)
        if union == 0:
            dice += 1.0  # Si la classe est absente dans les 2, on considère Dice = 1 (par convention)
        else:
            dice += (2.0 * intersection + smooth) / (union + smooth)
    return dice / num_classes

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 8, smooth: float = 1e-6) -> float:
    """
    Calcule le Mean IoU (Intersection over Union) moyen (macro) entre deux masques indexés.
    """
    iou = 0.0
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(np.uint8)
        y_pred_c = (y_pred == c).astype(np.uint8)
        intersection = np.sum(y_true_c * y_pred_c)
        union = np.sum(np.logical_or(y_true_c, y_pred_c))
        if union == 0:
            iou += 1.0  # Si la classe est absente dans les deux, on considère IoU = 1
        else:
            iou += (intersection + smooth) / (union + smooth)
    return iou / num_classes

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
