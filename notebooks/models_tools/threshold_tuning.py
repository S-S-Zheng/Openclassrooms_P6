import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from typing import Dict, Union, Optional, Literal

from notebooks.utils.metrics_classification import fbeta


def optimize_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    method: Literal['fbeta','recall','precision'] = "fbeta",
    target_value: Optional[float] = None,
    beta: float = 1.0
) -> Dict[str, Union[str, float]]:
    """
    Optimise le seuil de classification selon une métrique cible.
    
    Args:
        y_true: Étiquettes réelles.
        y_proba: Probabilités de la classe positive (1D array).
        method: Stratégie ('fbeta' pour max f-beta, 'recall' ou 'precision' pour contrainte).
        target_value: Valeur minimale à atteindre pour la méthode 'recall' ou 'precision'.
        beta: Poids (beta > 1 privilégie le rappel, beta < 1 la précision).
        
    Returns:
        Dictionnaire contenant les métriques au seuil optimal.
    """
    # Calcul de la courbe Precision-Recall
    # thresholds renvoyé par sklearn a une longueur n_thresholds
    # (un de moins que prec/rec)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # On aligne les vecteurs (sklearn ajoute 1 et 0 à la fin de prec/rec)
    precisions, recalls = precisions[:-1], recalls[:-1]

    # Définition du masque de recherche selon la méthode
    if method == "fbeta":
        mask = np.ones_like(thresholds, dtype=bool)
        current_beta = beta
    elif method == "recall":
        if target_value is None: 
            raise ValueError("target_value requis pour recall")
        mask = recalls >= target_value
        current_beta = 2.0  # Orienté métier : on veut le meilleur compromis si recall garanti
    elif method == "precision":
        if target_value is None: 
            raise ValueError("target_value requis pour precision")
        mask = precisions >= target_value
        current_beta = 0.5
    else:
        raise ValueError("Méthode inconnue. Choisir parmi 'f', 'recall', 'precision'.")

    # Application du masque et calcul du score F-beta
    if not np.any(mask):
        print(f"Aucun seuil ne satisfait la contrainte {method} >= {target_value}")
        # Fallback sur le meilleur possible si la contrainte est trop forte
        mask = np.ones_like(thresholds, dtype=bool)

    f_scores = fbeta(precisions[mask], recalls[mask], beta=current_beta)
    idx = np.argmax(f_scores)

    # Extraction des résultats optimaux
    return {
        "method": f"{method}_opt" if method != "fbeta" else f"f{beta}",
        "best_f_score": float(np.max(f_scores)),
        "precision": precisions[mask][idx],
        "recall": recalls[mask][idx],
        "optimal_threshold": thresholds[mask][idx]
    }