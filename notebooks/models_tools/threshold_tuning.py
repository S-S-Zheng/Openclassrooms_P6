import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Literal,Tuple

from notebooks.utils.metrics_classification import fbeta
from sklearn.metrics import confusion_matrix

def optimize_threshold(
    precisions:np.ndarray,
    recalls:np.ndarray,
    thresholds:np.ndarray,
    method: Literal['fbeta','recall','precision'] = "fbeta",
    target_value: Optional[float] = None,
    beta: float = 1.0
) -> Dict[str, Union[str, float]]:
    """
    Optimise le seuil de classification selon une métrique cible.
    
    Args:
        precisions: le vecteur de precisions ALIGNÉ
        recalls: le vecteur de recalls ALIGNÉ
        thresholds: le vecteur des seuils de validation
        method: Stratégie ('fbeta' pour max f-beta, 'recall' ou 'precision' pour contrainte).
        target_value: Valeur minimale à atteindre pour la méthode 'recall' ou 'precision'.
        beta: Poids (beta > 1 privilégie le rappel, beta < 1 la précision).
        
    Returns:
        Dictionnaire contenant les métriques au seuil optimal.\n
            {
                "method": f"{method}_opt" if method != "fbeta" else f"f{beta}",\n
                "best_f_score": float(np.max(f_scores)),\n
                "precision": precisions[mask][idx],\n
                "recall": recalls[mask][idx],\n
                "optimal_threshold": thresholds[mask][idx]\n
            }
    
    """

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
        raise ValueError("Méthode inconnue. Choisir parmi 'fbeta', 'recall', 'precision'.")

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


# =============================================================


def cost_function(
    y_true:Union[pd.Series, np.ndarray],
    y_pred:np.ndarray, 
    # threshold:float, 
    cost_fn:int=1, 
    cost_fp:int=1
)->Tuple[int,int,int,int,int]:
    """Calcule le coût métier total pour un seuil donné.

    Args:
        y_true (Union[pd.Series, np.ndarray]): Target réelle/test
        y_pred (np.ndarray): Target des prédiction binaires
        threshold (float): le seuil de validation
        cost_fn (int, optional): le cout sur les faux négatifs 
            (NON solvable accepté). Defaults to 1.
        cost_fp (int, optional): le cout sur les faux positifs 
            (solvable refusé). Defaults to 1.

    Returns:
        tuple: le cout, tn,fp,fn,tp
    """
    # # Définition des classes suivant le seuil
    # y_pred = (y_probas >= threshold).astype(int)
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost:int = (fn * cost_fn) + (fp * cost_fp)
    return total_cost,tn,fp,fn,tp