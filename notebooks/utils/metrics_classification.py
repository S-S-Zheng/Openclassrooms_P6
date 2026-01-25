import numpy as np
import pandas as pd
from typing import List, Tuple, Union


def top_score(
    df: pd.DataFrame,
    metrics: List[str],
    weights_configs: List[List[float]],
    ref_col: str,
    extra_cols: Union[List[str], None] = None,
    top_n: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calcule un score basé sur une moyenne pondérée des rangs des métriques.
    
    Args:
        df: DataFrame contenant les résultats.
        metrics: Liste des noms de colonnes (métriques).
        weights_configs: Liste de [poids, sens] (ex: [1, -1] pour minimiser).
        ref_col: Colonne d'identification (ex: 'model_name').
        extra_cols: Colonnes additionnelles à conserver dans le rapport.
        top_n: Nombre de lignes pour le classement final.
    """
    
    #  VALIDATION
    if len(metrics) != len(weights_configs):
        raise ValueError("Les listes 'metrics' et 'weights_configs' doivent avoir la même taille.")
    
    extra_cols = extra_cols or []
    required_cols = set(metrics) | {ref_col} | set(extra_cols)
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes absentes : {missing}")

    # CALCUL DES RANGS
    # On travaille sur une DataFrame dédiée aux rangs pour économiser la RAM
    rank_df = pd.DataFrame(index=df.index)
    
    for metric, (_, sense) in zip(metrics, weights_configs):
        # Si sense >= 0 (ex: R2), on veut le plus grand : ascending=False
        # Si sense < 0 (ex: MAE), on veut le plus petit : ascending=True
        is_higher_better = sense >= 0
        # Calcul du rang: 1 est toujours le "meilleur"
        # Ascending est inversé par rapport a is_higher_better
        # (==> higher_better = True implique Ascending False)
        # On range automatiquement en dernier les NaN dans le même sens que higherr_better
        rank_df[metric] = df[metric].rank(
            ascending=not is_higher_better, 
            method='min', 
            na_option='bottom' if is_higher_better else 'top'
        )
    
    # MOYENNE PONDÉRÉE
    # Formule : $$Score = \frac{\sum (Rank_i \times Weight_i)}{\sum Weight_i}$$
    weights_only = [weight[0] for weight in weights_configs]
    total_weight = sum(weights_only)
    
    # Calcul vectorisé du score
    # Utilisation de mul plutot que boucle for pour multiplier chaque colonne par son poids
    weighted_ranks = rank_df[metrics].mul(weights_only, axis=1)
    rank_df['final_score'] = weighted_ranks.sum(axis=1) / total_weight


    # FORMATAGE DES SORTIES
    # On réintègre les infos de base
    full_report = pd.concat([df[[ref_col] + extra_cols], rank_df['final_score']], axis=1)
    full_report = full_report.sort_values('final_score').reset_index(drop=True)


    # Classement par modèle (top N)
    model_ranking = (
        full_report.groupby(ref_col)['final_score']
        .mean()
        .sort_values()
        .head(top_n)
    )

    return full_report, model_ranking


# =============================================================


def fbeta(
    precision:np.ndarray,
    recall:np.ndarray,
    beta:float=1.0
):
    """
    Calcule le score F-beta
    """
    
    fbeta = (1 + beta**2) * (
        (precision * recall) / (beta**2 * precision + recall + 1e-12)
    )
    return fbeta

