import pandas as pd
from typing import List, Tuple


# ===========================================================================


def get_data_completion_report(
    df: pd.DataFrame, 
    group_col: str, 
    feature_list: List[str], 
    sort_by_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse la complétion des données (valeurs non-nulles) par groupe.
    
    Args:
        df: Le DataFrame à analyser.
        group_col: La colonne utilisée pour le regroupement (ex: 'SK_ID_CURR').
        feature_list: Liste des colonnes numériques à inspecter.
        sort_by_col: Colonne de feature_list utilisée pour trier les résultats.
        
    Returns:
        - completion_pct: % moyen de remplissage global par groupe d'indicateur.
        - completion_counts: Nombre exact de valeurs renseignées par feature et par groupe.
    """
    
    # VALIDATION PRÉVENTIVE (Fail Fast)
    missing_cols = []
    if group_col not in df.columns:
        missing_cols.append(group_col)
    
    # On vérifie la présence des features ET de la colonne de tri
    required_features = set(feature_list) | {sort_by_col}
    for col in required_features:
        if col not in df.columns:
            missing_cols.append(col)
            
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")

    # CALCULS
    # Groupement unique
    grouped = df.groupby(group_col)[feature_list]
    
    # Nombre de valeurs renseignées
    completion_counts = grouped.count()
    
    # On élimine les groupes totalement vides
    completion_counts = completion_counts[completion_counts.any(axis=1)]

    # Calcul des proportions (%) 
    # .div(axis=0) est très performant car vectorisé
    group_sizes = grouped.size().loc[completion_counts.index] 
    completion_ratios = completion_counts.div(group_sizes, axis=0)
    
    # Score global de complétion par groupe (moyenne des ratios 
    # des colonnes sélectionnées) et on trie
    completion_pct = (
        (completion_ratios.mean(axis=1) * 100)
        .to_frame(name="completion_rate_pct")
        .sort_values(by="completion_rate_pct", ascending=False)
        .reset_index()
    )
    
    # Tri final des comptes
    completion_counts = completion_counts.sort_values(by=sort_by_col, ascending=False)

    return completion_pct, completion_counts