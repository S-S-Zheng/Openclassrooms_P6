import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from typing import List


# ===========================================================================


def get_eta2(df: pd.DataFrame, cat_col: str, target_col: str) -> float:
    """
    Calcule l'Eta-carré (taille de l'effet) pour une ANOVA.
    Utilise la vectorisation Pandas pour plus de rapidité.
    """
    # Moyenne globale
    global_mean = df[target_col].mean()
    
    # Agrégation par groupe : moyenne et effectif
    stats = df.groupby(cat_col)[target_col].agg(['mean', 'count'])
    
    # Somme des carrés interclasses (SCE) (var au sein des groupes)
    sce = (stats['count'] * (stats['mean'] - global_mean)**2).sum()
    
    # Somme des carrés totale (SCT)
    sct = ((df[target_col] - global_mean)**2).sum()
    
    return (
        float(sce / sct)
        if sct != 0
        else 0.0
    )


# ===========================================================================


def get_anova_report(
    df: pd.DataFrame,
    target: str,
    cat_list: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Réalise le test ANOVA pour une liste de variables catégorielles par rapport à une cible numérique.
    Retourne un DataFrame de synthèse.
    """
    results = []
    
    # Nettoyage rapide : on enlève les lignes où la cible est NaN pour éviter les erreurs f_oneway
    df_clean = df.dropna(subset=[target])

    for cat in cat_list:
        # Préparation des groupes pour f_oneway (suppression des NaNs dans la catégorie)
        groups = [group[target] for _, group in df_clean.groupby(cat)]
        
        if len(groups) < 2:
            continue
            
        # Test ANOVA
        f_stat, p_val = f_oneway(*groups)
        
        h0_rejected = p_val < alpha
        eta2 = (
            get_eta2(df_clean, cat, target)
            if h0_rejected
            else np.nan
        )
        
        results.append({
            'feature': cat,
            'f_stat': f_stat,
            'p_value': p_val,
            'h0_rejected': h0_rejected,
            'eta2': eta2
        })
        
    return pd.DataFrame(results).sort_values(by='eta2', ascending=False)