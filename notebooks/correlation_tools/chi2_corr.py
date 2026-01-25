import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations
from typing import List, Union


# ===========================================================================


def cramersV(contingency_table:pd.DataFrame, chi2_stat:float)->Union[int,float]:
    """Calcule le coefficient V de Cramer pour l'intensité de l'association."""
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape)
    if n == 0 or k <= 1:
        return 0
    return np.sqrt(chi2_stat / (n * (k - 1)))


# ===========================================================================


def get_chi2_report(df: pd.DataFrame, cat_list: List[str], alpha:float=0.05)->pd.DataFrame:
    """
    Réalise les tests du chi2 sur toutes les combinaisons de la liste.
    Retourne un DataFrame structuré au lieu de faire des prints.
    """
    results = []
    
    # Remplacement des boucles manuelles par combinations (iterable, groupe de combinaison)
    for col1, col2 in combinations(cat_list, 2):
        table = pd.crosstab(df[col1], df[col2])
        chi2_stat, p_val, dof, _ = chi2_contingency(table)
        
        # On doit mettre type: ignore car sinon pylance retourne une erreur 
        # car il est persuadé que le format est mauvais
        h0_rejected = p_val < alpha # type: ignore
        
        v_cramer = (
            cramersV(table, chi2_stat) # type: ignore
            if h0_rejected 
            else np.nan
        )
        
        results.append({
            'var1': col1,
            'var2': col2,
            'f_stat': chi2_stat,
            'p_value': p_val,
            'dof': dof,
            'h0_rejected': h0_rejected,
            'v_cramer': v_cramer
        })
        
    return pd.DataFrame(results)