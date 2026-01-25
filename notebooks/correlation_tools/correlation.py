# imports
import pandas as pd
from typing import Literal, cast,Any


# ===========================================================================


def abs_correlation(
    df: pd.DataFrame,
    method: Literal['pearson', 'kendall', 'spearman']
) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation en valeur absolue.
    Responsabilité : Logique métier / Calcul.
    """
    return abs(df.corr(numeric_only=True, method=method))


# ===========================================================================


def get_correlated_pairs(
    df: pd.DataFrame,
    method: Literal['pearson', 'kendall', 'spearman'],
    vmin: float
) -> pd.DataFrame:
    """
    Calcule la corrélation et extrait les couples dépassant le seuil vmin.
    Responsabilité : Calcul et filtrage des paires.
    """
    # Calcul de la matrice de corrélation absolue
    corr_matrix = abs_correlation(df,method)
    cols = corr_matrix.columns
    
    pairs = []
    # On itère sur la partie triangle supérieur de la matrice (i < j)
    for i, ind1 in enumerate(cols):
        for j, ind2 in enumerate(cols):
            # float(cast(Any)) pour rassurer le lint
            val = float(cast(Any,corr_matrix.loc[ind1, ind2]))
            if i < j and val > vmin:
                pairs.append({
                    "Indicateur_1": ind1,
                    "Indicateur_2": ind2,
                    f"Coeff_corr_{method.capitalize()}": val,
                    "Numéro_indicateur_1": i,
                    "Numéro_indicateur_2": j
                })
    
    return pd.DataFrame(pairs)