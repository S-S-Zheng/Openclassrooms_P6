import pandas as pd
import numpy as np
from typing import Tuple

from features_type_list import features_type


def iqr_outliers(
    df: pd.DataFrame, 
    iqr_coeff: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Détecte les outliers via la méthode de l'Interquartile Range (IQR).
    Vectorise le calcul sur toutes les colonnes numériques simultanément.
    """
    
    # Sélection des colonnes numériques
    num_list, _ = features_type(df)
    df_num = df[num_list]
    
    if df_num.empty:
        return pd.DataFrame(columns=["feature", "nb_outliers"]), df.iloc[0:0]

    # Calcul vectorisé des quartiles (on calcule tout d'un coup)
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1

    lower_bounds = Q1 - iqr_coeff * IQR
    upper_bounds = Q3 + iqr_coeff * IQR

    # DataFrame de masque booléen (True si outlier)
    # L'alignement se fait automatiquement sur les noms de colonnes
    # car pandas voit que lower/upper_bounds sont des Series ==> cf operation SIMD
    # pour Single instruction, Multiple Data
    outliers_mask = (df_num < lower_bounds) | (df_num > upper_bounds)

    # 4. Construction du résumé (Summary)
    summary = (
        outliers_mask.sum()
        .reset_index()
        .rename(columns={
            "index": "feature",
            0: "nb_outliers"
        })
    )
    # On ne garde que les features qui ont au moins un outlier
    summary = (
        summary[summary["nb_outliers"] > 0]
        .sort_values(by="nb_outliers", ascending=False)
    )

    # Extraction des lignes (Outlier si au moins une colonne est True sur la ligne)
    outlier_rows_indices = outliers_mask.any(axis=1)
    df_outliers = df[outlier_rows_indices]

    return summary, df_outliers


def zmad_outliers(
    df: pd.DataFrame, 
    zmad_coeff: float = 3.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Détecte les outliers via le Modified Z-score basé sur la Median Absolute Deviation (MAD).
    Approche vectorisée pour traiter toutes les colonnes simultanément.
    """
    
    # Sélection des colonnes numériques
    num_list, _ = features_type(df)
    df_num = df[num_list]
    
    if df_num.empty:
        return pd.DataFrame(columns=["feature", "nb_outliers"]), df.iloc[0:0]

    # Calcul des statistiques de base
    median = df_num.median()
    
    # Calcul de la MAD : médiane de l'écart absolu à la médiane
    # (df_num - median) aligne automatiquement les colonnes (pareil que dans IQR
    # pandas reprère automatique que median est une Series)
    mad = (df_num - median).abs().median()

    # Calcul du Z-score modifié
    # Gestion du cas MAD = 0 (quand plus de 50% des valeurs sont identiques)
    # On remplace 0 par NaN pour éviter l'inf de la division, puis on remplit par 0
    safe_mad = mad.replace(0, np.nan)
    
    # Formule vectorisée : 0.6745 * (x - median) / MAD
    z_scores_abs = (0.6745 * (df_num - median) / safe_mad).abs()

    # Identification des outliers
    outliers_mask = z_scores_abs > zmad_coeff
    # Si MAD était 0 et que la valeur != médiane, c'est techniquement un outlier
    # On gère les NaNs générés par safe_mad si nécessaire
    outliers_mask = outliers_mask.fillna(False)

    # rapport
    summary = (
        outliers_mask.sum()
        .reset_index()
        .rename(columns={
            "index": "feature",
            0: "nb_outliers"
        })
    )
    summary = (
        summary[summary["nb_outliers"] > 0]
        .sort_values(by="nb_outliers", ascending=False)
    )

    # Extraction des lignes
    df_outliers = df[outliers_mask.any(axis=1)]

    return summary, df_outliers