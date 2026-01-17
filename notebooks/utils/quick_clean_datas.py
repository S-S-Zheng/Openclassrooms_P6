"Regroupe les fonctions de nettoyage rapide des fichiers de données"

# imports
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from typing import List, Tuple



def remove_duplicates(df:pd.DataFrame)->Tuple[pd.DataFrame, int]:
    """Supprime les lignes doublons parfait."""
    initial_shape = len(df)
    df = df.drop_duplicates()
    final_shape = len(df)
    dropped_rows = initial_shape - final_shape
    return df, dropped_rows


# Fonction spécifique a cette donnée: les XNA, XAP = Not available et not applicable
def replace_XNA_XAP_by_NaN(df:pd.DataFrame)->pd.DataFrame:
    """Remplace les valeurs XNA et XAP par des NaN dans les colonnes spécifiées."""
    text_codes = ["XNA", "XAP", "Unknown"]
    num_codes =[365243, 999999999]
    
    # Isole les colonnes par type
    cols_text = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    
    replace_map = {}
    
    # Mapping
    for col in cols_text:
        replace_map[col] = {code: np.nan for code in text_codes}
    for col in cols_num:
        replace_map[col] = {code: np.nan for code in num_codes}
    
    # Conversion groupée pour éviter les Warnings
    if cols_text:
        df[cols_text] = df[cols_text].astype(object)
    
    df = df.replace(replace_map)
    
    return df


def drop_empty_columns(df:pd.DataFrame, threshold:float=0.9)->Tuple[pd.DataFrame, List[str]]:
    """
    threshold = seuil de suppression (défaut = 90% donc si plus de 90% de Nan ==> suppr).
    """
    # Spécifique au projet: remplacement code speciaux par nan
    df = replace_XNA_XAP_by_NaN(df)
    
    # liste des colonnes qui vont être suppr
    cols_to_drop = df.columns[df.isna().mean() > threshold].tolist()
    
    if cols_to_drop:
        # thresh = sueil complémentaire a threshold
        keep_limit = int(len(df) * (1 - threshold))
        df = df.dropna(thresh=keep_limit, axis=1)
        
    return df, cols_to_drop


def drop_empty_rows(df:pd.DataFrame, threshold:float=0.9)->Tuple[pd.DataFrame, List[str]]:
    """
    threshold = seuil de suppression (défaut = 90% donc si plus de 90% de Nan ==> suppr).
    """
    # Spécifique au projet: remplacement code speciaux par nan
    df = replace_XNA_XAP_by_NaN(df)
    
    # liste des lignes qui vont être suppr
    rows_to_drop = df.index[df.isna().mean(axis=1) > threshold].tolist()
    
    if rows_to_drop:
        # thresh = sueil complémentaire a threshold
        keep_limit = int(df.shape[1] * (1 - threshold))
        df = df.dropna(thresh=keep_limit, axis=0)
        
    return df, rows_to_drop


def drop_col_with_unique_value(df:pd.DataFrame)->Tuple[pd.DataFrame, List[str]]:
    """
    Supprime les colonnes a variance nulle (valeur unique)
    """
    null_var=[col for col in df.columns if df[col].nunique()<=1]
    df = df.drop(columns=null_var)
    return df, null_var


def clean_infinites(df):
    """Remplace les +inf et -inf par NaN."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df