"""
Réduire la précision technique (Downcasting) pour diviser par deux l'occupation de RAM.
"""

#imports
import pandas as pd
import numpy as np
from typing import Tuple


# ===========================================================================


def optimize_dtypes(df: pd.DataFrame, categorization: bool = False) -> pd.DataFrame:
    """
    Modifie les types pour économiser la RAM.
    """
    # Identifier les colonnes SK_ID une seule fois
    sk_id_cols = [col for col in df.columns if "SK_ID" in col]
    cols_to_optimize = [col for col in df.columns if col not in sk_id_cols]
    
    int_list = df[cols_to_optimize].select_dtypes(include=["int"]).columns
    float_list = df[cols_to_optimize].select_dtypes(include=["float"]).columns
    cat_list = df[cols_to_optimize].select_dtypes(exclude=[np.number]).columns
    
    # Traiter les colonnes numériques en séparant int et float
    for col in int_list:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in float_list:
        df[col] = df[col].astype('float32')
    
    # Traiter les catégories si activé
    if categorization:
        for col in cat_list:
            # SEULEMENT si variance faible (cardinalité faible) et pas booleen
            if (df[col].nunique() / len(df) < 0.05
                and df[col].dtype != bool
            ):
                df[col] = df[col].astype('category')
    
    return df
    # for col in df.columns:
    #     if ("SK_ID" in col):
    #         continue 
        
    #     col_type = df[col].dtype
        
    #     # On tente de downcast les types pour éco mémoire.
    #     # Cas des int
    #     if 'int' in str(col_type):
    #         df[col] = pd.to_numeric(df[col], downcast='integer')
            
    #     # Cas des float
    #     elif 'float' in str(col_type):
    #         df[col] = pd.to_numeric(df[col], downcast='float')
        
    #     # Pose des soucis car ultra rigide (erreur si la creation/modif etc implique
    #     # une valeur non catégorisée à ce moment là), ex avec 'UNKNOWN':
    #     # df['existing_col']=(df['existing_col'].cat.add_categories('UNKNOWN').fillna('UNKNOWN'))
    #     # problème aussi en merging, feature engineering...
    #     # # Cas des object (conversion en categorie si variance faible)
    #     if categorization:
            
    #         if (col_type == 'object' 
    #         and df[col].nunique() / len(df) < 0.05
    #         ):
    #             df[col] = df[col].astype('category')
            
    # return df


# ===========================================================================


def log_metrics(df:pd.DataFrame, stage:str="")->Tuple[str, float, float]:
    "Calcul et retourne les métriques mémoires"
    mem = df.memory_usage().sum() / 1024**2 # RAM(MB)
    nulls = df.isnull().sum().sum() # Total Nan
    return stage, mem, nulls


# ===========================================================================


def float_to_int(df:pd.DataFrame, with_nan:bool = True)->pd.DataFrame:
    """
    Transforme les floats en int en s'assurant qu'il y a 0 perte d'information
    
        Args:
            df: Dataframe
            with_nan: convertie en Int (True, supporte les Nan) ou en int(False)
    """
    df_floats = df.select_dtypes(include='float').columns
    for col in df_floats:
        
        series_temp = df[col]
        
        if not (series_temp.dropna() % 1 == 0).all():
            continue
        
        if with_nan:
            series_temp = series_temp.astype("Int64")
        else:
            series_temp = series_temp.dropna().astype(np.int64)
        
        df[col] = pd.to_numeric(series_temp,downcast="integer")
        
    return df