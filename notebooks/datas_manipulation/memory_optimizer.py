"""
Réduire la précision technique (Downcasting) pour diviser par deux l'occupation de RAM.
"""

#imports
import pandas as pd
from typing import Tuple


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modifie les types pour économiser la RAM.
    """
    for col in df.columns:
        if ("SK_ID" in col):
            continue 
        
        col_type = df[col].dtype
        
        # On tente de downcast les types pour éco mémoire.
        # Cas des int
        if 'int' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Cas des float
        elif 'float' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Pose des soucis car ultra rigide (erreur si la creation/modif etc implique
        # une valeur non catégorisée à ce moment là), ex avec 'UNKNOWN':
        # df['existing_col']=(df['existing_col'].cat.add_categories('UNKNOWN').fillna('UNKNOWN'))
        # problème aussi en merging, feature engineering...
        # # Cas des object (conversion en categorie si variance faible)
        # elif (col_type == 'object' 
        # and df[col].nunique() / len(df) < 0.05
        # ):
        # df[col] = df[col].astype('category')
        
    return df

def log_metrics(df:pd.DataFrame, stage:str="")->Tuple[str, float, float]:
    "Calcul et retourne les métriques mémoires"
    mem = df.memory_usage().sum() / 1024**2 # RAM(MB)
    nulls = df.isnull().sum().sum() # Total Nan
    return stage, mem, nulls