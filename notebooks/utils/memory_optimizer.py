"""
Réduire la précision technique (Downcasting) pour diviser par deux l'occupation de RAM.
"""

#imports
import pandas as pd
import numpy as np
import gc

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modifie les types pour économiser la RAM.
    """
    for col in df.columns:
        # Erreur dans le diagramme et la donnée d'où la double contrainte
        if ("SK_ID" in col) or ("SK_BUREAU_ID" in col):
            continue 
        
        col_type = df[col].dtype
        
        # Cas des int
        if 'int' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Cas des float
        elif 'float' in str(col_type):
            # On vérifie si on le passer en entier (ex: 1.0, 2.0)
            # Sinon, on le passe en float32
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Pose des soucis donc autant le faire manuellement juste avant entrainement
        # ==> juste avant entrainement == > plus de modif sur la donnée.
        # # Cas des object (conversion en categorie si variance faible)
        # elif col_type == 'object':
        #     if df[col].nunique() / len(df) < 0.5:
        #         df[col] = df[col].astype('category')
        
    return df

def clear_memory():
    """Force la libération de la RAM orpheline."""
    gc.collect()