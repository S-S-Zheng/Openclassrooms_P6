import pandas as pd
import gc
from typing import List, Union, Literal

from notebooks.utils.features_type_list import features_type

def agg_features(
    df: pd.DataFrame,
    feature_to_groupby,
    prefix:Union[str,None]=None,
    drop_columns: Union[List[str], None] = None
)-> pd.DataFrame:
    """
    Agrège les caractéristiques numériques et catégorielles d'un DataFrame par groupe.

    Cette fonction effectue des agrégations statistiques (moyenne, max, min) sur les 
    colonnes numériques et calcule les fréquences d'apparition (moyenne du One-Hot Encoding) 
    pour les colonnes catégorielles. Elle est optimisée pour limiter l'usage de la RAM.

    Args:
        df (pd.DataFrame): Le DataFrame source à agréger.
        feature_to_groupby (str): Le nom de la colonne pivot pour le groupement (ex: 'SK_ID_CURR').
        prefix (str): Préfixe à ajouter aux noms de colonnes pour éviter les collisions (ex: 'PREV').
        drop_columns (List[str], optional): Liste des colonnes à exclure de l'agrégation. 
            Défaut : None.

    Returns:
        pd.DataFrame: Un DataFrame agrégé avec une ligne par identifiant unique, 
            contenant les statistiques calculées.

    """
    
    # Lists fes features numériques et catégorielles
    num_cols, cat_cols = features_type(df.drop(columns=drop_columns, errors='ignore'))
    
    # Agg numérique
    # dico d'agg
    agg_funcs = {col: ['mean', 'max', 'min'] for col in num_cols}
    # agg suivant la colonne de pivot (feature_to_groupby)
    df_agg = df.groupby(feature_to_groupby).agg(agg_funcs)
    # Renommage des colonnes suivant refix
    if not prefix:
        df_agg.columns = [f'{col[0]}_{col[1].upper()}' for col in df_agg.columns]
    else:
        df_agg.columns = [f'{prefix}_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
    
    # Agg catégorielle
    cat_aggs = []
    for col in cat_cols:
        # On ne sélectionne QUE l'ID et la colonne en cours pour le OHE (léger en RAM)
        temp_ohe = pd.get_dummies(df[[feature_to_groupby, col]], columns=[col], prefix=f'{prefix}_{col}')
        temp_agg = temp_ohe.groupby(feature_to_groupby).mean()
        cat_aggs.append(temp_agg)
    # Merging des agg catégorielles
    if cat_aggs:
        all_cat_df = pd.concat(cat_aggs, axis=1)
        df_agg = df_agg.join(all_cat_df, how='left')

        del cat_aggs, all_cat_df
        gc.collect()
        
    return df_agg


def agg_columns(
    df: pd.DataFrame,
    new_col_name:str,
    cols_to_fuse: List[str],
    operation: Literal['sum','max','mean','min'] = 'sum',
    axis: Literal[0,1]=1
)-> pd.DataFrame:
    """
    Fusionne plusieurs colonnes numériques en une seule par sommation et supprime les originales.

    Args:
        df (pd.DataFrame): Le DataFrame à modifier.
        new_col_name (str): Nom de la nouvelle colonne créée (ex: 'DOCUMENT_COUNT').
        cols_to_fuse (List[str]): Liste des noms de colonnes à additionner.
        operation: type d'opération a appliquer, sum, mean, max, min
        axis = 0 colonne par colonne et 1 line par ligne

    Returns:
        pd.DataFrame: Le DataFrame avec la nouvelle colonne et sans les anciennes.
    """
    
    # Evite les erreurs syntaxiques et liste vides
    existing_cols = [col for col in cols_to_fuse if col in df.columns]
    
    if not existing_cols:
        print('Aucune colonne trouvée ou existante a aggreger')
        return df
    
    df[new_col_name] = df[existing_cols].agg(operation.lower(),axis=axis)
    if new_col_name in df.columns:
        df.drop(columns=existing_cols, inplace=True, errors='ignore')
    
    return df