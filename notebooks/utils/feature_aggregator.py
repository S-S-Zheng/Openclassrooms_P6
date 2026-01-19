import pandas as pd
import numpy as np
import gc

from notebooks.utils.features_type_list import features_type

def agg_features(df: pd.DataFrame, feature_to_groupby, prefix):
    # Lists fes features numériques et catégorielles
    num_cols, cat_cols = features_type(df.drop(columns=[feature_to_groupby]))
    
    # Agg numérique
    # dico d'agg
    agg_funcs = {col: ['mean', 'max', 'min'] for col in num_cols}
    # agg suivant la colonne de pivot (feature_to_groupby)
    df_agg = df.groupby(feature_to_groupby).agg(agg_funcs)
    # Renommage des colonnes suivant refix
    df_agg.columns = [f'{prefix}_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
    
    # Agg catégorielle
    for col in cat_cols:
        # On calcule la fréquence des valeurs de chaque modalité d'une colonne cat via OHE
        # OHE avec diff suivant prefix
        temp = pd.get_dummies(df[col], prefix=f'{prefix}_{col}')
        # concat temp avec df[feature_to_groupby] en supposant que l'ordre est conservé
        temp = pd.concat([df[[feature_to_groupby]], temp], axis=1)
        # agg suivant feature_to_groupby et calcul 
        # la moyenne des colonnes OHE devenues binaires (frequence)
        temp_agg = temp.groupby(feature_to_groupby).mean()
        
        # Merge immédiat et suppression
        df_agg = df_agg.merge(temp_agg, on=feature_to_groupby, how='left')
        del temp,temp_agg
        gc.collect()
        
    return df_agg