"""
Permet d'assembler les données en un seul DataFrame
"""

# imports 
import pandas as pd
from typing import List, Literal
from pathlib import Path
import gc


# ===========================================================================


def assemble_data(path:Path, file_names:List[str])-> pd.DataFrame:
    """
    Importe une liste des dataframes + assign crée une colonne subset pour les différencier.
    On concat simplement ensuite les df en un seul
    """
    dataframes = [
        pd.read_parquet(path/file_name).assign(subset=file_name) for file_name in file_names
    ]
    return pd.concat(dataframes,axis=0, ignore_index=True)


# ===========================================================================


def merging_data(
    df:pd.DataFrame,
    df_agg:pd.DataFrame,
    on:str,
    how:Literal['left','right','inner','outer','cross']
)-> pd.DataFrame:
    """
    On merge deux df et on supprime de la mémoire le df importer pour le merge.
    
    Args:
        df: dataframe principal
        df_agg: dataframe à merger (pas le df qu'on manipule)
        on: colonne pivot
        how: type de merge: (left, right, inner, outer, cross)
    
    Returns:
        df mergé
    
    """
    
    df = df.merge(df_agg, on=on, how=how)
    
    del df_agg
    gc.collect()
    
    return df