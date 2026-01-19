"""
Permet d'assembler les données importées en un seul DataFrame
"""

# imports 
import pandas as pd
from typing import List
from pathlib import Path

def assemble_data(path:Path, file_names:List[str])-> pd.DataFrame:
    # liste des dataframes + assign crée une colonne subset pour les différencier
    dataframes = [
        pd.read_parquet(path/file_name).assign(subset=file_name) for file_name in file_names
    ]
    return pd.concat(dataframes,axis=0, ignore_index=True)