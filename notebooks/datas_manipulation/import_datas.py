#imports
import pandas as pd
from pathlib import Path


# ===========================================================================


def import_datas(folder_path:Path)->dict:
    """
    Charge uniquement les fichiers Parquet optimisés.
    Retourne un dictionnaire {nom_fichier: dataframe}
    """
    path = Path(folder_path)
    dataframes = {}
    
    # On ne cherche QUE les .parquet maintenant
    for file in path.glob("*.parquet"):
        dataframes[file.stem] = pd.read_parquet(file)
        
    return dataframes