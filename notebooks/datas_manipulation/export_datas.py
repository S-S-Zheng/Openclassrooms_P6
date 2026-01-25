#imports
import pandas as pd
from pathlib import Path


# ===========================================================================


def export_datas(df:pd.DataFrame, folder_path:Path, step:str="", prefix:str="")-> Path:
    """
    Eporte la dataframe vers le chemin spécifié
    
    Args:
        df: DataFrame a exporter
        folder_path: chemin du dossier a destination
        step: nom de l'étape de traitement
        prefix: préfixe du fichier
    """
    path = Path(folder_path/step)
    
    # Création du dossier si inexistant
    path.mkdir(parents=True,exist_ok=True)
    
    df.to_parquet(path/f"{prefix}{step}.parquet")
    
    return path