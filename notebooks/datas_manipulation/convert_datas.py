#imports
import pandas as pd
from pathlib import Path
import gc

from notebooks.datas_manipulation.encoding_detector import detect_encoding
from notebooks.datas_manipulation.memory_optimizer import optimize_dtypes


# ===========================================================================


def convert_csv_to_parquet(file_path: Path):
    """Convertit un fichier unique en optimisant sa taille."""
    parquet_path = file_path.with_suffix('.parquet')
    
    if not parquet_path.exists():
        # Lecture
        enc = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=enc, low_memory=False)
        
        # Réduction de la précision (Gain RAM)
        df = optimize_dtypes(df)
        
        # Sauvegarde
        df.to_parquet(parquet_path, index=False)
        
        # Nettoyage immédiat
        del df
        gc.collect()
        return True
    return False