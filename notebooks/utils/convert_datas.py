#imports
import pandas as pd
from pathlib import Path

from notebooks.utils.encoding_detector import detect_encoding
from notebooks.utils.memory_optimizer import optimize_dtypes, clear_memory


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
        clear_memory()
        return True
    return False