import chardet
from pathlib import Path

def detect_encoding(file_path: Path, n_lines: int = 10000):
    """
    Analyse les premiers octets d'un fichier pour deviner son encodage.
    n_lines: nombre d'octets à lire pour l'analyse (plus c'est élevé, plus c'est fiable).
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(n_lines)
    
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    
    print(f"Fichier : {file_path.name} | Encodage détecté : {encoding} (Confiance : {confidence:.2%})")
    return encoding