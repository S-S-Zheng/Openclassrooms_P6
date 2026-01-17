#imports
import pandas as pd

def preprocess_contingency_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame en retirant les lignes ou colonnes 'Total' si présentes.
    Responsabilité : Nettoyage de données.
    """
    df_clean = df.copy()
    
    # Vérification robuste pour la ligne 'Total'
    if not df_clean.index.empty:
        last_idx = df_clean.index[-1]
        # On gère le cas où l'index n'est pas une string pure
        if 'Total' in str(last_idx):
            df_clean = df_clean.iloc[:-1, :]

    # Vérification pour la colonne 'Total'
    if not df_clean.columns.empty:
        last_col = df_clean.columns[-1]
        if 'Total' in str(last_col):
            df_clean = df_clean.iloc[:, :-1]
            
    return df_clean