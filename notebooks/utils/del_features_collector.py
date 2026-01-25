import pandas as pd
from typing import Dict, List, Tuple


# ===========================================================================


def get_column_diff(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Compare les colonnes entre deux DataFrames via les sets.
    
        Returns:
            sorted(removed), sorted(added)
    
    """
    set_before = set(df_before.columns)
    set_after = set(df_after.columns)
    
    removed = list(set_before - set_after)
    added = list(set_after - set_before)
    
    return sorted(removed), sorted(added)


# ===========================================================================


def log_cleaning_step(
    name: str, 
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame
) -> Dict:
    """
    Analyse complète de l'impact d'une étape de nettoyage.
    Retourne un dictionnaire de stats au lieu de simplement imprimer.
    """
    rows_before, cols_before = df_before.shape
    rows_after, cols_after = df_after.shape
    
    removed_cols, added_cols = get_column_diff(df_before, df_after)
    
    rows_removed = max(0, rows_before - rows_after)
    cols_removed = len(removed_cols)
    
    stats = {
        "step_name": name,
        "rows_lost": rows_removed,
        "rows_lost_pct": (rows_removed / rows_before) * 100 if rows_before > 0 else 0,
        "cols_lost": cols_removed,
        "cols_lost_names": removed_cols,
        "cols_added_names": added_cols
    }
    
    # Affichage propre
    print(f"--- [STEP: {name}] ---")
    print(f"Lignes : -{stats['rows_lost']} ({stats['rows_lost_pct']:.2f}%)")
    print(f"Colonnes supprimées : {stats['cols_lost']} {removed_cols if cols_removed < 5 else ''}")
    if added_cols:
        print(f"Colonnes ajoutées : {len(added_cols)} {added_cols if len(added_cols) < 5 else ''}")
        
    return stats