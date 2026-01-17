import pandas as pd

# Fonction de comptage pour suivre l'effet des filtrages
def cleaning_counter(**kwargs)->dict:
    return {name:df.shape for name, df in kwargs.items()}


# Liste les colonnes absentent de la dataframe before
def removedAndAdded_col(df_before:pd.DataFrame,df_after:pd.DataFrame)->list:
    
    removed_col_list = df_before.loc[:,~df_before.columns.isin(df_after)].columns.tolist()
    added_col_list = df_after.loc[:,~df_after.columns.isin(df_before)].columns.tolist()
    
    return removed_col_list,added_col_list


# Compte le nombre de ligne et colonnes supprimé entre deux instants d'une dataframe --V2
def cleaning_results(before:dict, after:dict, removed_col=[])->print:
    for df in before.keys():
        rows = before[df][0]-after[df][0]
        cols = len(removed_col)
        if rows <0:
            rows = 0
        elif cols <0:
            cols = 0
        percent_rows = (rows/before[df][0])*100
        percent_cols = (cols/before[df][1])*100
        print(f"{df}: Supression de {rows} lignes soit {percent_rows:.4f}% et {cols} colonnes soit {percent_cols:.4f}% ")




