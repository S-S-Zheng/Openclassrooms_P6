import pandas as pd

# Fonction pour lister les features numériques et les catégorielles
def features_types(data:pd.DataFrame)->list:
    '''
    Créée deux listes à partir d'une dataframe:
    numeric_list: liste des indicateurs à valeur numérique
    non_numeric_list: liste des indicateurs non numériques
    '''

    numeric_list = data.select_dtypes(include=['int64', 'float64','int32', 'float32']).columns.tolist()
    cat_list = data.select_dtypes(exclude=['int64', 'float64','int32', 'float32']).columns.tolist()

    return numeric_list,cat_list

######################################################################################
######################################################################################
######################################################################################

# Fonction pour calculer les proportion de valeurs renseignées par feature
def data_props(df:pd.DataFrame, ind_name:str, numeric_ind_list:list, ind_sort_order:str)->pd.DataFrame:
    '''
    Retourne 2 dataframes listant les proportions de valeurs renseignées. data_props prend en argument \n 
    une dataframe df et requiert le nom de la colonne de l'indicateur, ind_name ainsi qu'une liste d'indicateurs numériques.

    ENTREES:
    
    propsInd_data: Calcul la proportion de ind_name avec des valeurs renseignées par feature\n
    df_ind_name: Agrege le nombre de valeurs non nulle présente pour ind_name suivant chaque features de numeric_ind_list \n 
    ind_sort_order: feature de référence pour le sorting de df_ind_name\n
    
    
    SORTIES:
    propsInd_data: Proportion de valeurs renseignées par groupe d'observations appartenant la feature indiquée par ind_name
    df_ind_name: Nombres de valeurs renseignées pour chaque groupe d'observation suivant les features présentes \n 
    dans la liste des features numeric_ind_list 
    '''

    if ind_name not in df.columns:
        raise ValueError(f"{ind_name} n'existe pas dans la DataFrame.")

    numeric_cols = [col for col in numeric_ind_list if col in df.columns]
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique valide fournie.")

    #############################################################################
    # 1) Proportion de valeurs renseignées dans ind_name par numeric_cols
    #############################################################################

    # Table pivotée = pour chaque ind_name : moyenne de non-null par colonne
    indicator_availability = df[numeric_cols].notna().astype(int).groupby(df[ind_name]).mean()

    # Moyenne globale (toutes colonnes) ⇒ % de complétion
    propsInd_data = indicator_availability.mean(axis=1) * 100
    propsInd_data = propsInd_data.rename("props_per_indicator(%)").reset_index()

    # Suppression des lignes entièrement vides (0%)
    propsInd_data = propsInd_data[propsInd_data["props_per_indicator(%)"] > 0]

    # Tri décroissant
    propsInd_data = propsInd_data.sort_values(
        by="props_per_indicator(%)",
        ascending=False,
        ignore_index=True
    )

    ##############################################################################
    # 2) Nombre de valeurs renseignées dans numeric_cols par ind_name
    ##############################################################################

    df_ind_name = df.groupby(ind_name)[numeric_cols].count()

    # Suppression des lignes entièrement vides
    df_ind_name = df_ind_name[(df_ind_name != 0).any(axis=1)]

    # Tri selon ind_sort_order
    if ind_sort_order not in df_ind_name.columns:
        raise ValueError(f"{ind_sort_order} n'est pas dans les colonnes numériques.")
    df_ind_name = df_ind_name.sort_values(by=ind_sort_order, ascending=False)

    return propsInd_data, df_ind_name

######################################################################################
######################################################################################
######################################################################################

# Réalise un classement issue d'une moyenne pondérée
def top_score(data:pd.DataFrame,ind_pond:list,weight_pond:list,ind_ref:str,cols = list,topN=5):
    '''
    Fournit un classement général issue d'une pondération des paramètres issus de ind_pond et un classement absolu

    ENTREES:
    
    data: Dataframe \n
    ind_pond: Liste des features \n 
    weight_pond: Pondération associée aux features de la forme [poids,sens]\n
    ind_ref: nom feature de référence pour distinguer les observations. en général la colonne modèle.
    cols: features complémentaires a représenter
    topN: limitation du classement absolu. Défaut top 5
    
    SORTIES:
    
    top_param
    top_model
    '''
    # ind_pond = [i for i in data.columns if '_test' in i]
    
    # # pondération [poids, sens]
    # weight_pond = [
    #     [1,-1], # MAE ==> on veut petit donc sens = -1
    #     [1,-1], # MSE idem
    #     [1,1], # R2 => on le veut le plus grand possible (tend vers 1)
    # ]

    #dictionnaire { nom variable: pondération associée}
    ind_weight = dict(zip(ind_pond,weight_pond))
    
    # Copie de la df
    rank_df = data.copy()
    
    for ind,sens in ind_weight.items():
            # Calcul des rang des lignes ( rang 1 = premier),NaN = dernier
        if sens[1] >= 0:
            rank_df.loc[:,ind] = data.loc[:,ind].rank(ascending=False,method='min',na_option='bottom')
            # Calcul des rang ( rang 1 = dernier), NaN = rang 1
        else:
            rank_df.loc[:,ind] = data.loc[:,ind].rank(ascending=True,method='min',na_option='top')

    
    # Moyenne pondérée: (\sum w_i*X_i)/(\sum w_i) avec w_i les poids associés et X_i les indicateurs
    rank_df.loc[:,'score'] = sum(
        rank_df.loc[:,ind]*weight[0] for ind,weight in ind_weight.items()
    )/sum(
        weight[0] for weight in ind_weight.values()
    )

    ###############################################################
    # 1) Classement suivant les paramètres
    ###############################################################
    selected_cols = [ind_ref, *cols, "score"]

    missing = set(selected_cols) - set(rank_df.columns)
    if missing:
        raise KeyError(f"Colonnes absentes du DataFrame : {missing}")
        
    top_param = (
        rank_df
        .sort_values('score')
        .reset_index(drop=True)
        .loc[:, selected_cols]
    )

    ################################################################
    # 2) Classement absolu
    ################################################################
    top_model = top_param.groupby(ind_ref)['score'].mean().sort_values()

    return top_param, top_model

######################################################################################
######################################################################################
######################################################################################

def iqr_outliers(datas:pd.DataFrame, iqr_coeff = 1.5):
    """
    Calcul interquartile (IQR) afin de débusquer les outliers potentiels

    ENTREES:
    
    datas : données\n
    iqr_coeff: coeff pour jauger la longueur des moustaches (ref: 1.5 de Tukey) \n

    SORTIES:
    
    summary : Nombre d'outliers par variable numérique
    df_outliers : Toutes les lignes considérées comme outliers
    """

    df_num = datas.select_dtypes(include=['int64', 'float64','int32', 'float32'])
    outlier_index = {}

    for col in df_num.columns:
        Q1 = df_num[col].quantile(0.25)
        Q3 = df_num[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - iqr_coeff * IQR
        upper = Q3 + iqr_coeff * IQR

        mask = (df_num[col] < lower) | (df_num[col] > upper)
        outlier_index[col] = df_num[mask].index.tolist()

    summary = pd.DataFrame({
        "feature": list(outlier_index.keys()),
        "nb_outliers": [len(v) for v in outlier_index.values()]
    })

    all_idx = sorted(set().union(*outlier_index.values()))
    df_outliers_iqr = datas.loc[all_idx]
    
    return summary, df_outliers_iqr


######################################################################################
######################################################################################
######################################################################################

def zmad_outliers(datas:pd.DataFrame,zmad_coeff=3.5):
    """
    Calcul zscore modifié (MAD) afin de débusquer les outliers potentiels

    ENTREES:
    
    datas : données\n
    zmad_coeff: coeff pour MAD z-score\n

    SORTIES:
    
    z_mad: z-score MAD
    df_outliers : Toutes les lignes considérées comme outliers
    """

    df_num = datas.select_dtypes(include=['int64', 'float64','int32', 'float32'])
    outlier_index = {}
    # Médiane de chaque feature numérique
    med = df_num.median()
    
    # MAD: médiane de la valeur absolu de la différence entre la valeur de la feature et la médiane de la feature
    mad = (df_num - med).abs().median()

    for col in df_num.columns:
        z_mad = 0.6745 * (df_num[col] - med[col]) / mad[col]#.replace(0, np.nan)
        mask = z_mad.abs() > zmad_coeff    
        outlier_index[col] = df_num.index[mask].tolist()

    summary = pd.DataFrame({
        "feature": list(outlier_index.keys()),
        "nb_outliers": [len(v) for v in outlier_index.values()]
    })
    
    all_idx = sorted(set().union(*outlier_index.values()))
    df_outliers_zmad = datas.loc[all_idx]

    return summary, df_outliers_zmad

######################################################################################
######################################################################################
######################################################################################

def model_attr(models:dict):
    '''
    Permet de vérifier si le(s) modèles possèdent les attributs demandées pour afficher les métriques.(Cas ici sur de la classification) 
    '''
    for name, model in models_full.items():
        print(
            name,
            hasattr(model, "predict_proba"),
            hasattr(model, "decision_function")
        )