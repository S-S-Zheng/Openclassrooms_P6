import pandas as pd
import numpy as np

#Selection
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV, 
    cross_validate,
    StratifiedShuffleSplit,
    cross_val_predict,
    KFold
)
# Metrics
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, 
    precision_score, recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
)
# mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

#Preprocess
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer, RobustScaler, PowerTransformer,LabelBinarizer
# from sklearn.pipeline import Pipeline#, make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline # Remplace la pipeline de sklearn, écéssaire car SMOTE n'a pas de fit_transform mais fit_resample()
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# from misc import features_types

def Xy_tf(
    scaler_tf=None,
    logX_tf=False,
    logy_tf=False,
    log_method = "Box-Cox",
    y_tf=False
):

    '''
    Initie les transformations à apporter ux prédicteurs et cible.

    ENTREES:

    scaler_tf: La standardisation des variables numériques\n
    logX_tf: Réaliser ou non une transformation log sur X\n
    logy_tf: Réaliser ou non une transformation log sur y\n
    log_method: la méthode log a employer sur X et y. Défaut: "Box-Cox"\n
    y_tf: Réaliser une transformation sur y ou non.

    SORTIES:
    num_tf: Pipeline des tf sur X
    y_tf: Pipeline des tf sur y
    '''
    # # Transformation des variables numériques (log + standardisation)
    #=================================================================================
    # # Adaptapteur pour appliquer la fonction Numpy log. Si log_tf=False, cette transformation n'est pas appliquée 
    # logarithm = FunctionTransformer(np.log, inverse_func = np.exp, validate=False) if log_tf else 'passthrough'
    logarithmX = PowerTransformer(method=log_method) if logX_tf else 'passthrough' # Fonctionne normalement avec get_names_out de feature_importance
    logarithmy = PowerTransformer(method=log_method) if logy_tf else 'passthrough'
    
    # Transformation sur X
    num_tf = Pipeline([
        ('log', logarithmX),
        ('scaler', scaler_tf)
    ])

    # Transformation sur y
    # Transformation de y: on log puis standardise
    y_tf = Pipeline([
        ('log', logarithmy),
        ('scaler', scaler_tf)
    ]) if y_tf else None #Si pas de transformation a apportée a y

    return num_tf,y_tf
    
######################################################################################
######################################################################################
######################################################################################

def preproc(
    numeric_list:list,
    num_tf,
    cat_list = None,
    cat_list_splitted = None,
    encoding={
        'ohe':OneHotEncoder(handle_unknown='ignore'),
        'label':LabelEncoder(),
        'binary':LabelBinarizer()
    },
    sparse_output = 'both',
    dense_encoding = {
        'ohe':OneHotEncoder(handle_unknown='ignore',sparse_output=True),
    }
):
    
    '''
    Réalise le pré-traitement.

    ENTREES:
    
    numeric_list: Liste des features numériques
    num_tf: La pipeline de transformation des features numériques de X
    cat_list = Liste des features catégorielles par défaut. défaut=None
    cat_list_splitted = Dictionnaire {'cat_typeEncodage':[liste]}. typeEncodage peut etre ohe, label, binary\n
    Alternative à la cat_list si on souhaite encoder de différentes manières les features catégorielles. défaut= None
    encoding: Encodage des features catégorielles à réaliser\n Défaut ={
        'ohe':OneHotEncoder(handle_unknown='ignore'),
        'label':LabelEncoder(),
        'binary':LabelBinarizer()
    },
    sparse_output: Matrices de préprocessing, sparse = creuse, dense = dense, both = les deux. Défaut = 'both',
    dense_encoding: Encodage en matrice dense. Défaut = {
        'ohe':OneHotEncoder(handle_unknown='ignore',sparse_output=True),
    }\n

    SORTIES:

    ct_sparse_full: Préproc en matrice creuse
    ct_dense_full: Préproc en matrice dense
    '''
    
    if sparse_output in ['both','sparse']:
        # Mise en place des préprocesseurs en sparse et dense
        transformers_sparse = []
        # Matrice creuse
        transformers_sparse.append(('num', num_tf, numeric_list))
        if cat_list_splitted:
            for cat_name,cat_listing in cat_list_splitted:
                encoding_key = cat_name.split('_')[1]
                transformers_sparse.append((cat_name, encoding[encoding_key], cat_listing))
        else:
            transformers_sparse.append(('cat', next(iter(encoding.values())), cat_list))

        ct_sparse_full = ColumnTransformer(transformers_sparse)
    else:
        ct_sparse_full= []
    
    if sparse_output in ['both','dense']:
        transformers_dense = []
        # Matrice pleine
        transformers_dense.append(('num', num_tf, numeric_list))
        if cat_list_splitted:
            for cat_name,cat_listing in cat_list_splitted:
                encoding_key = cat_name.split('_')[1]
                transformers_dense.append((cat_name, dense_encoding[encoding_key], cat_listing))
        else:
            transformers_dense.append(('cat', next(iter(dense_encoding.values())), cat_list))

        ct_dense_full = ColumnTransformer(transformers_dense)
    else:
        ct_dense_full=[]
        
    return ct_sparse_full,ct_dense_full
    
######################################################################################
######################################################################################
######################################################################################

def Xy_folds(
    X:pd.DataFrame,
    y:pd.Series,
    n_splits:int = 5,
    shuffle: bool = True,
    random_state:int = 42,
    method: str = 'kfold',
    test_size: float = None,
    train_size: float = None,
):
    '''
    Renvoie les folds trains/tests de X et y suivant la méthode employée (notamment utilisés dans la validation croisée).

    ENTREES:
    
    n_splits:int = 5, Nombre de plis \n
    shuffle: bool = True, Mélanger ou non le paquet avant de split.
    random_state:int = 42, A définir absolument si shuffle est True et qu'on souhaite retrouver les splits d'une validation croisée (même random_state)\n
    method: str = 'kfold', méthode de mélange. Alternatives possible: 'strat' pour le Shuffle stratifié.
    test_size: float = None, Proportion du test. Si None, la valeur prendra automatiquement le complémentaire du train ET si train aussi None alors test_size = 0.1 et train_size = 0.9
    train_size: float = None, Proportion du train. Si None, la valeur prendra automatiquement le complémentaire du test
    
    SORTIES:

    results_full: Dataframe contenant les résultats des plis.
    
    '''

    results=[]

    methods = {
        'kfold': KFold(n_splits = n_splits, shuffle = shuffle, random_state= random_state),
        'strat': StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size, train_size = train_size, random_state= random_state),
    }

    for n, (train_index, test_index) in enumerate(methods[method].split(X,y)):
        X_train,X_test,y_train,y_test = X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]

        results.append(
            {
                'splitter':methods[method],
                'fold': n,
                'X_train':X_train,
                'X_test':X_test,
                'y_train':y_train,
                'y_test':y_test,
            }
        )

    results_full = pd.DataFrame(results)

    return results_full

######################################################################################
######################################################################################
######################################################################################

def add_pipes(model,ct,**kwargs):

    '''
    Rend la pipeline pipe_full dynamique afin de pouvoir y incorporer des étapes supplémentaires.

    ENTREES:

    model: Modèle 
    ct: Pré-traitement
    **kwargs: Etapes de traitement supplémentaires exemple over/undersampling

    SORTIES:

    Pipeline(pipeling): La pipeline finale
    '''

    if kwargs:
        pipeling = [('preproc',ct)]

        for key, value in kwargs.items():
            pipeling.append((key,value))
        pipeling.append(('model',model))
    else:
        pipeling = [
            ('preproc',ct),
            ('model',model)
        ]
    return Pipeline(pipeling)
######################################################################################
######################################################################################
######################################################################################

def reg_modeling_cv(
    X:pd.DataFrame,
    y:pd.Series,
    numeric_list:list,
    cat_list:list,
    models_full : dict,
    log_tf = False,
    scaler_tf = None,
    cat_tf = None,
    scoring :dict = {
        'MAE':'neg_mean_absolute_error',
        'MSE':'neg_mean_squared_error',
        'R2':'r2',
    },
    random_state = 42,
    cv=5,
    show=False
):
    '''
    Fonction pour entrainer et évaluer les performances de plusieurs modèles de REGRESSION avec la méthode de la validation croisée.
    
    ENTREES:

    X : Liste des colonnes à supprimer de X avant modélisation.\n
    y : Nom de la variable cible.\n
    models_full: Dictionnaire des modèles à étudier\n
    log_tf: Transformation log1p à réaliser ou non.\n
    scaler_tf: Standardisation. Par défaut: StandardScaler(). Les alternatives sont:
    [
    MinMaxScaler() : borne juste sur [0,1]. Tres sensibles aux outliers
    RobustScaler() : utilise la mediane et l'iqr donc insensibles aux outliers mais l'echelle est conservée (mauvais pour SVR par exemple?)
    QuantileTransformer() : Transforme les données pour suivre une distribution donnée ce qui supprime les outliers et ramene l'echelle 
                            MAIS transformation non-linéaire + transformation couteuse!
    ]
    cat_tf: Transformation sur les catégories.
    scoring : les métriques (regression) choisies. Par défaut: ('neg_mean_absolute_error','neg_mean_squared_error','r2')
    {
        'ExplainedVar':'explained_variance',
        'MaxError':'neg_max_error',
        'MAE':'neg_mean_absolute_error',
        'MSE':'neg_mean_squared_error',
        'RMSE':'neg_root_mean_squared_error',
        'MSlogE':'neg_mean_squared_log_error',
        'RMSlogE':'neg_root_mean_squared_log_error',
        'MedianAE':'neg_median_absolute_error',
        'R2':'r2',
        'MPD':'neg_mean_poisson_deviance',
        'MGD':'neg_mean_gamma_deviance',
        'MAPE':'neg_mean_absolute_percentage_error',
        'd2AE':'d2_absolute_error_score'
    }\n

    random_state : par défaut = 42. \n
    cv = par défaut = 5\n
    
    SORTIES:
    
    ct_sparse_full: Préprocesseur avec matrice creuse\n
    ct_dense_full: Préprocesseur avec matrice dense\n
    results_full: Dataframe des résultats\n
    show: Affiche les résultats. Par défaut False)
    '''
###########################################################################
    # # X et y
    # X = data.drop(columns=X_drop)
    # y = data[target]
    
###########################################################################   
    
    # # Listes de variables numériques et catégorielles.
    # numeric_list,cat_list = features_types(data)

###########################################################################    

    # # Transformation des variables numériques (log + standardisation)
    #=================================================================================
    # # Adaptapteur pour appliquer la fonction Numpy log. Si log_tf=False, cette transformation n'est pas appliquée 
    # logarithm = FunctionTransformer(np.log, inverse_func = np.exp, validate=False) if log_tf else 'passthrough'
    logarithm = PowerTransformer(method="Box-Cox") if log_tf else 'passthrough' # Fonctionne normalement avec get_names_out de feature_importance
    
    # Transformation sur X
    num_tf = Pipeline([
        ('log', logarithm),
        ('scaler', scaler_tf)
    ])

    # Transformation sur y
    # Transformation de y: on log puis standardise
    y_tf = Pipeline([
        ('log', logarithm),
        ('scaler', scaler_tf)
    ])


###########################################################################
    
    # Mise en place des préprocesseurs en sparse et dense + injection de FunctionTransformer afin qu'il reste dans la pipeline
    transformers_sparse = []
    transformers_dense = []

    if numeric_list:
        transformers_sparse.append(('num', num_tf, numeric_list))
        transformers_dense.append(('num', num_tf, numeric_list))
    if cat_list:
        transformers_sparse.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_list))
        transformers_dense.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_list))

    ct_sparse_full = ColumnTransformer(transformers_sparse)
    ct_dense_full = ColumnTransformer(transformers_dense)
    
    # MEMO: Permet à la fonction d'éviter de planter si la liste numérique ou catégorielle est vide
    # MEMO: hasattr(model,'sparse_input_') permet de savoir si le modele peut accepter les matrices creuse (True)...a voir

###########################################################################    
    
    # Pipeline et validation croisée
    
    results_full_list = []
    
    # Pipeline
    for name,regressor in models_full.items():
        # Choix du préproc
        ct = ct_dense_full if name.lower() == 'svr' else ct_sparse_full

        # Transformation sur y (transforme y, entraine puis inverse la transformation)
        model = TransformedTargetRegressor(
            regressor=regressor,
            transformer=y_tf
        )
        
        pipe_full = Pipeline([
            ('preprocessing',ct),
            ('modele',model)
        ])
        
        # validation croisée du modele name ==> on utilise X et y car la validation croisée se charge justement de faire le train_split
        scores_full = cross_validate(
            pipe_full,
            X,
            y,
            cv=cv,
            scoring = list(scoring.values()),
            return_train_score=True
        )
    
        # 
        result = {'Modele':name}
        for metric_nickname,metric_name in scoring.items():
            mean_score_train = np.mean(scores_full[f'train_{metric_name}'])
            mean_score_test = np.mean(scores_full[f'test_{metric_name}'])
            result[f'{metric_nickname}_train'] = -mean_score_train if 'neg_' in metric_name else mean_score_train
            result[f'{metric_nickname}_test'] = -mean_score_test if 'neg_' in metric_name else mean_score_test
        results_full_list.append(result)

###########################################################################
    
    results_full = pd.DataFrame(results_full_list)#.sort_values(by='r2', ascending=False)
    if show == True:
        display(results_full)
    
    return numeric_list,cat_list,ct_sparse_full,ct_dense_full,results_full

######################################################################################
######################################################################################
######################################################################################

def cat_modeling_cv(
    X:pd.DataFrame,
    y:pd.Series,
    numeric_list:list,
    cat_list:list,
    models_full : dict,
    log_tf = False,
    scaler_tf = None,
    cat_tf = None,
    y_tf: bool = False,
    scoring :dict = {
        'accu':'accuracy',
        'prec':'precision',
        'recall':'recall'
    },
    random_state = 42,
    cv=5,
    show=False,
    add_pipe:dict = None,
):
    '''
    Fonction pour entrainer et évaluer les performances de plusieurs modèles de CLASSIFICATION avec la méthode de la validation croisée.
    
    ENTREES:
    
    X: Liste des colonnes à supprimer de X avant modélisation.\n
    y: Nom de la variable cible.\n
    models_full: Dictionnaire des modèles à étudier\n
    log_tf: Transformation log1p à réaliser ou non.\n
    scaler_tf: Standardisation. Par défaut: StandardScaler(). Les alternatives sont:
    [
    MinMaxScaler() : borne juste sur [0,1]. Tres sensibles aux outliers
    RobustScaler() : utilise la mediane et l'iqr donc insensibles aux outliers mais l'echelle est conservée (mauvais pour SVR par exemple?)
    QuantileTransformer() : Transforme les données pour suivre une distribution donnée ce qui supprime les outliers et ramene l'echelle 
                            MAIS transformation non-linéaire + transformation couteuse!
    ]
    cat_tf: Transformation sur les catégories.
    les métriques (classification) choisies. Par défaut: ('accuracy','precision','recall')
    {
        'accu':'accuracy',
        'f1':'f1',
        'prec':'precision',
        'recall':'recall',
        'RC':'roc_auc',
    }\n
    random_state : par défaut = 42. \n
    cv = par défaut = 5\n
    
    SORTIES:
    
    ct_sparse_full: Préprocesseur avec matrice creuse\n
    ct_dense_full: Préprocesseur avec matrice dense\n
    results_full: Dataframe des résultats\n
    show: Affiche les résultats. Par défaut False)
    '''
# ###########################################################################

#     # # Transformation des variables numériques (log + standardisation)
#     #=================================================================================
#     # # Adaptapteur pour appliquer la fonction Numpy log. Si log_tf=False, cette transformation n'est pas appliquée 
#     # logarithm = FunctionTransformer(np.log, inverse_func = np.exp, validate=False) if log_tf else 'passthrough'
#     logarithm = PowerTransformer(method="Box-Cox") if log_tf else 'passthrough' # Fonctionne normalement avec get_names_out de feature_importance
    
#     # Transformation sur X
#     num_tf = Pipeline([
#         ('log', logarithm),
#         ('scaler', scaler_tf)
#     ])

#     # Transformation sur y
#     # Transformation de y: on log puis standardise
#     y_tf = Pipeline([
#         ('log', logarithm),
#         ('scaler', scaler_tf)
#     ]) if y_tf else None #Si pas de transformation a apportée a y


# ###########################################################################
    
#     # Mise en place des préprocesseurs en sparse et dense + injection de FunctionTransformer afin qu'il reste dans la pipeline
#     transformers_sparse = []
#     transformers_dense = []

#     if numeric_list:
#         transformers_sparse.append(('num', num_tf, numeric_list))
#         transformers_dense.append(('num', num_tf, numeric_list))
#     if cat_list:
#         transformers_sparse.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_list))
#         transformers_dense.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_list))

#     ct_sparse_full = ColumnTransformer(transformers_sparse)
#     ct_dense_full = ColumnTransformer(transformers_dense)

# ###########################################################################  
    # Transformation X et y
    num_tf,y_tf = Xy_tf(scaler_tf=scaler_tf,logX_tf=log_tf,logy_tf=log_tf,log_method = "Box-Cox",y_tf=False)

    # Pré-traitement
    ct_sparse_full,ct_dense_full = preproc(
        numeric_list=numeric_list,
        num_tf=num_tf,
        cat_list = cat_list,
        cat_list_splitted = None,
        encoding={
            'ohe':OneHotEncoder(handle_unknown='ignore'),
            # 'label':LabelEncoder(),
            # 'binary':LabelBinarizer()
        },
        sparse_output = 'both',
        dense_encoding = {
            'ohe':OneHotEncoder(handle_unknown='ignore',sparse_output=True),
        }
    )
    
    # Pipeline et validation croisée
    results_full_list = []
    
    # Pipeline
    for name,classifier in models_full.items():
        print(f'Modèle : {name} en cours')
        # Choix du préproc
        ct = ct_dense_full if name.lower() == 'svc' else ct_sparse_full

        # En cas d'absence des fonctions predict_proba dans le modele de classif
        if not hasattr(classifier, "predict_proba"):
            classifier = CalibratedClassifierCV(
                base_estimator = classifier,
                method = "sigmoid",
                cv = 3
            )

        # Transformation de y si y_tf activé
        if y_tf:
            model = TransformedTargetRegressor(
                regressor=classifier,
                transformer=y_tf
            )
        else:
            model = classifier
        
        # pipe_full = Pipeline([
        #     ('preprocessing',ct),
        #     ('modele',model)
        # ])
        if add_pipe:
            pipe_full = add_pipes(model,ct,**add_pipe)
        else:
            pipe_full = Pipeline([
            ('preprocessing',ct),
            ('modele',model)
        ])
        
        # validation croisée du modele name ==> on utilise X et y car la validation croisée se charge justement de faire le train_split
        scores_full = cross_validate(
            pipe_full,
            X,
            y,
            cv=cv,
            scoring = list(scoring.values()),
            return_train_score=True
        )
    
        # 
        result = {'Modele':name}
        for metric_nickname, metric_name in scoring.items():
            result[f'{metric_nickname}_train'] = np.mean(scores_full[f'train_{metric_name}'])
            result[f'{metric_nickname}_test'] = np.mean(scores_full[f'test_{metric_name}'])
        results_full_list.append(result)

###########################################################################
    
    results_full = pd.DataFrame(results_full_list)
    
    if show:
        display(results_full)
    
    return numeric_list,cat_list,ct_sparse_full,ct_dense_full,results_full

######################################################################################
######################################################################################
######################################################################################

def gridsearchcv(
    X, 
    y, 
    ct,
    param_grid,
    regressor,
    log_tf = False,
    scaler_tf = None,
    cat_tf = None,
    scoring: dict= None, 
    # {
    #     'MAE':'neg_mean_absolute_error',
    #     'MSE':'neg_mean_squared_error',
    #     'R2':'r2',
    # },
    random_state = 42,
    cv=5
):
    '''
    Fonction pour déployer et retourner les résultats de la recherche d'optimisation des hyperparamètres 
    d'un modèle de REGRESSION par GridSearchCV.
    
    ENTREES:
    
    X : Prédicteurs\n
    y : Variable cible.\n
    ct : Préprocessing.\n
    param_grid: Grille des hyperparamètres avec leur valeurs à étudier.\n
    regressor: Modele de regression\n
    log_tf: Transformation log1p à réaliser ou non.\n
    scaler_tf: Standardisation. Par défaut: StandardScaler(). Les alternatives sont:
    [
    MinMaxScaler() : borne juste sur [0,1]. Tres sensibles aux outliers
    RobustScaler() : utilise la mediane et l'iqr donc insensibles aux outliers mais l'echelle est conservée (mauvais pour SVR par exemple?)
    QuantileTransformer() : Transforme les données pour suivre une distribution donnée ce qui supprime les outliers et ramene l'echelle 
                            MAIS transformation non-linéaire + transformation couteuse!
    ]
    cat_tf: Transformation sur les catégories.
    scoring : les métriques (regression) choisies. Par défaut: ('neg_mean_absolute_error','neg_mean_squared_error','r2')
    {
        'ExplainedVar':'explained_variance',
        'MaxError':'neg_max_error',
        'MAE':'neg_mean_absolute_error',
        'MSE':'neg_mean_squared_error',
        'RMSE':'neg_root_mean_squared_error',
        'MSlogE':'neg_mean_squared_log_error',
        'RMSlogE':'neg_root_mean_squared_log_error',
        'MedianAE':'neg_median_absolute_error',
        'R2':'r2',
        'MPD':'neg_mean_poisson_deviance',
        'MGD':'neg_mean_gamma_deviance',
        'MAPE':'neg_mean_absolute_percentage_error',
        'd2AE':'d2_absolute_error_score'
    }\n
    random_state : par défaut = 42. \n
    cv = par défaut = 5\n
    
    SORTIES:
    
    best_model: Renvoie l'estimateur complet (modèle ou pipeline) entrainé avec les meilleurs paramètres. 
    Peut être utiliser directement pour la prédiction\n
    best_params: Renvoie la combinaison des meilleurs paramètres\n
    best_score: Renvoie le score moyen obtenu par validation croisée
    preproc_fitted: Renvoie le preprocessing entrainé sur le modele optimisé\n
    results: Renvoie la dataframe de résultats
    '''
###########################################################################

    # Transformation des variables numériques (log + standardisation)

    # Adaptapteur pour appliquer la fonction Numpy log. Si log_tf=False, cette transformation n'est pas appliquée 
    # logarithm = FunctionTransformer(np.log, inverse_func = np.exp, validate=False) if log_tf else 'passthrough'
    logarithm = FunctionTransformer(np.log, inverse_func = np.exp, validate=False) if log_tf else 'passthrough'

    
    # Transformation sur X
    num_tf = Pipeline([
        ('log', logarithm),
        ('scaler', scaler_tf)
    ])

    # Transformation sur y
    # Transformation de y: on log puis standardise
    y_tf = Pipeline([
        ('log', logarithm),
        ('scaler', scaler_tf)
    ])

    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=y_tf
    )
    
    # Pipeline
    pipe = Pipeline([
        ('preproc', ct),
        ('modele', model)
    ])

    # GridSearchCV 
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit='R2',
        cv=cv,
        return_train_score=True
    )

    # Entraînement
    grid.fit(X, y)

    # Réultats
    results = pd.DataFrame(grid.cv_results_)

    # Optimisation
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    preproc_fitted = best_model.named_steps['preproc']

    return best_model, best_params, best_score, preproc_fitted, results

######################################################################################
######################################################################################
######################################################################################

def cat_modeling_cv_predict(
    X:pd.DataFrame,
    y:pd.Series,
    numeric_list:list,
    cat_list:list,
    models_full : dict,
    log_tf = False,
    scaler_tf = None,
    cat_tf = None,
    y_tf: bool = False,
    random_state = 42,
    cv=5,
    show=False,
    class_threshold: float = 0.5,
    add_pipe:dict = None,
):
    '''
    Validation croisée retournant les y_pred des modèles
    
    ENTREES:
    
    X: Liste des colonnes à supprimer de X avant modélisation.\n
    y: Nom de la variable cible.\n
    models_full: Dictionnaire des modèles à étudier\n
    log_tf: Transformation log1p à réaliser ou non.\n
    scaler_tf: Standardisation. Par défaut: StandardScaler(). Les alternatives sont:
    [
    MinMaxScaler() : borne juste sur [0,1]. Tres sensibles aux outliers
    RobustScaler() : utilise la mediane et l'iqr donc insensibles aux outliers mais l'echelle est conservée (mauvais pour SVR par exemple?)
    QuantileTransformer() : Transforme les données pour suivre une distribution donnée ce qui supprime les outliers et ramene l'echelle 
                            MAIS transformation non-linéaire + transformation couteuse!
    ]
    cat_tf: Transformation sur les catégories.
    random_state : par défaut = 42. \n
    cv = par défaut = 5\n
    class_threshold: Seuil de décision de la classe. Défaut 0.5
    add_pipes: Si on veut ajouter des étapes intermédiaires comme l'oversampling à la pipeline finale
    
    SORTIES:
    
    results_full: Dataframe des résultats\n
    show: Affiche les résultats. Par défaut False)
    '''
    from sklearn.model_selection import cross_val_predict # Ne fonctionne pas en import globale
    ###########################################################################
    # Transformation X et y
    num_tf,y_tf = Xy_tf(scaler_tf=scaler_tf,logX_tf=log_tf,logy_tf=log_tf,log_method = "Box-Cox",y_tf=False)

    ###########################################################################
    
    # Pré-traitement
    ct_sparse_full,ct_dense_full = preproc(
        numeric_list=numeric_list,
        num_tf=num_tf,
        cat_list = cat_list,
        cat_list_splitted = None,
        encoding={
            'ohe':OneHotEncoder(handle_unknown='ignore'),
            # 'label':LabelEncoder(),
            # 'binary':LabelBinarizer()
        },
        sparse_output = 'both',
        dense_encoding = {
            'ohe':OneHotEncoder(handle_unknown='ignore',sparse_output=True),
        }
    )
    ###########################################################################    
    
    # Pipeline et validation croisée
    results_full_list = []
    
    # Pipeline
    for name,classifier in models_full.items():
        print(f'Modèle : {name} en cours')
        # Choix du préproc
        ct = ct_dense_full if name.lower() == 'svc' else ct_sparse_full

        # En cas d'absence des fonctions predict_proba dans le modele de classif
        if not hasattr(classifier, "predict_proba"):
            classifier = CalibratedClassifierCV(
                base_estimator = classifier,
                method = "sigmoid",
                cv = cv
            )

        # Transformation de y si y_tf activé
        if y_tf:
            model = TransformedTargetRegressor(
                regressor=classifier,
                transformer=y_tf
            )
        else:
            model = classifier

        
        # pipe_full = Pipeline([
        #     ('preprocessing',ct),
        #     # ('smote',SMOTE()),
        #     ('modele',model)
        # ])
        if add_pipe:
            pipe_full = add_pipes(model,ct,**add_pipe)
        else:
            pipe_full = Pipeline([
            ('preprocessing',ct),
            ('modele',model)
        ])
        
        # validation croisée du modele name ==> on utilise X et y car la validation croisée se charge justement de faire le train_split
        predict_proba = cross_val_predict(
            pipe_full,
            X,
            y,
            cv=cv,
            method = 'predict_proba'
        )
    
        # 
        result = {'Modele':name,'y_pred_proba':predict_proba,'y_pred':(predict_proba[:,1]>=class_threshold).astype("int64")}
        results_full_list.append(result)

    
###########################################################################
    
    results_full = pd.DataFrame(results_full_list)
    # premier crochet : [modele, colonne] / deuxieme: [numero employé] / troisième: [classe]
    # [(results_full.loc[model,'y_pred_proba'][employee_idx][1] >=0.5).astype(int) for model in results_full['Modele'] for employee_idx in range(len(results_full))]
    
    if show:
        display(results_full)
    
    return results_full
