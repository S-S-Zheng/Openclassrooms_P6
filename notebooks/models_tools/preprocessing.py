"""
issu de Xy_tf + preproc
"""
# import pandas as pd
import numpy as np
from typing import List, Literal, Any

# Scikit-Learn / Imblearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer, PowerTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer

# ===========================================================================

def preproc_numerical_features(
    inferance: bool = False,
    scale_method: Any = None,
    log_method: Literal['box-cox', 'yeo-johnson', None] = None
) -> Pipeline:
    """
    Construit le pipeline de transformation pour les variables numériques.
    reamrque; issu de de la fonction Xy_tf, ne se concentre que sur
    la Classification ==> y rarement tf donc pas mentionner!
    
    Args:
        inferance: réalise de l'inférence sur les valeurs si True. impute la médiane au NaN
        scale_method: Instance de scaler.\n
            EXMPLES:
            'StandardScaler()': Classique, normalise en supprimant la moyenne 
            'MinMaxScaler()' : borne juste sur [0,1]. Tres sensibles aux outliers
            'RobustScaler()' : utilise la mediane et l iqr donc 
                insensibles aux outliers mais l echelle est conservée 
                (mauvais pour SVR)
            'QuantileTransformer()' : Transforme les données pour suivre 
                une distribution donnée ce qui supprime les outliers 
                et ramene l echelle MAIS transformation non-linéaire 
                + transformation couteuse!
    ]
        log_method: Méthode de transformation ('box-cox', 'yeo-johnson') ou None.
        
    Returns:
        Pipeline
    """
    steps = []
    
    # Impute médiane aux NaN
    if inferance:
        steps.append(('imputer', SimpleImputer(missing_values=np.nan,strategy='median')))
    
    # Transformation Log / Power
    if log_method:
        # Note: Yeo-Johnson supporte les négatifs, 
        # Box-Cox requiert du positif strict
        steps.append(('power_tf', PowerTransformer(method=log_method)))
        
    # Scaling
    if scale_method:
        steps.append(('scaler', scale_method))
        
    # Si aucune étape, on met un passthrough pour éviter une pipeline vide
    if not steps:
        steps.append(('identity', FunctionTransformer(func=None))) # No-op
        
    return Pipeline(steps)


# ===========================================================================


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    num_pipeline: Pipeline,
    sparse_output: bool = False
) -> ColumnTransformer:
    """
    Assemble le ColumnTransformer complet (Numérique + Catégoriel).
    
    Args:
        numeric_features: Liste des noms de colonnes numériques.
        categorical_features: Liste des noms de colonnes catégorielles.
        num_pipeline: Pipeline de traitement numérique 
            (ISSUE de build_numerical_pipeline).
        sparse_output: Si True, retourne une matrice sparse 
            (plus léger en RAM pour OHE).
        
    Returns:
        ColumnTransformer prêt à l'emploi.
    """
    transformers = []
    
    # Branche Numérique
    if numeric_features:
        transformers.append(('num', num_pipeline, numeric_features))
        
    # Branche Catégorielle (OHE par défaut pour compatibilité générique)
    if categorical_features:
        # Pour les arbres (RF, XGB), dense est souvent mieux. 
        # Pour Linear, sparse est mieux.
        ohe = OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=sparse_output
        )
        transformers.append(('cat', ohe, categorical_features))
        
    return ColumnTransformer(
        transformers=transformers, 
        remainder='drop', # On jette ce qui n'est pas listé
        sparse_threshold=1.0 if sparse_output else 0.0
    )


