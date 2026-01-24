import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Any, Literal

# Scikit-Learn / Imblearn
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PowerTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from imblearn.pipeline import Pipeline  # Compatible avec SMOTE

######################################################################################
# 1. PREPROCESSING FACTORY
######################################################################################

def build_numerical_pipeline(
    scale_method: Any = None,
    log_method: Literal['box-cox', 'yeo-johnson', None] = None
) -> Pipeline:
    """
    Construit le pipeline de transformation pour les variables numériques.
    
    Args:
        scale_method: Instance de scaler (ex: StandardScaler(), RobustScaler()).
        log_method: Méthode de transformation ('box-cox', 'yeo-johnson') ou None.
        
    Returns:
        Pipeline Scikit-Learn pour les numériques.
    """
    steps = []
    
    # 1. Transformation Log / Power
    if log_method:
        # Note: Yeo-Johnson supporte les négatifs, Box-Cox requiert du positif strict
        steps.append(('power_tf', PowerTransformer(method=log_method)))
        
    # 2. Scaling
    if scale_method:
        steps.append(('scaler', scale_method))
        
    # Si aucune étape, on met un passthrough pour éviter une pipeline vide
    if not steps:
        steps.append(('identity', FunctionTransformer(func=None))) # No-op
        
    return Pipeline(steps)




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
        num_pipeline: Pipeline de traitement numérique (issue de build_numerical_pipeline).
        sparse_output: Si True, retourne une matrice sparse (plus léger en RAM pour OHE).
        
    Returns:
        ColumnTransformer prêt à l'emploi.
    """
    transformers = []
    
    # Branche Numérique
    if numeric_features:
        transformers.append(('num', num_pipeline, numeric_features))
        
    # Branche Catégorielle (OHE par défaut pour compatibilité générique)
    if categorical_features:
        # Pour les arbres (RF, XGB), dense est souvent mieux. Pour Linear, sparse est mieux.
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


