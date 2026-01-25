"""
issue de add_pipes et de la partie pipeline des differents *modeling_cv
"""
from typing import Optional, Any

# Scikit-Learn / Imblearn
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from imblearn.pipeline import Pipeline  # Compatible avec SMOTE

from notebooks.models_tools.model_attributes import predict_proba_wrapper


# ===========================================================================


def build_classification_pipeline(
    model: BaseEstimator,
    preprocessor: Optional[ColumnTransformer] = None,
    sampler: Optional[Any] = None
) -> Pipeline:
    """
    Construit la Pipeline finale : Preprocessing -> Echantillonnage -> Model.\n
    REMRQUE: Certains modeles gerent le preproc en interne (souvent mieux) et
    ajouter une étape de preproc va affecter la perf voir le casser ==> Optionnel.
    ex: catboost avec l'encadage des categories ou le scaling avec les arbres.
    
    Args:
        model: L'estimateur (ex: RandomForestClassifier, XGBClassifier).
        preprocessor: Le ColumnTransformer 
            (OPTIONNEL ==> ex: inutile pour CatBoost natif).
        sampler: Méthode d'échantillonnage 
            (OPTIONNEL ==> ex: SMOTE(), RandomUnderSampler()).
        
    Returns:
        Pipeline Imblearn (Comme Pipeline standard mais compatible avec fit_resample).
    """
    steps = []
    
    # Preprocessing (Optionnel)
    if preprocessor:
        steps.append(('preprocessor', preprocessor))
        
    # Sampling (Optionnel, doit être supporté par imblearn)
    if sampler:
        steps.append(('sampler', sampler))
        
    # Modèle (Toujours en dernier)
    # On s'assure que le modèle peut prédire des probabilités
    model_ready = predict_proba_wrapper(model)
    steps.append(('classifier', model_ready))
    
    return Pipeline(steps)