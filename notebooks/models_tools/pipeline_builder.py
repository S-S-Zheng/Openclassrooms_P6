from typing import List, Dict, Union, Tuple, Optional, Any

# Scikit-Learn / Imblearn
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from imblearn.pipeline import Pipeline  # Compatible avec SMOTE


######################################################################################
# 2. PIPELINE BUILDER
######################################################################################

def ensure_proba_capability(model: Any, cv: int = 3) -> Any:
    """
    Vérifie si le modèle a 'predict_proba'. Sinon, l'enveloppe dans un CalibratedClassifierCV.
    Utile pour SVM (SVC) ou RidgeClassifier.
    """
    if not hasattr(model, "predict_proba"):
        print(f"⚠️ {model.__class__.__name__} n'a pas predict_proba. Ajout de CalibratedClassifierCV.")
        return CalibratedClassifierCV(estimator=model, method="sigmoid", cv=cv)
    return model

def build_classification_pipeline(
    model: BaseEstimator,
    preprocessor: Optional[ColumnTransformer] = None,
    sampler: Optional[Any] = None
) -> Pipeline:
    """
    Construit la Pipeline finale : Preprocessing -> Sampling (SMOTE) -> Model.
    
    Args:
        model: L'estimateur (ex: RandomForestClassifier, XGBClassifier).
        preprocessor: Le ColumnTransformer (optionnel, ex: inutile pour CatBoost natif).
        sampler: Méthode d'échantillonnage (ex: SMOTE(), RandomUnderSampler()).
        
    Returns:
        Pipeline Imblearn (compatible fit_resample).
    """
    steps = []
    
    # 1. Preprocessing (Optionnel)
    if preprocessor:
        steps.append(('preprocessor', preprocessor))
        
    # 2. Sampling (Optionnel, doit être supporté par imblearn)
    if sampler:
        steps.append(('sampler', sampler))
        
    # 3. Modèle (Toujours en dernier)
    # On s'assure que le modèle peut prédire des probabilités
    model_ready = ensure_proba_capability(model)
    steps.append(('classifier', model_ready))
    
    return Pipeline(steps)