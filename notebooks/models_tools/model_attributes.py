import pandas as pd
from typing import Dict, Any, Union, List, Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

# ===========================================================================

def model_attr(
    models: Dict[str, Any],
    default_attrs:List[str]|None=["predict_proba", "decision_function"],
    *extra_attrs: str
) -> pd.DataFrame:
    """
    Vérifie les capacités techniques d'un dictionnaire de modèles.
    par défaut verifie si le modèle ML possède predict_proba et decision_function.
    
    Args:
        models: Dictionnaire {nom: instance_du_modele}.
        *extra_attrs: Noms d'attributs supplémentaires à vérifier 
            (ex: 'coef_', 'feature_importances_').
        
    Returns:
        Un DataFrame récapitulatif (booléens).
    """
    # On fusionne avec les extras en évitant les doublons
    # fromkeys permet de garder default_attrs en premier VS set qui mélange
    attrs_to_check = (
        list(dict.fromkeys(default_attrs + list(extra_attrs)))
        if default_attrs
        else list(dict.fromkeys(list(extra_attrs)))
    )
    
    capabilities = []
    
    for name, model in models.items():
        # Création d'un dictionnaire de statut pour ce modèle
        # On doit justifier la forme de status pour plaire à Pylance
        status: Dict[str, Union[str, bool]] = {"model_name": name}
        for attr in attrs_to_check:
            status[attr] = hasattr(model, attr)
        
        capabilities.append(status)
        
    return pd.DataFrame(capabilities).set_index("model_name")


# ===========================================================================

def predict_proba_wrapper(
    model: Any, 
    method:Literal['sigmoid','isotonic']='sigmoid', 
    cv: int = 3
) -> Any:
    """
    Vérifie si le modèle a 'predict_proba'. Sinon, l'enveloppe dans un CalibratedClassifierCV.
    Utile pour SVM (SVC) ou RidgeClassifier.
    """
    if not hasattr(model, "predict_proba"):
        print(
            f"{model.__class__.__name__} n'a pas predict_proba."\
                "Ajout de CalibratedClassifierCV."
            )
        return CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    
    # ======== pb avec catboost
    # On vérifie juste s'il lui manque les tags Scikit-Learn 1.6
    if not hasattr(model, "__sklearn_tags__"):
        # On lui "injecte" dynamiquement la méthode par défaut de BaseEstimator
        
        model.__class__.__sklearn_tags__ = BaseEstimator.__sklearn_tags__ #type:ignore
    
    return model