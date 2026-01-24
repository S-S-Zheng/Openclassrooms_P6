import pandas as pd
from typing import Dict, Any, Union, List


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