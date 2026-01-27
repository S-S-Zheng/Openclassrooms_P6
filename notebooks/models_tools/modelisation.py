import pandas as pd
import numpy as np
from typing import Dict, Union
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline

def modeling_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: Dict[str, Pipeline],
    scoring: Dict[str, str]={
        'f1': 'f1',
        'prec': 'precision',
        'recall':'recall'
    },
    cv: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Exécute la validation croisée sur un dictionnaire de Pipelines déjà configurées.
    On utilise cross validate() ==> Comparaison de modèles.
    
    Args:
        X, y: Données d'entraînement.
        pipelines: Dictionnaire {nom: pipeline_complète}.
        scoring: Dictionnaire des métriques Scikit-Learn.\n
            Remarque: A revoir pour la regression\n
            REGRESSION:
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
            CLASSIFICATION:
            'accu':'accuracy',
            'f1':'f1',
            'prec':'precision',
            'recall':'recall',
            'RC':'roc_auc'
        cv: Nombre de folds.
        
    Returns:
        DataFrame comparatif des performances.
    """
    results_list = []
    
    for name, pipe in pipelines.items():
        if verbose:
            print(f" Évaluation de {name}...")
            
        # Cross-Validation
        # Note: on utilise n_jobs=-1 pour paralléliser, 
        # sauf si le modèle le fait déjà (ex: XGBoost)
        # Si conflit de threads, mettre n_jobs=1
        cv_results = cross_validate(
            pipe, 
            X, 
            y,
            cv=cv, 
            scoring=scoring, 
            return_train_score=True,
            # n_jobs=-1
        )
        
        # Agrégation des résultats
        row = {'Model': name}
        for metric_alias, metric_sklearn_name in scoring.items():
            # Scikit-Learn renvoie 'test_<metric_name>' et 'train_<metric_name>'
            # Les clés de cv_results dépendent des valeurs de scoring, pas des clés
            # Astuce: on utilise l'index des clés scoring pour retrouver les résultats
            
            # Note: cross_validate utilise les noms passés en valeurs dans scoring
            # Si scoring={'AUC': 'roc_auc'}, cross_validate renvoie 'test_roc_auc'
            train_score = np.mean(cv_results[f'train_{metric_sklearn_name}'])
            test_score = np.mean(cv_results[f'test_{metric_sklearn_name}'])
            
            row[f'Train {metric_alias}'] = train_score
            row[f'Test {metric_alias}'] = test_score
            
        row['Time'] = np.mean(cv_results['fit_time'])
        results_list.append(row)
        
    # Tri par la première métrique de test
    first_metric = list(scoring.keys())[0]
    return (
        pd.DataFrame(results_list)
        .sort_values(by=f'Test {first_metric}', ascending=False)
    )


# ===========================================================================


def predict_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: Dict[str, Pipeline],
    cv: int = 5,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Génère les prédictions (Proba et Classe) via cross_val_predict() qui est utile pour:
        - Calibration: Ajustement FP/FN
        - Stacking: utiliser les prédictions d'un modèle comme features por un autre
        - Analyser les erreurs: Comprendre les résultats de FP/FN du modèle
        - Seuil de validation : Tracer les AUC
    
    Se base sur cat_modeling_cv_predict.
    """
    results_list = []
    
    for name, pipe in pipelines.items():
        # method='predict_proba' retourne (n_samples, n_classes)
        y_probas = cross_val_predict(
            pipe, 
            X, 
            y, 
            cv=cv, 
            method='predict_proba', 
            # n_jobs=-1
        )
        # On prend la proba de la classe positive (1)
        positive_probs = y_probas[:, 1]
        predictions = (positive_probs >= threshold).astype("int8")
        
        res = pd.DataFrame({
            'Modele': name,
            'Indice': X.index,
            'True_Label': y,
            'y_proba1': positive_probs,
            'y_pred1': predictions
        })
        results_list.append(res)
        
    return pd.concat(results_list, ignore_index=True)