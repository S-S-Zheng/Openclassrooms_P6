"""
issue des *modeling_cv
"""


import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

# Scikit-Learn / Imblearn
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

from notebooks.models_tools.pipeline_builder import build_classification_pipeline


# ===========================================================================

def modeling_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, BaseEstimator],
    preprocessor: Optional[ColumnTransformer],
    sampler: Optional[Any] = None,
    scoring: Dict[str, str] = {
        'f1': 'f1',
        'prec': 'precision',
        'recall':'recall'
    },
    cv: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare plusieurs modèles via validation croisée.
    Va servir pour choisir le modèle parmis ceux tester.\n
    se base sur cat_modeling_cv
    
    Args:
        X, y: Données d'entraînement.
        models: Dictionnaire {nom: instance_modele}.
        preprocessor: ColumnTransformer commun.
        sampler: Instance SMOTE ou None.
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
        DataFrame résumant les scores (Train/Test) pour chaque métrique.
    """
    results_list = []
    
    for name, model_instance in models.items():
        if verbose:
            print(f" Évaluation de {name}...")
            
        # Construction Pipeline Unique pour ce modèle
        pipeline = build_classification_pipeline(model_instance, preprocessor, sampler)
        
        # Cross-Validation
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, 
            return_train_score=True, n_jobs=-1
        )
        
        # Agrégation des résultats
        row = {'Model': name}
        for metric_alias, _ in scoring.items():
            # Scikit-Learn renvoie 'test_<metric_name>' et 'train_<metric_name>'
            # Les clés de cv_results dépendent des valeurs de scoring, pas des clés
            # Astuce: on utilise l'index des clés scoring pour retrouver les résultats
            
            # Note: cross_validate utilise les noms passés en valeurs dans scoring
            # Si scoring={'AUC': 'roc_auc'}, cross_validate renvoie 'test_roc_auc'
            metric_sklearn_name = scoring[metric_alias]
            
            row[f'Train {metric_alias}'] = np.mean(cv_results[f'train_{metric_sklearn_name}'])
            row[f'Test {metric_alias}'] = np.mean(cv_results[f'test_{metric_sklearn_name}'])
            row[f'Time'] = np.mean(cv_results['fit_time'])
            
        results_list.append(row)
        
    return (
        pd.DataFrame(results_list)
        .sort_values(by=f'Test {list(scoring.keys())[0]}', ascending=False)
    )


# ===========================================================================


def predict_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, BaseEstimator],
    preprocessor: Optional[ColumnTransformer],
    sampler: Optional[Any] = None,
    cv: int = 5,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Génère les prédictions (Proba et Classe) via Cross-Validation (Out-of-Fold predictions).
    Utile pour construire une Stacking (utiliser les prédictions d'un modèle
    comme features por un autre) ou analyser les erreurs (Comprendre les résultats
    de Faux positi ou Négatif du modèle).
    Se base sur cat_modeling_cv_predict.
    """
    results_list = []
    
    for name, model_instance in models.items():
        pipeline = build_classification_pipeline(model_instance, preprocessor, sampler)
        
        # Cross-Val Predict (Probabilités)
        # method='predict_proba' retourne une matrice (n_samples, n_classes)
        y_probas = cross_val_predict(
            pipeline, 
            X, 
            y, 
            cv=cv, 
            method='predict_proba', 
            # n_jobs=-1
        )
        
        # Proba de la classe 1
        positive_probs = y_probas[:, 1]
        predictions = (positive_probs >= threshold).astype("int8")
        
        # On stocke le résultat global (pas par fold, mais pour tout le dataset)
        # Pour une analyse par ligne, on pourrait retourner un DataFrame avec Index
        res = pd.DataFrame({
            'Modele': name,
            'Indice': X.index,
            'True_Label': y,
            'y_proba1': positive_probs,
            'y_pred1': predictions
        })
        results_list.append(res)
        
    return pd.concat(results_list, ignore_index=True)