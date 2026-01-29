import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
from matplotlib.figure import Figure
# import numpy as np
from typing import List, Literal, Any, Optional,Tuple
from pathlib import Path
from catboost import CatBoostClassifier, Pool

from notebooks.plotting.config_figures import setup_subplots, save_figure


from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve,auc

# ===========================================================================


def plot_hyperparam_effect(
    cv_results: pd.DataFrame, 
    param_prefix: str = 'param_', 
    metric_prefix: str = 'mean_test_',
    model_type: Literal['regression', 'classification'] = "regression",
    title_save: str="hyperparams",
    save_path: Path|None = None
) -> Figure:
    """
    Visualise l'impact des hyperparamètres sur les métriques de cross-validation.
    
    ENTREES:
    cv_results: DataFrame issu d'une validation croisée (ex GridSearchCV.cv_results_)
    param_prefix: Filtre pour les colonnes paramètres (ex: 'param_')
    metric_prefix: Filtre pour les colonnes métriques (ex: 'mean_test_')
    model_type: 'regression' (inverse les scores négatifs) ou 'classification'
    """
    
    # Identification dynamique des colonnes
    params = [
        col for col in cv_results.columns 
        if col.startswith(param_prefix)
    ]
    metrics = [
        col for col in cv_results.columns 
        if col.startswith(metric_prefix) 
        and 'split' not in col
    ] # On exclut les splits individuels

    # Preparation figure et axes
    n_rows, n_cols = len(metrics), len(params)
    fig, axes = setup_subplots(n_rows*n_cols,n_cols)

    # Ajout de la liste de combinaison métrique/paramètre
    tasks = [(metric, param) for metric in metrics for param in params]
    for ax, (metric, param) in zip(axes,tasks):
        param_name = param.replace(param_prefix, '')
        
        # Gestion spécifique pour l'affichage (MSE négative en Sklearn)
        y_values = cv_results[metric]
        if model_type == "regression" and (y_values < 0).any():
            y_values = -y_values
            metric_label = f"Abs({metric})"
        else:
            metric_label = metric

        # Tracé avec intervalle de confiance
        sns.lineplot(
            data=cv_results, x=param, y=y_values,
            marker='o', errorbar='sd', ax=ax
        )
        
        ax.set_title(f"{metric_label} vs {param_name}")
        ax.set_xlabel(param_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_figure(title_save,save_path)
    return fig


# ===========================================================================


def get_feature_importance(
    model: Any, 
    feature_names: List[str]|None = None,
    preprocessor: Any = None,
    X_val: pd.DataFrame|None = None, 
    y_val: pd.Series|None = None,
    method: Literal['auto','native','permutation'] = "auto"
) -> pd.Series:
    """
    Extrait les importances des features de manière générique.
    Eviter pour les réseaux de neuronnes comme MLP ou sinon réduire significativement
    le nombre d'observations (1000-2000).
    
    ENTREES:
        model: ML entraîné
        features_names:
        preprocessor: preprocessing
        X_val: dataframe des features
        y_val: series de la cible
        method: 'auto', 'native' (tree), 'permutation'
    """
    importances = None
    
    # Extraction via Permutation (Modèle agnostique)
    if method == "permutation" or (method == "auto" and not hasattr(model, "feature_importances_")):
        if X_val is None or y_val is None:
            raise ValueError("X_val et y_val sont requis pour la permutation importance.")
        
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=10,
            random_state=42,
            # n_jobs=-1
        )
        # On force Pylance à comprendre que result a l'attribut importances_mean
        # 'result' est un objet de type Bunch qui se comporte comme un dict mais avec des attributs
        importances_val = getattr(result, "importances_mean", None)
        
        if importances_val is None:
            raise AttributeError(
                "L'objet retourné par permutation_importance ne contient pas 'importances_mean'."
            )
        
        importances = pd.Series(importances_val, index=X_val.columns)

    # Extraction Native (Arbres: RF, XGBoost, CatBoost)
    else:
        # Gestion pipeline Sklearn : on cherche l'étape finale
        estimator = model.named_steps['modele'] if hasattr(model, 'named_steps') else model
        
        if hasattr(estimator, "feature_importances_"):
            raw_importances = estimator.feature_importances_
            
            # Tentative de récupération des noms de features
            final_names = feature_names
            
            # Si un preprocesseur est fourni et qu'on n'a pas les noms
            if final_names is None and preprocessor:
                try:
                    final_names = preprocessor.get_feature_names_out()
                except AttributeError:
                    # Fallback si get_feature_names_out n'existe pas 
                    # (ex: vieux sklearn ou CustomTransformer)
                    final_names = [f"feat_{i}" for i in range(len(raw_importances))]
            
            # Si toujours pas de noms, on met des indices
            if final_names is None or len(final_names) != len(raw_importances):
                final_names = [f"feat_{i}" for i in range(len(raw_importances))]
                
                importances = pd.Series(raw_importances, index=final_names)
            
        elif hasattr(estimator, "get_feature_importance"): # CatBoost spécifique
            raw_importances = estimator.get_feature_importance()
            importances = pd.Series(
                raw_importances, index=feature_names 
                if feature_names 
                else range(len(raw_importances))
            )

    if importances is None:
        raise ValueError("Impossible d'extraire l'importance des features pour ce modèle.")
        
    return importances.sort_values(ascending=False)


# ===========================================================================


def plot_feature_importance(
    importances: pd.Series, 
    top_n: int = 20, 
    title_save: str = "Feature Importance",
    save_path: Path|None = None
) -> Figure:
    """
    Affiche l'importance des features.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # On prend le Top N
    data_to_plot = importances.head(top_n)
    
    sns.barplot(
        x=data_to_plot.values, 
        y=data_to_plot.index, 
        ax=ax, 
        palette="viridis"
    )
    
    ax.set_title(title_save)
    ax.set_xlabel("Importance Relative")
    ax.grid(True, axis='x', alpha=0.5)
    
    if save_path:
        save_figure(title_save,save_path)
        
    return fig


# ===========================================================================

# A revoir
def catboost_graphs(
    X:pd.DataFrame,
    best_model:CatBoostClassifier,
    pool:Optional[Pool]=None
):
    """ """

    # ========================================================================
    # Feature importance (split-base)
    feature_importance = best_model.get_feature_importance(
        type="PredictionValuesChange" # type: ignore
    )

    df_feat_importance = pd.DataFrame(
        {"Features": X.columns, "Feature importance": feature_importance}
    ).sort_values("Feature importance", ascending=False)

    # =======================================================================
    # Si présence de pool alors on ajoute la permutation importance

    if pool:

        permutation_importance = best_model.get_feature_importance(
            type="PredictionValuesChange", # type: ignore
            data=pool
        )

        df_perm_importance = pd.DataFrame(
            {"Features": X.columns, "Permutation importance": permutation_importance}
        ).sort_values("Permutation importance", ascending=False)

    return df_feat_importance, df_perm_importance



# ===========================================================================

def pr_curve(
    y_true:np.ndarray|pd.Series, 
    y_proba:np.ndarray|pd.Series, 
)->Tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    """
    Calcul precision, recall et threshold ainsi que l'auc.
    
    Args:
        y_true: target réelles.
        y_proba: Probabilités de classe positive.
    
    """
    # thresholds renvoyé par sklearn a une longueur n_thresholds
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    
    # On aligne les vecteurs pour prec et rec 
    # (sklearn ajoute 1 et 0 à la fin de prec/rec)
    return prec[:-1], rec[:-1], thresh, float(auc(rec, prec))