import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

######################################################################################
######################################################################################
######################################################################################

# Structure des plots
def create_fig(num_features):
    '''
    ENTREES:

    num_features: nombre de features\n

    SORTIES:
    fig: la figure \n
    axes: les axes aplatis\n
    '''
    cols = 1 if num_features <= 1 else 2
    rows = int(np.ceil(num_features / cols))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(8 * cols, 6 * rows),
        sharex=False, sharey=False,
        clear=True,
    )
    axes = np.array(axes).reshape(-1) # flatten

    return fig, axes

######################################################################################
######################################################################################
######################################################################################

def make_figure(data:pd.DataFrame, x:str, kind:str, ycols:list, scale:bool=False):
    """
    Crée la figure pour un type de plot donné.

    ENTREES:
    
    data: Dataframe\n
    x: variable en abscisse\n
    kind : 'scatter', 'box', 'hist'\n
    ycols : liste des features\n

    SORITES:
    fig: la figure\n
    
    """
    if not ycols:
        return None
    # num_col = len(ycols)
    fig, axes = create_fig(len(ycols))

    for ax, ind in zip(axes, ycols):
        
        
        if kind == "scatter":
            sns.scatterplot(
                data=data, 
                x=x, 
                y=ind, 
                ax=ax
            )
            if scale:
                ax.set_xscale("log")
                ax.set_yscale("log")

        elif kind == "box":
            if x: # Comparaison des catégories d'une feature par rapport aux autres features
                sns.boxplot(
                    data=data,
                    x=x, # Catégories de la feature de réf
                    y=ind, # Les autres features
                    ax=ax,
                    medianprops={"color": "r", "linewidth": 2},
                    log_scale = scale
                )
            else: # Boxplot classique
                sns.boxplot(
                    data=data,
                    y=ind,
                    ax=ax,
                    medianprops={"color": "r", "linewidth": 2},
                    log_scale = scale
                )
                ax.set_title(f"Boîte à moustache de {ind}", fontsize=10)
                ax.set_ylabel(ind)
                continue

        elif kind == "hist":
            if data[ind].dtype in ['int64', 'float64','int32', 'float32'] :
                sns.histplot(
                    data=data, 
                    x=ind, 
                    # y=ind, 
                    ax=ax
                )
                ax.set_title(f"Distribution de {ind}", fontsize=10)
                ax.set_xlabel(ind)
                continue
            else:
                sns.histplot(
                    data=data, 
                    # x=ind,
                    y=ind, 
                    ax=ax
                )
                ax.set_title(f"Distribution de {ind}", fontsize=10)
                ax.set_ylabel(ind)
                continue
        
        if kind != "hist" and x:
            ax.set_title(f"{ind} suivant {x}", fontsize=10)
            ax.set_xlabel(x)
            ax.set_ylabel(ind)

    # Suppression axes vides
    for ax in axes[len(ycols):]:
        fig.delaxes(ax)

    fig.tight_layout()
    return fig


######################################################################################
######################################################################################
######################################################################################

# Plot jusqu'à 4 types de graphiques : scatter, box, histogram et pairplot
def graphs(
    data: pd.DataFrame,
    x: str = None,
    plot_list = None,
    plot_list_scatter = None,
    plot_list_box = None,
    plot_list_hist = None,
    plot: str = "all",
    scale: bool = False,
    title_save: str = "scatter_box_hist_pair",
    save: bool = False,
    # Options pour le pairplot
    pairplot_hue: str = None,
    pairplot_diag_kind: str = "auto",  # "auto", "kde" ou "hist"
    pairplot_kind: str = "scatter"     # "scatter" ou "reg"
)->plt.Figure:
    '''
    data: dataframe\n
    x : variable utilisé pour x
    plot_list_scatter: liste des indicateurs de data à plotter en nuage de point\n
    scale: log-log les axes ou non (par défaut None)
    plot_list_box: liste des indicateurs de data à plotter en boîte à moustache\n
    plot_list_hist: liste des indicateurs de data à plotter en histogramme\n
    plot: type de graphique ("all" pour tous, "scatter" pour le nuage de point, "box" pour la boîte à moustache et "hist" pour l'histogramme)
    
    Retourne jusqu'à 4 types de graphiques : scatter, box, histogram et pairplot
    '''

    if x is None and plot != "hist":
        raise ValueError("x doit être spécifié.")

    if plot_list is not None:
        plot_list_scatter = plot_list_box = plot_list_hist = plot_list

    plot_map = {
        "scatter":plot_list_scatter,
        "box": plot_list_box,
        "hist":plot_list_hist,  
    }

    # Liste des types de plots à exécuter
    if plot in ["all","all+"]:
        to_do = ["scatter", "box", "hist"]
    else:
        to_do = [plot]

    #
    row_flag = 'Total' in str(data.index[-1])
    col_flag = 'Total' in str(data.columns[-1])
    data2 = data.iloc[: -1 if row_flag else None, : -1 if col_flag else None]

######################################## Graphiques scatter / box et hist

    figs = {}
    if plot != "pair":
        for kind in to_do:
            figs[kind] = make_figure(
                data = data2, 
                x = x, 
                kind = kind, 
                ycols = plot_map.get(kind), 
                scale = scale
            )


######################################## Graphique pairplot
    if plot in ["all+", "pair"]:
        # # Détermination automatique des variables
        # cols = plot_list or data2.columns.tolist()
        # OU
        cols = data2.select_dtypes(include=np.number).columns.tolist()
        
        # On limite aux colonnes numériques ou catégorielles simples
        data_pair = data2[cols].copy()
        

        # Pairplot seaborn
        g = sns.pairplot(
            data=data_pair,
            hue=pairplot_hue,
            kind=pairplot_kind,
            diag_kind=pairplot_diag_kind if pairplot_diag_kind != "auto" else "kde"
        )
        figs["pairplot"] = g.fig


    if save:
        for plot, fig in figs.items():
            if fig is not None:
                fig.savefig(
                    fname=f"{title_save}_{plot}_{scale}.png",
                    dpi=300,
                    bbox_inches="tight"
                )
    
    plt.show()
    return figs

######################################################################################
######################################################################################
######################################################################################

def graph_hyperParamEffect(
    results:pd.DataFrame, 
    params_prefix='param_modele__regressor__', 
    metrics_prefixe='mean_test_',
    title_save = "hyperParamEffect",
    save = False,
    model_type = "regression"
)->plt.Figure:
    '''
    Visualise l'influence moyenne de chaque hyperparamètre du modèle sur les métriques à partir des résultats issus d'une GridSearchCV.

    ENTREES:

    results: Dataframe de résultats de la GridSearchCV, 
    params_prefix: préfixe précédent le nom des paramètres en question issu de la structure pipeline. Par défaut 'param_modele__regressor__', 
    metrics_prefixe = préfixe des métriques. Par défaut 'mean_test_'
    model_type = type de modélisation: regression / classification. défaut regression
    
    SORTIES:

    Ensemble de subplot: ligne = suivant une métrique / colonne = suivant un paramètre
    '''
    
    param_cols = [hyperparam for hyperparam in results.columns if hyperparam.startswith(params_prefix)]
    metric_cols = [metric for metric in results.columns if metric.startswith(metrics_prefixe)]

    n_metrics = len(metric_cols)
    n_params = len(param_cols)
    
    fig, axes = plt.subplots(n_metrics, n_params, figsize=(6*n_params, 4*n_metrics),squeeze=False) #squeeze False pour garder 2D sur l'axe

    for i, metric in enumerate(metric_cols):
        for j, param in enumerate(param_cols):
            # param_values = str(param) if not isinstance(param,str) else param # Convertie les valeurs du paramètre en str si ce n'est pas le cas
            ax=axes[i,j]
            param_name = param.replace(params_prefix, '')
            if model_type == "regression":
                metric_values = results[metric] if 'R2' in metric else -results[metric]
            else:
                metric_values = results[metric]
            sns.lineplot(
                data=results,
                x=param,
                y=metric_values,
                marker='o',
                errorbar='sd',  # intervalle de confiance IC par défaut = 95%/ sd pour la variabilité effective des folds
                ax=ax
            )

            ax.set_title(f"{metric} vs {param_name}")
            ax.set_xlabel(param_name)
            ax.grid(True)
            
    plt.tight_layout()
    if save: 
        plt.savefig(
            fname = f'{title_save}_Influence des hyperparametres sur le modele.png',
            dpi = 300,
            format = 'png',
            bbox_inches = 'tight'
        )
    plt.show()

######################################################################################
######################################################################################
######################################################################################

def graph_importance(
    model, 
    preproc=None,  
    numeric_list=None,non_numeric_list=None,
    X=None,y=None,n_repeats=30,random_state=42,
    mode = "feature",
    scoring = 'r2',
    title_save="importance",
    save = False
):
    """
    Extrait les importances des features à partir d’un modèle d'arbre si le mode est en feature pour tout autre si en mode permutation.
    """
    # Extraction du modele entrainé depuis la pipeline
    regressor = model.named_steps['modele'].regressor_

    if mode == "feature":
        try:
            importances = pd.Series(regressor.feature_importances_, 
                                index=model.named_steps['modele'].get_feature_names_out()#preproc.get_feature_names_out()
                               ).sort_values(ascending=False)
        except:    
            #####################get_feature_names_out() ne fonctionne pas car FunctionTransformer n'a pas cette méthode donc alternative:
            # Récupération des noms de variables catégorielles encodées
            cat_encoder = preproc.named_transformers_['cat']
            encoded_cat_features = cat_encoder.get_feature_names_out(non_numeric_list)
            # Fusion avec les numériques
            feature_names = np.concatenate([numeric_list, encoded_cat_features])
            #######################
            importances = pd.Series(regressor.feature_importances_, 
                                index=feature_names#preproc.get_feature_names_out()
                               ).sort_values(ascending=False)
    else:
        results = permutation_importance(model, X, y,
                                         n_repeats=n_repeats,
                                         random_state=random_state,
                                         scoring=scoring
                                        )
        importances = pd.Series(results.importances_mean, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(6,8))
    sns.barplot(
        x=importances.values, 
        y=importances.index
    )
    plt.title(f"{mode} Importance - {regressor}")
    plt.xlabel("Importance moyenne")
    plt.ylabel("Variables")
    plt.tick_params(axis = 'x',
                          # labelrotation = 20,
                          labelbottom = True,
                          labelsize = 10,
                          length=6,
                          width=2, 
                          colors='r',
                          grid_color='r',
                          grid_alpha=0.5
                         )
    
    plt.grid(True)
    if save:
        plt.savefig(
            fname = f'{title_save}_Importance des features.png',
            dpi = 300,
            format = 'png',
            bbox_inches = 'tight'
        )
    plt.show()

    return importances

######################################################################################
######################################################################################
######################################################################################
def catboost_graphs(X:list,pool=None):

    '''
    
    '''

    #================================================================================
    # Feature importance (split-base)
    feature_importance = best_model.get_feature_importance(type = "PredictionValuesChange")
        
    df_feat_importance = pd.DataFrame({
        'Features':X.columns,
        'Feature importance': feature_importance
    }).sort_values('Feature importance',ascending=False)

   #=================================================================================== 
    # Si présence de pool alors on ajoute aujoute la permutation importance
    
    if pool:
        
        permutation_importance = best_model.get_feature_importance(
        type = "PredictionValuesChange",
        data = pool #Pool(X,y,cat_features=cat_list)
        )
        
        df_perm_importance = pd.DataFrame({
            'Features':X.columns,
            'Permutation importance': permutation_importance
        }).sort_values('Permutation importance',ascending=False)

    #================================================================================

    # num_col = 2 if pool else 1
    # fig, axes = create_fig(num_col)

    # sns.barplot(
    #     x=importances.values, 
    #     y=importances.index,
    #     ax=ax
    # )
    # plt.title(f"{mode} Importance - {regressor}")
    # plt.xlabel("Importance moyenne")
    # plt.ylabel("Variables")

    # fig.grid(True)
    # if save:
    #     plt.savefig(
    #         fname = f'{title_save}_Importance des features.png',
    #         dpi = 300,
    #         format = 'png',
    #         bbox_inches = 'tight'
    #     )
    # plt.show()
        


    