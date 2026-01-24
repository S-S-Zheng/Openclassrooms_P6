# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from typing import List, Literal,cast
from pathlib import Path

from notebooks.utils.preproc_contingency import preprocess_contingency_data
from notebooks.utils.config_figures import style_heatmap_axis, save_figure, setup_subplots
from notebooks.correlation_tools.correlation import abs_correlation


def plot_heatmaps(
    df: pd.DataFrame, 
    vmin: float, 
    vmax: float, 
    methods: List[Literal['pearson', 'kendall', 'spearman']] = ['pearson', 'spearman'],
    title_save: str = "Heatmap",
    path_save: Path| None = None, 
    save: bool = False
) -> None:
    '''
    Génère la heatmap suivant Pearson ou un comparatif Pearson/Spearman
    
    Args:
        df: Dataframe
        vmin: Valeur min
        vmax: Valeur max
        methods: méthodes de corrélation (défaut: pearson, spearman)
        title_save: Titre du fichier sauvegardé
        save: Booléen pour activer la sauvegarde
    '''
    
    # Préparation des données
    clean_df = preprocess_contingency_data(df)

    # Configuration de la figure
    fig, axes = setup_subplots(len(methods),1, True, True, True)
    
    # S'assurer que 'axes' est toujours une liste (iterable) même s'il n'y a qu'un plot
    if len(methods) == 1:
        axes = [axes]

    # Boucle de création des graphiques
    for i, method in enumerate(methods):
        ax = cast(Axes,axes[i])
        
        # Calcul
        corr_matrix = abs_correlation(clean_df, method)
        
        # Génération des labels numériques pour l'axe X
        ind_name_to_num_list = np.arange(len(corr_matrix)).astype(str)

        # Plotting
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=list(ind_name_to_num_list),
            linewidth=.5,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        # Styling
        titre_graphique = f'Heatmap suivant la correlation de {method} (valeur absolue)'
        style_heatmap_axis(ax, titre_graphique)


    plt.tight_layout()
    
    if save:
        save_figure(title_save, path_save)
    plt.show()