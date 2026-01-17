# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Literal

from notebooks.utils.preproc_contingency import preprocess_contingency_data
from notebooks.utils.config_figures import style_heatmap_axis, save_figure
from notebooks.plotting.correlation import abs_correlation


def plot_heatmaps(
    data: pd.DataFrame, 
    vmin: float, 
    vmax: float, 
    methods: List[Literal['pearson', 'kendall', 'spearman']] = ['pearson', 'spearman'],
    title_save: str = "Heatmap", 
    save: bool = False
) -> None:
    '''
    Génère la heatmap suivant Pearson ou un comparatif Pearson/Spearman
    
    Args:
        data: Dataframe
        vmin: Valeur min
        vmax: Valeur max
        methods: méthodes de corrélation (défaut: pearson, spearman)
        title_save: Titre du fichier sauvegardé
        save: Booléen pour activer la sauvegarde
    '''
    
    # 1. Préparation des données
    clean_data = preprocess_contingency_data(data)

    # 2. Configuration de la figure
    fig, axes = plt.subplots(
        len(methods), 
        1, 
        figsize=(16, 8 * len(methods)), # Hauteur dynamique selon le nombre de plots
        sharex=True, 
        sharey=True,
        clear=True
    )
    
    # S'assurer que 'axes' est toujours une liste (iterable) même s'il n'y a qu'un plot
    if len(methods) == 1:
        axes = [axes]

    # 3. Boucle de création des graphiques
    for i, method in enumerate(methods):
        ax = axes[i]
        
        # Calcul
        corr_matrix = abs_correlation(clean_data, method)
        
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

    # 4. Finalisation et Sauvegarde
    plt.tight_layout()
    
    if save:
        save_figure(title_save)
    plt.show()