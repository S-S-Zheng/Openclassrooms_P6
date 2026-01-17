
#imports
import matplotlib.pyplot as plt
from matplotlib.axes import Axes



def style_heatmap_axis(ax: Axes, title: str) -> None:
    """
    Applique le style spécifique aux axes du graphique.
    Responsabilité : Mise en forme (Styling).
    """
    ax.set_title(title, fontsize=12)
    
    # Paramètres communs pour les axes X et Y
    common_params = {
        'labelsize': 10,
        'length': 6,
        'width': 2,
        'colors': 'r',
        'grid_color': 'r',
        'grid_alpha': 0.5
    }
    
    ax.tick_params(axis='x', labelbottom=True, **common_params)
    ax.tick_params(axis='y', **common_params)


def save_figure(title: str) -> None:
    """
    Sauvegarde la figure courante.
    Responsabilité : Entrées/Sorties (I/O).
    """
    plt.savefig(
        fname=title,
        dpi=300,
        format='png',
        bbox_inches='tight'
    )

