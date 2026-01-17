import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency, f_oneway, chi2

# Heatmap
def graphs_corr(data: pd.DataFrame, vmin: float, vmax: float,title_save="Heatmap", save = False):
    '''
    data: dataframe du tableau de contingence\n
    vmin: valeur min de la heatmap \n
    vmax: valeur max de la heatmap\n
    title_save = Titre de la sauvegarde

    Retourne une figure heatmap
    '''

    # # Confirme si présence de Total dans les dernieres cellules de data
    try:
        row_flag = 'Total' in data.index[-1]
    except (TypeError):
        row_flag = 'Total' in str(data.index[-1])
    
    col_flag = 'Total' in data.columns[-1]
    
    data2 = data.iloc[: -1 if row_flag else None, : -1 if col_flag else None]

    # Met en place la figure 
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(16,16),
        sharex = True,
        sharey = True,
        clear = True
    )

    # Liste des méthodes de correlation
    corr_method = ['pearson','spearman']

    # Subplots des Heatmap suivant la méthodede correlation
    for i,meth in enumerate(corr_method):

        # On regarde en valeur absolue + on retire les lignes et colonnes totaux + on transpose afin d'étudier la corr des indicateurs
        
        abs_corr_matrix = abs(data2.corr(numeric_only=True,method=meth))

        # Liste de numéros pour éviter de doubler les noms des indicateurs sur l'autre axe
        ind_name_to_num_list = np.arange(0,len(abs_corr_matrix),1)
    
        sns.heatmap(
            abs_corr_matrix,
            annot=True, # Inscrit dans les cases, les valeurs de correlation
            fmt=".2f",
            xticklabels = ind_name_to_num_list,
            linewidth=.5,
            vmin = vmin,
            vmax = vmax,
            ax = ax[i]
        )
    
        # ax[i].set_xlabel()
        # ax[i].set_ylabel()        
        ax[i].set_title(f'Heatmap suivant la correlation de {meth} (valeur absolue)', fontsize=12)

        # Modification des ticks pour rendre la figure globale plus lisible
        ax[i].tick_params(axis = 'x',
                          # labelrotation = 20,
                          labelbottom = True,
                          labelsize = 10,
                          length=6,
                          width=2, 
                          colors='r',
                          grid_color='r',
                          grid_alpha=0.5
                         )
        ax[i].tick_params(axis = 'y',
                          labelsize = 10,
                          length=6,
                          width=2, 
                          colors='r',
                          grid_color='r',
                          grid_alpha=0.5
                         )
        # plt.xticks(rotation = 60, ha='right',fontsize = 8)
        # plt.yticks(fontsize = 8)
    # plt.xticks(ha='right') 
    plt.tight_layout()
    if save:
        plt.savefig(
        fname = title_save,
        dpi = 300,
        format = 'png',
        bbox_inches = 'tight'
        )
    plt.show()

######################################################################################
######################################################################################
######################################################################################

# # Construit un dataframe répertoriant les couples corrélés
def CorrCouples_VIF(data :pd.DataFrame, vmin=0.7, vif_max=10)->pd.DataFrame:
    '''
    Etudie la correlation suivant la méthodede de pearson et spearman d'un dataframe et dresse 3 dataframes qui récapitule les couples,
    leur correlation et le facteur d'inflation de la variance (VIF) des indicateurs.
    '''

    # # Confirme si présence de Total dans les dernieres cellules de data

    row_flag = 'Total' in str(data.index[-1])
    col_flag = 'Total' in data.columns[-1]
        
    data2 = data.iloc[: -1 if row_flag else None, : -1 if col_flag else None]

     # # Matrice de correlation méthode Pearson et Spearman
    corr_matrix_pearson = data2.corr(numeric_only=True,method='pearson')
    corr_matrix_spearman = data2.corr(numeric_only=True,method='spearman')

    abs_corr_matrix_pearson = abs(corr_matrix_pearson)
    abs_corr_matrix_spearman = abs(corr_matrix_spearman)
    
    # vmin = 0.7
    corr_dict_pearson = [
        {
            "Indicateur_1":ind1,
            "Indicateur_2":ind2,
            "Coeff_corr_Pearson":abs_corr_matrix_pearson.loc[ind1,ind2],
            "Numéro_indicateur_1":i,
            "Numéro_indicateur_2":j
        }
        for i,ind1 in enumerate(abs_corr_matrix_pearson.columns)
        for j,ind2 in enumerate(abs_corr_matrix_pearson.columns)
        if i < j and abs_corr_matrix_pearson.loc[ind1,ind2] > vmin
    ]

    corr_dict_spearman = [
        {
            "Indicateur_1":ind1,
            "Indicateur_2":ind2,
            "Coeff_corr_Spearman":abs_corr_matrix_spearman.loc[ind1,ind2],
            "Numéro_indicateur_1":i,
            "Numéro_indicateur_2":j
        }
        for i,ind1 in enumerate(abs_corr_matrix_spearman.columns)
        for j,ind2 in enumerate(abs_corr_matrix_spearman.columns)
        if i < j and abs_corr_matrix_spearman.loc[ind1,ind2] > vmin
    ]

    corr_df_pearson = pd.DataFrame(corr_dict_pearson)
    corr_df_spearman = pd.DataFrame(corr_dict_spearman)
 ###################################### VIF#############################################
    # Calcul du VIF qui permet de déterminer les indicateurs fortement multi-correlés 
    
    ind_list = list(set(list(corr_df_pearson['Indicateur_1'].values) + list(corr_df_pearson['Indicateur_2'].values)))
    df_data = data2[ind_list]
    df_data = df_data.dropna()
    
    df_vif = pd.DataFrame()
    df_vif["Indicateur"] = df_data.columns
    df_vif["VIF"] = [variance_inflation_factor(df_data.values, i) 
                       for i in range(len(df_data.columns))]
    df_vif = df_vif[df_vif['VIF'] > vif_max].sort_values(by='VIF',ascending=False)

    return corr_df_pearson,corr_df_spearman,df_vif

######################################################################################
######################################################################################
######################################################################################

def chi2_test(data:pd.DataFrame, cat_list:list,alpha=0.05):
    '''
    Fonction réalisant le test du chi2 sur les variables catégorielles d'une liste données

    data: Dataframe
    cat_list: liste des catagories a comparer de la dataframe
    alpha: niveau de confiance (par défaut = 0.05)
    '''
    chi2_list_featEng = cat_list.copy()
    start = 1 #Décale l'indice pour la seconde boucle de variable

# 1ere variable
    for col in chi2_list_featEng:
    # 2e variable
        for i in range(start,len(chi2_list_featEng)):
            # Tableau de contingence
            table = pd.crosstab(
                data[col],
                data[chi2_list_featEng[i]]
            )
            # resultats du test
            results_chi2 = chi2_contingency(table)
            
            print(f'====================Résultats pour le couple {col} et {chi2_list_featEng[i]}=================')
            print(f'stat de test: {results_chi2[0]}')
            print(f'p-value: {results_chi2[1]}')
            print(f'degré de liberté: {results_chi2[2]}')
    
            # chi2 critique et V de Cramer si H0 est rejetée
            if results_chi2[1] < alpha:
                print("****************H0 rejetée********************")
                chi2_crit = chi2.ppf(1-alpha, results_chi2[2])
                print(f'chi2 critique: {chi2_crit}')
                # if results_chi2[1] > chi2_crit:
                #     print("****************H0 rejetée********************")
                n = table.sum().sum()
                k = min(table.shape)
                v_cramer = np.sqrt(results_chi2[0]/(n*(k-1)))
                print(f"V de Cramer: {v_cramer}")
        start +=1

######################################################################################
######################################################################################
######################################################################################

def anova_test(data:pd.DataFrame,target:None, cat_list:list,alpha=0.05):
    '''
    Fonction réalisant le test du anova sur les variables catégorielles d'une liste données

    data: Dataframe
    target: variable de comparaison
    cat_list: liste des variables a comparer de la dataframe
    alpha: niveau de confiance (par défaut = 0.05)
    '''
    anova_list_featEng = cat_list.copy()

    # Variable catégorielle
    for cat in anova_list_featEng:
        category = [data.loc[
            data[cat]==group, # mask de regroupement sur la catégorie
            target 
            ] for group in data[cat].unique()]
    
        # resultats du test
        results_anova = f_oneway(*category)
        
        print(f'====================Résultats pour le couple {cat} et {target}=================')
        print(f'stat de test: {results_anova[0]}')
        print(f'p-value: {results_anova[1]}')

        # Taille de l'effet eta2 si H0 rejetée
        if results_anova[1] < alpha:
            print("****************H0 rejetée: calcul de eta2********************")
            
            # Moyenne globale
            global_mean = data[target].mean()
            
            # SCT: variation totale
            SCT = ((data[target] - global_mean)**2).sum()
            
            # SCE: variation interclasse ==> variation au sein des groupes
            SCE=0
            # Regroupement au sein des variables categorielles
            for group in data[cat].unique():
                interclass = data.loc[data[cat]==group]
                n = len(interclass) # nombre d'element dans le groupe
                # Moyenne du groupe
                interclass_mean = interclass[target].mean()
                # Variation entre la moyenne du groupe et la moyenne globale
                SCE += n*(interclass_mean-global_mean)**2
    
            eta2 = SCE/SCT
            print(f'eta2: {eta2}')
