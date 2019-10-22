#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:36:54 2019

@author: mikaelapisanileal
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations

def get_num_components(x, threshold):
    """ Returns the number of principal components that covers the given threshold
        The scree plot is provided.

    Parameters:
        x(np.array):the standardized data to apply PCA.
        threshold(float): threshold for described variance.

    Returns:
        num_components(int): The necessary number of principal components to describe 
        at least the given threshold percentage of the variance of the data.

    """
    pca = PCA(random_state=0)
    pca.fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Scree plot')
    plt.show()
    variance = np.cumsum(pca.explained_variance_ratio_)
    variance = np.where(variance>=threshold)
    if (len(variance)==0):
        raise Exception('PCA does not explain data')
    num_compoments = variance[0][0] + 1
    return num_compoments

def generate_components(x, colnames, n_components):
    """ Applies PCA to the data x for n_components components.
        
    Parameters:
        x(np.array):the standardized data to apply PCA.
        colnames(list): list of column names for the features.
        n_components(int): number of components.
        
    Returns:
        pca(PCA): result of application of PCA.
        
    """
    pca = PCA(n_components=n_components)
    pcs = pca.fit(x)
    plt.matshow(pcs.components_,cmap='viridis')
    components_range = np.arange(1, n_components+1, 1)
    components_names = list(map(lambda x: 'PC' + str(x), components_range))
    plt.yticks(range(0,n_components), components_names,fontsize=10)
    plt.colorbar()
    plt.xticks(range(0,len(colnames)),colnames,rotation=90,ha='left')
    plt.tight_layout()
    plt.show()
    return pcs, components_names

def get_PCA(features, threshold=0.8):
    """ Returns the pca result for the given features.
      
    Parameters:
        features(DataFrame):the data to apply PCA.
        threshold(float): threshold for described variance.

    Returns:
        pca(PCA): result of application of PCA.
        new_df: result of transform.

    """
    x = StandardScaler().fit_transform(features.values)
    n_components = get_num_components(x, threshold)
    pca,compoments_names = generate_components(x,features.columns, n_components)
    new_df = pca.transform(x)
    return pca, new_df


def biplot(xs, ys, coeff, title, pc_1_name, pc_2_name, colname, df, labels=None):
    """ Draws biplot graph for the principal components given.

    Parameters:
        xs(DataFrame): column for transformed values for x axis.
        ys(DataFrame): column for transformed values  for y axis.
        coeff(list): principal components values to be ploted.
        title(str): title fot the graph.
        pc_1_name(str): name for the principal component for x axis.
        pc_2_name(str):name for the principal component for y axis.
        colname(str): target variable name.
        df(DataFrame): dataframe that contains the features and the target values.
        labels(list): list of features' names.   
    """
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel(pc_1_name)
    plt.ylabel(pc_2_name)
    plt.title(title)
    plt.grid()
    plt.scatter(xs * scalex,ys * scaley, c = df[colname])
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'red',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, str(i+1), 
                     color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], 
                     color = 'green', ha = 'center', va = 'center')
 
    plt.show()


 
def generate_biplot_plots(features, new_df, pca_indexes, pca_names, pca_components, 
                          target, target_name, labels=None):
    """ Generates biplot graphs.

    Parameters:
        features(DataFrame): features values.
        new_df(DataFrame): principal componentes values (result of transform).
        pca_indexes(list): list of indexes for principal components to be ploted.
        pca_names(list): list of names for principal components to be ploted.
        pca_components(array, shape (n_components, n_features)): result from pca.components_
        target(DataFrame): target values.
        target_name: target varaible name.
    """
    df = pd.concat([features, target], axis = 1)
    components = list(combinations(pca_indexes, 2))
    for (p1, p2) in components:
        biplot(new_df[:,p1], new_df[:,p2],
               np.transpose(np.array([pca_components[p1,:],
                                      pca_components[p2,:]])), 
               title="Biplot %s %s" %(pca_names[p1],pca_names[p2]), 
               pc_1_name = pca_names[p1],
               pc_2_name = pca_names[p2],
               colname = target_name,
               df = df,
               labels=labels)
            

def color_classes(pc_x_index, pc_y_index, pc_x_name, pc_y_name, new_df, target, 
                  cdict, labels, marker, alpha, s=40, fontsize=14):
    """ Generates scatter plot for the principal components given, 
        coloring by the target variable's class.

    Parameters:
        pc_x_index(int): index for principal component for x axis.
        pc_y_index(int): index for principal component for y axis.
        pc_x_name(str): name that represents principal component in x axis.
        pc_y_name(str): name that represents principal component in y axis.
        new_df(DataFrame): dataframe with transformed values.
        target(DataFrame): target values.
        cdict(dict): dict of colors. Example: {'A':'red','B':'green'}. 
        labels(dict): dict of labels. Example: {'A':'NameA','B':'NameB'}.  
        marker(dict): dict of markers. Example: {'A':'*','B':'o'}. 
        alpha(dict): dict desinty of dots for each label. Example: {'A':.5, 'B':.5}.
        s(int): size of dots.
    """
    Xax=new_df[:,pc_x_index]
    Yax=new_df[:,pc_y_index]
    fig,ax=plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')
    for l in np.unique(target):
     ix=np.where(target==l)
     ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=s,label=labels[l],
                marker=marker[l],alpha=alpha[l])

    plt.xlabel(pc_x_name,fontsize=fontsize)
    plt.ylabel(pc_y_name,fontsize=fontsize)
    plt.legend()
    plt.show()
   

def generate_color_classes_plots(pc_indexes, pc_names, new_df, target, 
                                 cdict, labels, marker, alpha, s=40, fontsize=14):
    
    """ Generates color_classess plots for all the combinations of principal
         componentes provided.
     
     Parameters:
         pc_x_index(int): index for principal component for x axis
         pc_y_index(int): index for principal component for y axis
         pc_x_name(str): name that represents principal component in x axis
         pc_y_name(str): name that represents principal component in y axis
         new_df(DataFrame): dataframe with transformed values
         target(DataFrame): target values
         cdict(dict): dict of colors. Example: {'A':'red','B':'green'} 
         labels(dict): dict of labels. Example: {'A':'NameA','B':'NameB'}  
         marker(dict): dict of markers. Example: {'A':'*','B':'o'} 
         alpha(dict): dict desinty of dots for each label. Example: {'A':.5, 'B':.5} 
         s(int): size of dots
        
     """
    components = list(combinations(pc_indexes, 2))
    for (p1, p2) in components:
        color_classes(p1, p2, pc_names[p1], pc_names[p2], new_df, target, 
                      cdict, labels, marker, alpha, s, fontsize)
        
        
