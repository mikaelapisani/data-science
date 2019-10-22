#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:45:37 2019

@author: mikaelapisanileal
"""

import pandas as pd
import process_PCA as p

#Download data from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/download
data = pd.read_csv('data.csv')
data.drop(['id','Unnamed: 32'],axis=1, inplace=True)
#Target variable: Benign=B / Malignant=M
features = data.drop('diagnosis', axis=1)
target = data['diagnosis']
pc_names={0:'PC1', 1:'PC2', 2:'PC3', 3:'PC4', 4:'PC5'}
#PCA

pca, new_df = p.get_PCA(features)

p.generate_biplot_plots(features, new_df, [0,1,4], pc_names, pca.components_, target, 'diagnosis', features.columns)
    
cdict={'M':'red','B':'green'} #colors
labels={'M':'Malignant','B':'Benign'} #categories 
marker={'M':'*','B':'o'} #markers
alpha={'M':.5, 'B':.5} #desinty
s=40 #size
#draw all graphs: p.generate_color_classes_plots(list(range(0,new_df.shape[1])), pc_names, new_df, target, cdict, labels, marker, alpha, s)
#draw specific principal componentes
p.generate_color_classes_plots([0,3,1], pc_names, 
                               new_df, target, cdict, labels, marker, alpha, s)
