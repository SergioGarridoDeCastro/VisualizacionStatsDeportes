import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def model_shot_creators(data):
    # Preparar los datos
    data['Player'] = data['Player'].str.split('\\', expand=True)[0]
    data['Nation'] = data['Nation'].str.split(' ', expand=True)[1]
    data['Pos'] = data['Pos'].str[:2]

    add_list = ['Pass SCA', 'Deadball SCA', 'Dribble SCA', 'Shot SCA', 'Fouled SCA']
    data['Sum SCA'] = data[add_list].sum(axis=1)
    data['Pass SCA Ratio'] = data['Pass SCA']/data['Sum SCA']
    new_cols_list = [each + ' Ratio' for each in add_list]

    for idx, val in enumerate(new_cols_list):
        data[val] = data[add_list[idx]]/data['Sum SCA']
    data['Sum SCA Ratio'] = data[new_cols_list].sum(axis=1)
    
    data_mffw = data[((data['Pos'] == 'FW') | (data['Pos'] == 'MF')) & (data['90s'] > 5) & (data['SCA'] > 15)]

    km = KMeans(n_clusters=5, init='random', random_state=0)
    y_km = km.fit_predict(data_mffw[new_cols_list])
    data_mffw['Cluster'] = y_km


    def plotClusters(xAxis, yAxis):
        plt.scatter(data_mffw[data_mffw['Cluster']==0][xAxis], data_mffw[data_mffw['Cluster']==0][yAxis], s=40, c='red', label ='Cluster 1')
        plt.scatter(data_mffw[data_mffw['Cluster']==1][xAxis], data_mffw[data_mffw['Cluster']==1][yAxis], s=40, c='blue', label ='Cluster 2')
        plt.scatter(data_mffw[data_mffw['Cluster']==2][xAxis], data_mffw[data_mffw['Cluster']==2][yAxis], s=40, c='green', label ='Cluster 3')
        plt.scatter(data_mffw[data_mffw['Cluster']==3][xAxis], data_mffw[data_mffw['Cluster']==3][yAxis], s=40, c='pink', label ='Cluster 4')
        plt.scatter(data_mffw[data_mffw['Cluster']==4][xAxis], data_mffw[data_mffw['Cluster']==4][yAxis], s=40, c='gold', label ='Cluster 5')
        plt.xlabel(xAxis)
        plt.ylabel(yAxis)    
        plt.legend() 

    plotClusters('Pass SCA Ratio', 'Dribble SCA Ratio')
    plotClusters('SCA90', 'Age')