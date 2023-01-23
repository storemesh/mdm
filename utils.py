import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn import metrics
import hdbscan
import time
import umap
import umap.plot


def prepare_dataset():
    df = pd.read_parquet('dataset/semantic_similarity/hscode.parquet')
    df_hscode2022 = df[df['HSYear'] == 2022]
    df_hscode4_th = df_hscode2022[['thdescriptions4','HSCode2']]
    df_hscode4_th = df_hscode4_th.rename(columns={'thdescriptions4': 'hscode4_text'})

    df_hscode4_en = df_hscode2022[['endescriptions4','HSCode2']]
    df_hscode4_en = df_hscode4_en.rename(columns={'endescriptions4': 'hscode4_text'})

    df_hscode4 = pd.concat([
        df_hscode4_th,
        df_hscode4_en
    ])
    df_hscode4 = df_hscode4.drop_duplicates()
    df_hscode4 = df_hscode4[df_hscode4['hscode4_text'].notnull()]
    
    return df_hscode4
    

def find_accuracy(series):    
    n = 2
    count_series = series.value_counts()
    sum_all = count_series.sum()
    count_series_no_noise = count_series[count_series.index != -1]
    sum_top_n = count_series_no_noise.iloc[:n].sum()
    accuracy = sum_top_n / sum_all
    
    return accuracy


def calculate_all_metric(df_hscode4, X):
    acc_each_hscode = df_hscode4.groupby('HSCode2')['cluster_group_id'].apply(find_accuracy)
    acc_mean = acc_each_hscode.mean()
    
    df_result = df_hscode4.copy()
    count_cluster = df_result['cluster_group_id'].value_counts() 
    
    n_each_cluster = count_cluster[count_cluster.index != -1]
    cluster_size = n_each_cluster.shape[0]
    n_top_20_cluster = n_each_cluster.iloc[:20].values
    
    n_noise = count_cluster[count_cluster.index == -1].sum()
    
    if cluster_size == 0:
        silhouette = 0.0
    else:
        silhouette = metrics.silhouette_score(X, df_hscode4['cluster_group_id'].values, metric='euclidean')
    
    dict_result = {
        'acc_mean': acc_mean,
        'silhouette': silhouette,
        'cluster_size': cluster_size,
        'n_noise': n_noise,
        'n_top_20_cluster': n_top_20_cluster,
    }
    
    return dict_result, df_result