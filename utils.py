import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn import metrics
import torch
import hdbscan
import time
import umap
import umap.plot

def prepare_dataset_en():
    df = pd.read_csv('dataset/semantic_similarity/hscode4_en_translate.csv', dtype={'HSCode2': 'string'})
    df = df.drop(columns="hscode4_text")
    df = df.rename(columns={'TranslatedText': 'hscode4_text'})
    df['HSCode2'] = df['HSCode2'].str.zfill(2)
    return df
    
def find_spearman_pearson(X, labels):
    cos_sim = util.cos_sim(X, X)
    dot_products = util.dot_score(X,X)

    cosine_scores = torch.reshape(cos_sim, (labels.shape[0], ))
    dot_products = torch.reshape(dot_products, (labels.shape[0], ))
    
    pearson_cosine, _ = pearsonr(labels, cosine_scores)
    spearman_cosine, _ = spearmanr(labels, cosine_scores)

    pearson_dot, _ = pearsonr(labels, dot_products)
    spearman_dot, _ = spearmanr(labels, dot_products)
    
    result = {
        'spearman_cosine': spearman_cosine,
        'pearson_cosine': pearson_cosine,
        'spearman_dot': spearman_dot,
        'pearson_dot': pearson_dot,        
    }
    
    return result