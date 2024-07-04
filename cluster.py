# %%
import numpy as np 
import pandas as pd
import PyPDF2
from urllib.parse import urlparse
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
import fitz
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from helper import *
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
df = pd.read_csv('data.csv')
df.head

df['Abstract'] = None
def extract_all_abstracts(directory_path):
    abstracts = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_name = get_pdf_title(file_path)
        file_name_normal = normalise_name(file_name)
        for n in df['Name']:
            base = normalise_name(n)
            if base == file_name_normal:
                name = n
                break
        abstract = extract_abstract(file_path)
        if abstract:
            df.loc[df['Name'] == name, 'Abstract'] = abstract
        else:
            abstracts[filename] = None
    return abstracts

abstract_text = extract_all_abstracts('data')

new_df = df[['Name','Abstract']]

new_df = new_df.dropna(subset=['Abstract'])
new_df = new_df.reset_index(drop=True)

new_df['processed_abstract'] = new_df['Abstract'].apply(preprocess_text)
clean_df = new_df.drop_duplicates()

vectors,feature_names = vectorize_texts(clean_df['processed_abstract'])

# %%
pca = PCA(n_components=2) 
reduced_vectors = pca.fit_transform(vectors.toarray())

# %%
hi_relation = linkage(reduced_vectors, method='average')
max_d = 0.26
clusters = fcluster(hi_relation, max_d, criterion='distance')
print(np.unique(clusters)) 

# %%
def find_optimal_clusters(data, max_k):
    i = range(1, max_k + 1)
    sse = []
    
    for k in i:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(i, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

find_optimal_clusters(reduced_vectors, 70)

def k_means(optimal_k):
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    km.fit(reduced_vectors)
    sil_score = silhouette_score(reduced_vectors, km.labels_)
    db_score = davies_bouldin_score(reduced_vectors, km.labels_)
    clean_df.loc[:, 'cluster'] = km.labels_
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=km.labels_, cmap='viridis')
    plt.title(f'k-means Clusters (k={optimal_k})')
    plt.colorbar(scatter)
    plt.show()
    return sil_score, db_score

optimal_k = 3
silhouette_score, db_score = k_means(optimal_k)
print(f'Silhouette Score for k={optimal_k}: {silhouette_score:.2f}')
print(f'DB score for k={optimal_k}: {db_score:.2f}')

clean_df

clean_df.to_csv('clustering_results.csv', index=False)

generate_report(optimal_k, clean_df, 10, feature_names,vectors)




