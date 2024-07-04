import numpy as np 
import pandas as pd
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pymupdf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def extract_abstract(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(3): 
        page = doc.load_page(page_num)
        text += page.get_text()
    
    abstract_start = re.search(r'abstract|A B S T R A C T|ABSTRACT|Iodide redox|Discovery of|Water electrolysis', text, re.IGNORECASE)
    if not abstract_start:
        return None
    
    start_index = abstract_start.end()
    end_markers = [r'©',r'\x01',r'\*', r'\b Index Terms:', r'\b1\.\s*Introduction\b', r'\bIntroduction\b','\bNomenclature\b']
    end_index = len(text)
    
    for marker in end_markers:
        match = re.search(marker, text[start_index:], re.IGNORECASE)
        if match:
            end_index = min(end_index, start_index + match.start())
    
    abstract = text[start_index:end_index].strip()
    abstract = re.sub(r'\s+', ' ', abstract)
    
    if len(abstract)> 1:
        return abstract
    else:
        return None
    
def get_pdf_title(filepath):
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f, strict=False)
        if reader.isEncrypted:
            try:
                reader.decrypt('')
            except:
                return None 
        info = reader.getDocumentInfo()
        if info is not None:
            title = info.title
            return title if title is not None else None
        return None

def normalise_name(s):
    s = str(s)
    s = re.sub(r'[\s:"\'\-_&#x2013;\u2010-\u2015\u201C-\u201F\u2018-\u201B&nbsp;]', '', s)
    s = s.lower()  
    return s

def preprocess_text(text):
    if pd.isna(text):
        return None
    
    text = text.lower()
    text = re.sub(r'[/\-]', ' ', text)
    text = re.sub(r'abstract',' ',text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\b(?!\d{4}\b)\d+\b', '', text)
    text = re.sub(r'[^\w\s\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    tokens = [token for token in tokens if token not in stop_words]
    
    processed_tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]
    
    return ' '.join(processed_tokens)

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return vectors,feature_names

def silhouette_method(data, max_k):
    iters = range(2, max_k + 1)
    silhouette_scores = []
    
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
        
    plt.figure(figsize=(10, 6))
    plt.plot(iters, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()


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

def generate_report(optimal_k, clean_df, n_terms, feature_names,vectors):
    cluster_terms = {}
    for cluster_num in range(optimal_k):
        cluster_indices = np.where(clean_df['cluster'] == cluster_num)[0]
        cluster_tfidf_sum = np.asarray(vectors[cluster_indices].sum(axis=0)).flatten()
        top_terms_indices = np.argsort(cluster_tfidf_sum)[::-1][:n_terms]
        top_terms = [feature_names[i] for i in top_terms_indices]
        cluster_terms[cluster_num] = top_terms

    papers_in_cluster = clean_df['cluster'].value_counts().sort_index()
    print(f'No. clusters: {optimal_k}')
    summary_report = f'No. Clusters: {optimal_k}\n\n'

    summary_report += 'No. papers per cluster:\n'
    summary_report += papers_in_cluster.to_string() + '\n\n'
    print(f'No. papers per cluster: {papers_in_cluster}')
    summary_report += f'Top {n_terms} terms in each cluster:\n'
    print(f'Top {n_terms} terms in each cluster')
    for cluster_num, terms in cluster_terms.items():
        print(f'Cluster {cluster_num}: {", ".join(terms)}')
        summary_report += f'Cluster {cluster_num}: {", ".join(terms)}\n'

    with open('summary.txt', 'w') as file:
        file.write(summary_report)


