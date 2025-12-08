# -*- coding: utf-8 -*-
"""Clustering and similarity analysis for character dialogues."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from .config import OUTPUT_DIR, N_CLUSTERS, FIG_SIZE_LARGE


def calculate_character_similarity(character_sentences, top_characters):
    """
    Calculate cosine similarity between character dialogues.
    
    Args:
        character_sentences: Dictionary mapping character names to their sentences
        top_characters: Dictionary of top N characters
        
    Returns:
        Dictionary containing:
        - similarity_matrix: Cosine similarity matrix
        - vectorizer: Fitted TfidfVectorizer
    """
    character_corpus = []
    for character in top_characters.keys():
        text = ' '.join(character_sentences[character])
        character_corpus.append(text)
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(character_corpus)
    
    similarity = cosine_similarity(X)
    
    return {
        'similarity_matrix': similarity,
        'vectorizer': vectorizer
    }


def create_dendrogram(similarity, labels, title, filename):
    """
    Create and save dendrogram.
    
    Args:
        similarity: Similarity matrix
        labels: List of labels for characters
        title: Title for the plot
        filename: Output filename (without extension)
    """
    plt.figure(figsize=FIG_SIZE_LARGE)
    linkage_matrix = linkage(similarity, method='ward')
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def cluster_characters(similarity, top_characters):
    """
    Perform agglomerative clustering on characters.
    
    Args:
        similarity: Similarity matrix
        top_characters: Dictionary of top N characters
        
    Returns:
        List of dictionaries with 'Character', 'Cluster'
    """
    clustering = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric='cosine',
        linkage='average'
    )
    clusters = clustering.fit_predict(similarity)
    
    cluster_data = []
    for i, character in enumerate(top_characters.keys()):
        cluster_data.append({
            'Character': character,
            'Cluster': f'Cluster {clusters[i] + 1}'
        })
    
    return cluster_data
