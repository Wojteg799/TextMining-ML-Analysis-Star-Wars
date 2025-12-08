# -*- coding: utf-8 -*-
"""Topic modeling using NMF (Non-negative Matrix Factorization)."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from .config import N_TOPICS, RANDOM_STATE


def analyze_character_topics(character_terms_advanced, top_characters):
    """
    Perform topic modeling on character dialogues using NMF.
    
    Args:
        character_terms_advanced: Dictionary mapping character names to processed terms
        top_characters: Dictionary of top N characters (name -> word_count)
        
    Returns:
        Dictionary containing:
        - nmf_output: NMF transformation output
        - vectorizer: Fitted TfidfVectorizer
        - nmf_model: Fitted NMF model
        - feature_names: Feature names from vectorizer
    """
    character_corpus_nmf = []
    for character in top_characters.keys():
        text = ' '.join(character_terms_advanced[character])
        character_corpus_nmf.append(text)
    
    vectorizer_nmf = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)
    X_nmf = vectorizer_nmf.fit_transform(character_corpus_nmf)
    
    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE, l1_ratio=0.5)
    nmf_output = nmf.fit_transform(X_nmf)
    
    feature_names = vectorizer_nmf.get_feature_names_out()
    
    return {
        'nmf_output': nmf_output,
        'vectorizer': vectorizer_nmf,
        'nmf_model': nmf,
        'feature_names': feature_names
    }


def prepare_topic_visualization_data(nmf_output, top_characters):
    """
    Prepare data for topic distribution visualization.
    
    Args:
        nmf_output: NMF transformation output
        top_characters: Dictionary of top N characters
        
    Returns:
        List of dictionaries with 'Character', 'Topic', 'Share'
    """
    topics = []
    for i, character in enumerate(top_characters.keys()):
        for j in range(N_TOPICS):
            topics.append({
                'Character': character,
                'Topic': f'Topic {j+1}',
                'Share': nmf_output[i][j]
            })
    return topics


def print_topic_keywords(nmf_model, feature_names, n_words=10):
    """
    Print top words for each topic.
    
    Args:
        nmf_model: Fitted NMF model
        feature_names: Feature names from vectorizer
        n_words: Number of top words to print per topic
    """
    print("\nTop keywords for each topic:")
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
        print(f"\nTopic {topic_idx + 1}:")
        print(", ".join(top_words))
