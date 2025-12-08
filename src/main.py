# -*- coding: utf-8 -*-
"""Main script for Star Wars Sentiment Analysis project."""

import warnings
import nltk
import spacy
import pandas as pd
from .config import (
    NLTK_DOWNLOADS,
    TOP_N_CHARACTERS_STATS,
    TOP_N_CHARACTERS_ANALYSIS,
    TOP_N_CHARACTERS_CLUSTERING,
    FILM_NAMES
)
from .data_loading import prepare_data
from .statistics import (
    calculate_character_word_counts,
    get_top_characters,
    prepare_comparative_data,
    prepare_sentence_length_data,
    find_unique_words,
    print_film_statistics
)
from .visualization import (
    create_wordcloud,
    create_wordcloud_from_frequencies,
    plot_top_characters_bar,
    plot_comparative_characters,
    plot_sentence_lengths,
    plot_topic_distribution,
    plot_clusters,
    plot_emotions
)
from .topic_modeling import (
    analyze_character_topics,
    prepare_topic_visualization_data,
    print_topic_keywords
)
from .clustering import (
    calculate_character_similarity,
    create_dendrogram,
    cluster_characters
)
from .classification import (
    prepare_classification_data,
    train_binary_classifier,
    train_logarithmic_classifier,
    train_tfidf_classifier,
    train_word_embeddings_classifier,
    plot_confusion_matrix,
    predict_character
)
from .sentiment_analysis import (
    analyze_film_sentiments,
    analyze_character_sentiments,
    plot_sentiment_boxplot_films,
    plot_sentiment_boxplot_characters,
    plot_mean_sentiment_comparison,
    plot_sentiment_evolution,
    analyze_character_emotions,
    prepare_emotion_visualization_data,
    plot_emotion_heatmap
)
from .preprocessing import preprocess_text_basic

warnings.filterwarnings('ignore')


def download_nltk_data():
    """Download required NLTK data."""
    for item in NLTK_DOWNLOADS:
        try:
            nltk.download(item, quiet=True)
        except:
            pass


def main():
    """Main execution function."""
    print("=" * 80)
    print("Star Wars Sentiment Analysis - Text Mining and ML Analysis")
    print("=" * 80)
    
    # Download NLTK data
    print("\n[1/10] Downloading NLTK data...")
    try:
        download_nltk_data()
    except Exception as e:
        print(f"Warning: NLTK download failed: {e}")
        print("Continuing anyway...")
    
    # Load and prepare data
    print("[2/10] Loading and preparing data...")
    try:
        data = prepare_data()
    except FileNotFoundError as e:
        print(f"Error: Data files not found. Please ensure script files are in data/processed/ directory.")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    character_dialogues_by_film = data['character_dialogues_by_film']
    film_sentences = data['film_sentences']
    film_terms_basic = data['film_terms_basic']
    film_terms_advanced = data['film_terms_advanced']
    character_sentences = data['character_sentences']
    character_terms_basic = data['character_terms_basic']
    character_terms_advanced = data['character_terms_advanced']
    
    # Print basic statistics
    print("[3/10] Calculating basic statistics...")
    print_film_statistics(character_dialogues_by_film, film_sentences)
    
    print("\nNumber of terms in each film:")
    for film, terms in film_terms_basic.items():
        print(f"{film}: {len(terms)} terms")
    
    # Calculate character word counts
    character_words = calculate_character_word_counts(character_sentences)
    top_5_characters = get_top_characters(character_words, TOP_N_CHARACTERS_STATS)
    top_7_characters = get_top_characters(character_words, TOP_N_CHARACTERS_ANALYSIS)
    top_10_characters = get_top_characters(character_words, TOP_N_CHARACTERS_CLUSTERING)
    
    # Visualizations - Top characters per film
    print("\n[4/10] Creating visualizations - Top characters...")
    films_dict = {
        'nowa_nadzieja': FILM_NAMES['nowa_nadzieja'],
        'imperium_kontratakuje': FILM_NAMES['imperium_kontratakuje'],
        'powrot_jedi': FILM_NAMES['powrot_jedi']
    }
    
    for film_key, film_name in films_dict.items():
        film_character_words = {}
        for character, sentences in character_dialogues_by_film[film_key].items():
            all_sentences_text = ' '.join(sentences)
            terms = preprocess_text_basic(all_sentences_text)
            film_character_words[character] = len(terms)
        
        plot_top_characters_bar(
            film_character_words,
            f'Top 5 Characters by Word Count - {film_name}',
            f'top_5_characters_{film_key}',
            top_n=5
        )
    
    # Comparative visualization
    comparative_data = prepare_comparative_data(
        character_dialogues_by_film,
        top_5_characters,
        films_dict
    )
    plot_comparative_characters(
        comparative_data,
        'Comparison of Word Counts for Top 5 Characters Across Films',
        'character_comparison_across_films'
    )
    
    # Sentence length analysis
    length_data = prepare_sentence_length_data(character_sentences, top_5_characters)
    plot_sentence_lengths(
        length_data,
        'Sentence Length Distribution for Top 5 Characters',
        'sentence_length_distribution'
    )
    
    # Word clouds for films
    print("\n[5/10] Creating word clouds for films...")
    for film_key, film_name in films_dict.items():
        create_wordcloud(
            film_terms_advanced[film_key],
            f'Word Cloud - {film_name}',
            f'wordcloud_{film_key}'
        )
    
    # Word clouds for top characters
    print("[6/10] Creating word clouds for top characters...")
    for character in top_5_characters.keys():
        create_wordcloud(
            character_terms_advanced[character],
            f'Word Cloud - {character}',
            f'wordcloud_{character}'
        )
    
    # Unique words analysis
    unique_words_films = find_unique_words(film_terms_advanced)
    for film_key, unique in unique_words_films.items():
        if unique:
            create_wordcloud_from_frequencies(
                unique,
                f'Unique Words - {films_dict[film_key]}',
                f'unique_words_{film_key}'
            )
    
    # Topic modeling
    print("\n[7/10] Performing topic modeling...")
    topic_results = analyze_character_topics(character_terms_advanced, top_7_characters)
    topics_data = prepare_topic_visualization_data(
        topic_results['nmf_output'],
        top_7_characters
    )
    plot_topic_distribution(
        topics_data,
        'Topic Distribution in Dialogues of 7 Characters with Most Words',
        'topic_distribution'
    )
    print_topic_keywords(
        topic_results['nmf_model'],
        topic_results['feature_names']
    )
    
    # Clustering
    print("\n[8/10] Performing clustering analysis...")
    similarity_results = calculate_character_similarity(character_sentences, top_10_characters)
    create_dendrogram(
        similarity_results['similarity_matrix'],
        list(top_10_characters.keys()),
        'Dialogue Similarity Dendrogram for Top 10 Characters',
        'character_dendrogram'
    )
    
    cluster_data = cluster_characters(
        similarity_results['similarity_matrix'],
        top_10_characters
    )
    plot_clusters(
        cluster_data,
        'Character Clustering Based on Dialogue Similarity',
        'character_clusters'
    )
    
    # Classification
    print("\n[9/10] Training classification models...")
    X, y = prepare_classification_data(character_sentences, top_7_characters)
    
    from sklearn.model_selection import train_test_split
    from .config import TEST_SIZE, RANDOM_STATE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Train all classifiers
    binary_results = train_binary_classifier(
        X_train, X_test, y_train, y_test, top_7_characters
    )
    print("\nBinary Vectorization Results:")
    print(f"Accuracy: {binary_results['results']['accuracy']:.2%}")
    plot_confusion_matrix(
        y_test,
        binary_results['y_pred'],
        top_7_characters,
        'Confusion Matrix - Binary Model',
        'confusion_matrix_binary'
    )
    
    log_results = train_logarithmic_classifier(
        X_train, X_test, y_train, y_test, top_7_characters
    )
    print("\nLogarithmic Vectorization Results:")
    print(f"Accuracy: {log_results['results']['accuracy']:.2%}")
    plot_confusion_matrix(
        y_test,
        log_results['y_pred'],
        top_7_characters,
        'Confusion Matrix - Logarithmic Model',
        'confusion_matrix_logarithmic'
    )
    
    tfidf_results = train_tfidf_classifier(
        X_train, X_test, y_train, y_test, top_7_characters
    )
    print("\nTF-IDF Vectorization Results:")
    print(f"Accuracy: {tfidf_results['results']['accuracy']:.2%}")
    plot_confusion_matrix(
        y_test,
        tfidf_results['y_pred'],
        top_7_characters,
        'Confusion Matrix - TF-IDF Model',
        'confusion_matrix_tfidf'
    )
    
    # Load spaCy model for word embeddings
    try:
        nlp = spacy.load('en_core_web_sm')
        w2v_results = train_word_embeddings_classifier(
            X_train, X_test, y_train, y_test, top_7_characters, nlp
        )
        print("\nWord Embeddings Results:")
        print(f"Accuracy: {w2v_results['results']['accuracy']:.2%}")
        plot_confusion_matrix(
            y_test,
            w2v_results['y_pred'],
            top_7_characters,
            'Confusion Matrix - Word Embeddings Model',
            'confusion_matrix_word_embeddings'
        )
    except OSError:
        print("\nWord Embeddings: spaCy model not found, skipping...")
        w2v_results = None
    
    # Compare models
    all_results = {
        'Binary': binary_results['results'],
        'Logarithmic': log_results['results'],
        'TF-IDF': tfidf_results['results']
    }
    if w2v_results:
        all_results['WordEmbeddings'] = w2v_results['results']
    
    results_df = pd.DataFrame(all_results).T
    print("\n" + "=" * 80)
    print("Model Comparison:")
    print("=" * 80)
    print(results_df)
    
    # Find best method
    best_method = max(all_results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest method: {best_method}")
    
    # Sentiment analysis
    print("\n[10/10] Performing sentiment analysis...")
    film_sentiments = analyze_film_sentiments(character_dialogues_by_film)
    character_sentiments = analyze_character_sentiments(character_sentences, top_7_characters)
    
    plot_sentiment_boxplot_films(
        film_sentiments,
        'Sentiment Distribution in Dialogues for Each Film',
        'sentiment_films'
    )
    
    plot_sentiment_boxplot_characters(
        character_sentiments,
        'Sentiment Distribution in Dialogues for Main Characters',
        'sentiment_characters'
    )
    
    plot_mean_sentiment_comparison(
        film_sentiments,
        character_sentiments,
        'Mean Sentiment Comparison',
        'mean_sentiment_comparison'
    )
    
    plot_sentiment_evolution(
        film_sentiments,
        'Sentiment Evolution in Films',
        'sentiment_evolution'
    )
    
    # Emotion analysis
    character_emotions = analyze_character_emotions(character_sentences, top_7_characters)
    emotions_data = prepare_emotion_visualization_data(character_emotions)
    plot_emotions(
        emotions_data,
        'Emotion Distribution in Dialogues of 7 Characters with Most Words',
        'emotions_characters'
    )
    plot_emotion_heatmap(
        emotions_data,
        'Emotion Heatmap in Character Dialogues',
        'emotions_heatmap'
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete! All outputs saved to 'output/' directory.")
    print("=" * 80)


if __name__ == '__main__':
    main()

