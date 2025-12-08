# -*- coding: utf-8 -*-
"""Character classification models using various vectorization methods."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import spacy
from .config import OUTPUT_DIR, TEST_SIZE, RANDOM_STATE, FIG_SIZE_CONFUSION
from .preprocessing import preprocess_text_advanced


def sentence_to_vector(sentence, nlp):
    """
    Convert sentence to vector using spaCy.
    
    Args:
        sentence: Input sentence
        nlp: spaCy language model
        
    Returns:
        Vector representation
    """
    doc = nlp(sentence)
    if len(doc) > 0:
        return doc.vector
    return np.zeros(96)


def evaluate_classifier(y_true, y_pred, y_prob=None):
    """
    Evaluate classifier performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    return results


def train_binary_classifier(X_train, X_test, y_train, y_test, top_characters):
    """
    Train classifier with binary vectorization.
    
    Args:
        X_train: Training sentences
        X_test: Test sentences
        y_train: Training labels
        y_test: Test labels
        top_characters: Dictionary of top characters
        
    Returns:
        Dictionary with classifier, vectorizer, predictions, and results
    """
    vectorizer = CountVectorizer(binary=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    results = evaluate_classifier(y_test, y_pred)
    
    return {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'y_pred': y_pred,
        'results': results
    }


def train_logarithmic_classifier(X_train, X_test, y_train, y_test, top_characters):
    """
    Train classifier with logarithmic vectorization.
    
    Args:
        X_train: Training sentences
        X_test: Test sentences
        y_train: Training labels
        y_test: Test labels
        top_characters: Dictionary of top characters
        
    Returns:
        Dictionary with classifier, vectorizer, predictions, and results
    """
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    X_train_vec = np.log1p(X_train_vec.toarray())
    X_test_vec = np.log1p(X_test_vec.toarray())
    
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    results = evaluate_classifier(y_test, y_pred)
    
    return {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'y_pred': y_pred,
        'results': results
    }


def train_tfidf_classifier(X_train, X_test, y_train, y_test, top_characters):
    """
    Train classifier with TF-IDF vectorization.
    
    Args:
        X_train: Training sentences
        X_test: Test sentences
        y_train: Training labels
        y_test: Test labels
        top_characters: Dictionary of top characters
        
    Returns:
        Dictionary with classifier, vectorizer, predictions, and results
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    results = evaluate_classifier(y_test, y_pred)
    
    return {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'y_pred': y_pred,
        'results': results
    }


def train_word_embeddings_classifier(X_train, X_test, y_train, y_test, top_characters, nlp):
    """
    Train classifier with word embeddings (spaCy).
    
    Args:
        X_train: Training sentences
        X_test: Test sentences
        y_train: Training labels
        y_test: Test labels
        top_characters: Dictionary of top characters
        nlp: spaCy language model
        
    Returns:
        Dictionary with classifier, vectorizer (nlp), predictions, and results
    """
    X_train_vec = np.array([sentence_to_vector(sent, nlp) for sent in X_train])
    X_test_vec = np.array([sentence_to_vector(sent, nlp) for sent in X_test])
    
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    results = evaluate_classifier(y_test, y_pred)
    
    return {
        'classifier': classifier,
        'vectorizer': nlp,
        'y_pred': y_pred,
        'results': results
    }


def plot_confusion_matrix(y_test, y_pred, top_characters, title, filename):
    """
    Create and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        top_characters: Dictionary of top characters
        title: Plot title
        filename: Output filename (without extension)
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=FIG_SIZE_CONFUSION)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=top_characters.keys(),
        yticklabels=top_characters.keys()
    )
    plt.title(title)
    plt.xlabel('Predicted Character')
    plt.ylabel('Actual Character')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def prepare_classification_data(character_sentences, top_characters):
    """
    Prepare data for classification.
    
    Args:
        character_sentences: Dictionary mapping character names to their sentences
        top_characters: Dictionary of top N characters
        
    Returns:
        Tuple of (X, y) where X is list of processed sentences and y is labels
    """
    X = []
    y = []
    
    for character in top_characters.keys():
        for sentence in character_sentences[character]:
            processed_sentence = ' '.join(preprocess_text_advanced(sentence))
            if processed_sentence:
                X.append(processed_sentence)
                y.append(character)
    
    return X, y


def predict_character(text, classifier, vectorizer, best_method, top_characters):
    """
    Predict which character said a given text.
    
    Args:
        text: Input text
        classifier: Trained classifier
        vectorizer: Vectorizer or nlp model
        best_method: Name of the best method ('Binary', 'Logarithmic', 'TF-IDF', 'WordEmbeddings')
        top_characters: Dictionary of top characters
        
    Returns:
        Dictionary mapping character names to probabilities
    """
    processed_text = ' '.join(preprocess_text_advanced(text))
    
    if best_method == 'WordEmbeddings':
        X_new = sentence_to_vector(processed_text, vectorizer)
        X_new = X_new.reshape(1, -1)
    else:
        X_new = vectorizer.transform([processed_text])
    
    probabilities = classifier.predict_proba(X_new)[0]
    results = dict(zip(classifier.classes_, probabilities))
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return results

