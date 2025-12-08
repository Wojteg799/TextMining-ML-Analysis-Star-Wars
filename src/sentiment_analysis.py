# -*- coding: utf-8 -*-
"""Sentiment and emotion analysis for Star Wars dialogues."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from textblob import TextBlob
from nrclex import NRCLex
from .config import OUTPUT_DIR, FIG_SIZE_LARGE


def analyze_sentiment(text):
    """
    Analyze sentiment of a text using TextBlob.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment polarity (-1 to 1)
    """
    return TextBlob(text).sentiment.polarity


def analyze_film_sentiments(character_dialogues_by_film):
    """
    Analyze sentiment for each film.
    
    Args:
        character_dialogues_by_film: Dictionary mapping film names to character dialogues
        
    Returns:
        Dictionary mapping film names to sentiment statistics
    """
    film_sentiments = {}
    for film, dialogues in character_dialogues_by_film.items():
        all_sentiments = []
        for character_dialogues in dialogues.values():
            for sentence in character_dialogues:
                sentiment = analyze_sentiment(sentence)
                all_sentiments.append(sentiment)
        film_sentiments[film] = {
            'mean': np.mean(all_sentiments),
            'median': np.median(all_sentiments),
            'std': np.std(all_sentiments),
            'all': all_sentiments
        }
    return film_sentiments


def analyze_character_sentiments(character_sentences, top_characters):
    """
    Analyze sentiment for top characters.
    
    Args:
        character_sentences: Dictionary mapping character names to their sentences
        top_characters: Dictionary of top N characters
        
    Returns:
        Dictionary mapping character names to sentiment statistics
    """
    character_sentiments = {}
    for character in top_characters.keys():
        all_sentiments = []
        for sentence in character_sentences[character]:
            sentiment = analyze_sentiment(sentence)
            all_sentiments.append(sentiment)
        character_sentiments[character] = {
            'mean': np.mean(all_sentiments),
            'median': np.median(all_sentiments),
            'std': np.std(all_sentiments),
            'all': all_sentiments
        }
    return character_sentiments


def plot_sentiment_boxplot_films(film_sentiments, title, filename):
    """
    Create box plot for sentiment distribution across films.
    
    Args:
        film_sentiments: Dictionary mapping film names to sentiment statistics
        title: Plot title
        filename: Output filename (without extension)
    """
    fig = go.Figure()
    for film, data in film_sentiments.items():
        fig.add_trace(go.Box(
            y=data['all'],
            name=film,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title='Sentiment Polarity (-1 to 1)',
        showlegend=True,
        height=600
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_sentiment_boxplot_characters(character_sentiments, title, filename):
    """
    Create box plot for sentiment distribution across characters.
    
    Args:
        character_sentiments: Dictionary mapping character names to sentiment statistics
        title: Plot title
        filename: Output filename (without extension)
    """
    fig = go.Figure()
    for character, data in character_sentiments.items():
        fig.add_trace(go.Box(
            y=data['all'],
            name=character,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title='Sentiment Polarity (-1 to 1)',
        showlegend=True,
        height=600
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_mean_sentiment_comparison(film_sentiments, character_sentiments, title, filename):
    """
    Create bar chart comparing mean sentiment.
    
    Args:
        film_sentiments: Dictionary mapping film names to sentiment statistics
        character_sentiments: Dictionary mapping character names to sentiment statistics
        title: Plot title
        filename: Output filename (without extension)
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Mean Sentiment in Films', 'Mean Sentiment by Character')
    )
    
    fig.add_trace(
        go.Bar(
            x=list(film_sentiments.keys()),
            y=[data['mean'] for data in film_sentiments.values()],
            name='Films'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=list(character_sentiments.keys()),
            y=[data['mean'] for data in character_sentiments.values()],
            name='Characters'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=title,
        height=600,
        showlegend=False
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_sentiment_evolution(film_sentiments, title, filename):
    """
    Create line plot showing sentiment evolution over time.
    
    Args:
        film_sentiments: Dictionary mapping film names to sentiment statistics
        title: Plot title
        filename: Output filename (without extension)
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=('Sentiment Evolution in Films',)
    )
    
    for film, data in film_sentiments.items():
        moving_average = pd.Series(data['all']).rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                y=moving_average,
                name=film,
                mode='lines'
            ),
            row=1, col=1
        )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def analyze_character_emotions(character_sentences, top_characters):
    """
    Analyze emotions for characters using NRC Lexicon.
    
    Args:
        character_sentences: Dictionary mapping character names to their sentences
        top_characters: Dictionary of top N characters
        
    Returns:
        Dictionary mapping character names to emotion frequencies
    """
    character_emotions = {}
    for character in top_characters.keys():
        all_emotions = defaultdict(int)
        for sentence in character_sentences[character]:
            emotions = NRCLex(sentence)
            for emotion, value in emotions.affect_frequencies.items():
                all_emotions[emotion] += value
        character_emotions[character] = dict(all_emotions)
    return character_emotions


def prepare_emotion_visualization_data(character_emotions):
    """
    Prepare emotion data for visualization.
    
    Args:
        character_emotions: Dictionary mapping character names to emotion frequencies
        
    Returns:
        List of dictionaries with 'Character', 'Emotion', 'Value'
    """
    emotions_data = []
    for character, emotions in character_emotions.items():
        for emotion, value in emotions.items():
            if emotion != 'anticip':
                emotions_data.append({
                    'Character': character,
                    'Emotion': emotion,
                    'Value': value
                })
    return emotions_data


def plot_emotion_heatmap(emotions_data, title, filename):
    """
    Create heatmap for emotion distribution.
    
    Args:
        emotions_data: List of dictionaries with 'Character', 'Emotion', 'Value'
        title: Plot title
        filename: Output filename (without extension)
    """
    emotions_df = pd.DataFrame(emotions_data)
    emotions_pivot = emotions_df.pivot(index='Character', columns='Emotion', values='Value')
    plt.figure(figsize=FIG_SIZE_LARGE)
    sns.heatmap(emotions_pivot, annot=True, cmap='YlOrRd', fmt='.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

