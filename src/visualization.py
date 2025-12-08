# -*- coding: utf-8 -*-
"""Visualization functions for Star Wars analysis."""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from .config import OUTPUT_DIR, FIG_SIZE, FIG_SIZE_LARGE


def create_wordcloud(terms, title, filename):
    """
    Create and save word cloud.
    
    Args:
        terms: List of terms or string
        title: Title for the plot
        filename: Output filename (without extension)
    """
    if isinstance(terms, list):
        all_terms = ' '.join(terms)
    else:
        all_terms = terms
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_terms)
    
    plt.figure(figsize=FIG_SIZE)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_wordcloud_from_frequencies(frequencies, title, filename):
    """
    Create word cloud from frequency dictionary.
    
    Args:
        frequencies: Dictionary mapping words to frequencies
        title: Title for the plot
        filename: Output filename (without extension)
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
    
    plt.figure(figsize=FIG_SIZE)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_characters_bar(character_words, title, filename, top_n=5):
    """
    Create bar chart for top N characters by word count.
    
    Args:
        character_words: Dictionary mapping character names to word counts
        title: Title for the plot
        filename: Output filename (without extension)
        top_n: Number of top characters to show
    """
    top_characters = dict(sorted(character_words.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    fig = px.bar(
        x=list(top_characters.keys()),
        y=list(top_characters.values()),
        title=title,
        labels={'x': 'Character', 'y': 'Word Count'}
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass  # Skip if kaleido not available


def plot_comparative_characters(comparative_data, title, filename):
    """
    Create comparative bar chart for characters across films.
    
    Args:
        comparative_data: List of dictionaries with 'Character', 'Film', 'Word Count'
        title: Title for the plot
        filename: Output filename (without extension)
    """
    fig = px.bar(
        comparative_data,
        x='Character',
        y='Word Count',
        color='Film',
        title=title,
        barmode='group'
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_sentence_lengths(length_data, title, filename):
    """
    Create box plot for sentence length distribution.
    
    Args:
        length_data: List of dictionaries with 'Character', 'Sentence Length'
        title: Title for the plot
        filename: Output filename (without extension)
    """
    fig = px.box(
        length_data,
        x='Character',
        y='Sentence Length',
        title=title,
        points='all'
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_topic_distribution(topics_data, title, filename):
    """
    Create stacked bar chart for topic distribution.
    
    Args:
        topics_data: List of dictionaries with 'Character', 'Topic', 'Share'
        title: Title for the plot
        filename: Output filename (without extension)
    """
    fig = px.bar(
        topics_data,
        x='Character',
        y='Share',
        color='Topic',
        title=title,
        barmode='stack'
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_clusters(cluster_data, title, filename):
    """
    Create scatter plot for character clustering.
    
    Args:
        cluster_data: List of dictionaries with 'Character', 'Cluster'
        title: Title for the plot
        filename: Output filename (without extension)
    """
    fig = px.scatter(
        cluster_data,
        x='Character',
        y='Cluster',
        title=title,
        color='Cluster'
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass


def plot_emotions(emotions_data, title, filename):
    """
    Create bar chart for emotion distribution.
    
    Args:
        emotions_data: List of dictionaries with 'Character', 'Emotion', 'Value'
        title: Title for the plot
        filename: Output filename (without extension)
    """
    fig = px.bar(
        emotions_data,
        x='Character',
        y='Value',
        color='Emotion',
        title=title,
        barmode='group'
    )
    fig.write_html(OUTPUT_DIR / f'{filename}.html')
    try:
        fig.write_image(OUTPUT_DIR / f'{filename}.png')
    except:
        pass
