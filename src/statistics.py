# -*- coding: utf-8 -*-
"""Statistical analysis and data preparation functions."""

from collections import Counter
from .preprocessing import preprocess_text_basic


def find_unique_words(terms_dict, min_occurrences=3):
    """
    Find words unique to each key in the dictionary.
    
    Args:
        terms_dict: Dictionary mapping keys to lists of terms
        min_occurrences: Minimum number of occurrences for a word to be considered
        
    Returns:
        Dictionary mapping keys to unique word frequencies
    """
    all_terms = []
    for terms in terms_dict.values():
        all_terms.extend(terms)
    all_counter = Counter(all_terms)
    
    unique_words = {}
    for key, terms in terms_dict.items():
        key_counter = Counter(terms)
        unique = {
            word: count for word, count in key_counter.items()
            if count >= min_occurrences and count == all_counter[word]
        }
        unique_words[key] = unique
    
    return unique_words


def calculate_character_word_counts(character_sentences):
    """
    Calculate total word count for each character.
    
    Args:
        character_sentences: Dictionary mapping character names to their sentences
        
    Returns:
        Dictionary mapping character names to word counts
    """
    character_words = {}
    for character, sentences in character_sentences.items():
        all_sentences_text = ' '.join(sentences)
        terms = preprocess_text_basic(all_sentences_text)
        character_words[character] = len(terms)
    return character_words


def get_top_characters(character_words, n):
    """
    Get top N characters by word count.
    
    Args:
        character_words: Dictionary mapping character names to word counts
        n: Number of top characters to return
        
    Returns:
        Dictionary of top N characters
    """
    return dict(sorted(character_words.items(), key=lambda x: x[1], reverse=True)[:n])


def prepare_comparative_data(character_dialogues_by_film, top_characters, films):
    """
    Prepare data for comparative visualization across films.
    
    Args:
        character_dialogues_by_film: Dictionary mapping film names to character dialogues
        top_characters: Dictionary of top N characters
        films: Dictionary mapping film keys to film names
        
    Returns:
        List of dictionaries with 'Character', 'Film', 'Word Count'
    """
    comparative_data = []
    for character in top_characters.keys():
        for film_key, film_name in films.items():
            if character in character_dialogues_by_film[film_key]:
                all_sentences_text = ' '.join(character_dialogues_by_film[film_key][character])
                terms = preprocess_text_basic(all_sentences_text)
                word_count = len(terms)
                comparative_data.append({
                    'Character': character,
                    'Film': film_name,
                    'Word Count': word_count
                })
    return comparative_data


def prepare_sentence_length_data(character_sentences, top_characters):
    """
    Prepare data for sentence length analysis.
    
    Args:
        character_sentences: Dictionary mapping character names to their sentences
        top_characters: Dictionary of top N characters
        
    Returns:
        List of dictionaries with 'Character', 'Sentence Length'
    """
    length_data = []
    for character in top_characters.keys():
        for sentence in character_sentences[character]:
            length = len(sentence.split())
            length_data.append({
                'Character': character,
                'Sentence Length': length
            })
    return length_data


def print_film_statistics(character_dialogues_by_film, film_sentences):
    """
    Print statistics for each film.
    
    Args:
        character_dialogues_by_film: Dictionary mapping film names to character dialogues
        film_sentences: Dictionary mapping film names to all sentences
    """
    for film_name, characters in character_dialogues_by_film.items():
        print(f"\nStatistics for film: {film_name}")
        print(f"Number of characters in film: {len(characters)}")
        for character, sentences in characters.items():
            print(f"{character}: {len(sentences)} sentences")
    
    print("\nNumber of sentences in each film:")
    for film, sentences in film_sentences.items():
        print(f"{film}: {len(sentences)} sentences")
    
    print("\nNumber of terms in each film:")
    # This would need film_terms_basic, but we'll keep it simple here
    # The actual printing will be done in main.py with full data

