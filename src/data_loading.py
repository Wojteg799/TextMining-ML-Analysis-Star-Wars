# -*- coding: utf-8 -*-
"""Data loading and preparation functions."""

from collections import defaultdict
from pathlib import Path
from .config import DATA_DIR, FILE_MAPPINGS
from .preprocessing import (
    preprocess_text_basic,
    preprocess_text_advanced,
    extract_character_dialogues
)


def load_script(file_name):
    """
    Load script file.
    
    Args:
        file_name: Name of the file (will be looked up in FILE_MAPPINGS)
        
    Returns:
        File content as string
    """
    if file_name in FILE_MAPPINGS:
        file_name = FILE_MAPPINGS[file_name]
    
    file_path = DATA_DIR / file_name
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def prepare_data():
    """
    Load and prepare all data from scripts.
    
    Returns:
        Dictionary containing:
        - character_dialogues_by_film: Dialogues by character per film
        - film_sentences: All sentences per film
        - film_terms_basic: Basic processed terms per film
        - film_terms_advanced: Advanced processed terms per film
        - character_sentences: All sentences per character (across all films)
        - character_terms_basic: Basic processed terms per character
        - character_terms_advanced: Advanced processed terms per character
    """
    films = {
        'nowa_nadzieja': 'nowa_nadzieja',
        'imperium_kontratakuje': 'imperium_kontratakuje',
        'powrot_jedi': 'powrot_jedi'
    }
    
    character_dialogues_by_film = {}
    film_sentences = {}
    film_terms_basic = {}
    film_terms_advanced = {}
    character_sentences = defaultdict(list)
    character_terms_basic = defaultdict(list)
    character_terms_advanced = defaultdict(list)
    
    # Process each film
    for film_name, file_name in films.items():
        script_text = load_script(file_name)
        character_dialogues_by_film[film_name] = extract_character_dialogues(script_text)
        
        all_sentences = []
        for sentences in character_dialogues_by_film[film_name].values():
            all_sentences.extend(sentences)
        
        film_sentences[film_name] = all_sentences
        all_sentences_text = ' '.join(all_sentences)
        film_terms_basic[film_name] = preprocess_text_basic(all_sentences_text)
        film_terms_advanced[film_name] = preprocess_text_advanced(all_sentences_text)
        
        for character, sentences in character_dialogues_by_film[film_name].items():
            character_sentences[character].extend(sentences)
    
    # Process character terms
    for character, sentences in character_sentences.items():
        all_sentences_text = ' '.join(sentences)
        character_terms_basic[character] = preprocess_text_basic(all_sentences_text)
        character_terms_advanced[character] = preprocess_text_advanced(all_sentences_text)
    
    return {
        'character_dialogues_by_film': character_dialogues_by_film,
        'film_sentences': film_sentences,
        'film_terms_basic': film_terms_basic,
        'film_terms_advanced': film_terms_advanced,
        'character_sentences': character_sentences,
        'character_terms_basic': character_terms_basic,
        'character_terms_advanced': character_terms_advanced
    }
