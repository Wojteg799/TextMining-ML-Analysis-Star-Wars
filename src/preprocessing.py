# -*- coding: utf-8 -*-
"""Text preprocessing functions for Star Wars script analysis."""

import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


def preprocess_text_basic(text):
    """
    Basic text preprocessing: lowercase, remove punctuation and digits.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    return tokens


def preprocess_text_advanced(text):
    """
    Advanced text preprocessing: basic preprocessing + stopwords removal + lemmatization.
    
    Args:
        text: Input text string
        
    Returns:
        List of processed tokens
    """
    tokens = preprocess_text_basic(text)
    
    stop_words = set(stopwords.words('english'))
    additional_stop_words = {
        'dont', 'youre', 'youll', 'youd', 'youve', 'thats', 'thats', 'theyre',
        'theyve', 'theyll', 'theyd', 'shes', 'shed', 'shell', 'hes', 'hed',
        'hell', 'ive', 'im', 'id', 'ill', 'were', 'weve', 'wed', 'well',
        'arent', 'isnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
        'didnt', 'doesnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'cant',
        'cannot', 'aint', 'gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'gotta',
        'ya', 'yeah', 'yes', 'no', 'ok', 'okay', 'hey', 'hi', 'hello', 'oh',
        'ah', 'um', 'uh', 'er', 'hmm', 'huh', 'ha', 'ha', 'heh', 'hehe',
        'lol', 'wow', 'whoa', 'whoah', 'whoops', 'oops', 'ouch', 'ow', 'oww'
    }
    stop_words.update(additional_stop_words)
    
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def extract_character_dialogues(text):
    """
    Extract character dialogues from script text.
    
    Character dialogue is a line that appears directly after a line with character name
    (always written in capital letters) and [POCZATEK_DIALOGU].
    Uses only the first word of character name as key (e.g., LUKE and LUKE (through comlink)
    are treated as the same character).
    
    Args:
        text: Script text
        
    Returns:
        Dictionary mapping character names to their dialogue sentences
    """
    lines = text.split('\n')
    character_dialogues = defaultdict(list)
    
    i = 0
    while i < len(lines) - 1:
        if '[POCZATEK_DIALOGU]' in lines[i]:
            full_name = lines[i].split('[POCZATEK_DIALOGU]')[0].strip()
            character = full_name.split()[0]
            
            if i + 1 < len(lines):
                if lines[i + 1].strip().startswith('(') and lines[i + 1].strip().endswith(')'):
                    if i + 2 < len(lines) and lines[i + 2].strip():
                        sentence = lines[i + 2].strip()
                        character_dialogues[character].append(sentence)
                        i += 3
                        continue
                else:
                    if lines[i + 1].strip():
                        sentence = lines[i + 1].strip()
                        character_dialogues[character].append(sentence)
        i += 1
    
    # Remove scene directions and numeric keys
    keys_to_remove = []
    for key in character_dialogues.keys():
        if key in ['INT.', 'EXT.', 'INT', 'EXT'] or any(c.isdigit() for c in key):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del character_dialogues[key]
    
    return character_dialogues
