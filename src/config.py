# -*- coding: utf-8 -*-
"""Configuration file for Star Wars Sentiment Analysis project."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# File mappings - map internal names to actual file names
FILE_MAPPINGS = {
    'nowa_nadzieja': 'new_hope.txt',
    'imperium_kontratakuje': 'empire_strikes_back.txt',
    'powrot_jedi': 'return_of_the_jedi.txt'
}

# Film names for display
FILM_NAMES = {
    'nowa_nadzieja': 'Nowa Nadzieja',
    'imperium_kontratakuje': 'Imperium Kontratakuje',
    'powrot_jedi': 'Powrót Jedi'
}

# NLTK downloads
NLTK_DOWNLOADS = ['punkt_tab', 'stopwords', 'wordnet']

# Model parameters
N_TOPICS = 5
N_CLUSTERS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Top N characters for analysis
TOP_N_CHARACTERS_STATS = 5
TOP_N_CHARACTERS_ANALYSIS = 7
TOP_N_CHARACTERS_CLUSTERING = 10

# Visualization settings
FIG_SIZE = (10, 5)
FIG_SIZE_LARGE = (12, 8)
FIG_SIZE_CONFUSION = (10, 8)

