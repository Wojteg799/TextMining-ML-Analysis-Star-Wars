# Text Mining & ML Analysis - Star Wars Original Trilogy

A comprehensive text mining and machine learning analysis project examining dialogue patterns, sentiment, and emotional trends in the Star Wars original trilogy (Episodes 4-6). This project utilizes natural language processing, topic modeling, clustering, classification, and sentiment analysis to explore character dialogues and narrative patterns.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business / Scientific Motivation](#business--scientific-motivation)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

---

## Project Overview

This repository contains a complete end-to-end text mining and ML project:

1. **Data Preprocessing**: Text cleaning, tokenization, and advanced preprocessing
2. **Statistical Analysis**: Character dialogue statistics, word counts, and distributions
3. **Visualization**: Word clouds, bar charts, box plots, and interactive visualizations
4. **Topic Modeling**: NMF-based topic analysis of character dialogues
5. **Clustering**: Hierarchical clustering and similarity analysis of characters
6. **Classification**: Multiple ML models to classify dialogue by character
7. **Sentiment Analysis**: TextBlob-based sentiment analysis across films and characters
8. **Emotion Analysis**: NRC Lexicon-based emotion detection and visualization

The project is designed to demonstrate professional data science workflows with clean, modular code structure suitable for portfolio presentation.

---

## Business / Scientific Motivation

Understanding narrative patterns, character development, and emotional arcs in film dialogues is valuable for:

- **Film Studies**: Analyzing narrative structure and character development
- **Natural Language Processing**: Demonstrating text mining techniques on structured dialogue
- **Sentiment Analysis**: Exploring emotional trends across a narrative arc
- **Machine Learning**: Building classification models for character identification
- **Data Visualization**: Creating compelling visualizations of text data

This project serves as a comprehensive example of text mining and ML techniques applied to a well-known dataset.

---

## Dataset

- **Source**: Star Wars Episodes IV, V, VI scripts
- **Format**: Processed text files with character dialogue markers
- **Files**:
  - `new_hope.txt` - Episode IV: A New Hope
  - `empire_strikes_back.txt` - Episode V: The Empire Strikes Back
  - `return_of_the_jedi.txt` - Episode VI: Return of the Jedi

The scripts have been preprocessed to mark character dialogues with `[POCZATEK_DIALOGU]` tags, making it easy to extract character-specific dialogue.

---

## Objectives

Main objectives of this project:

1. **Text Preprocessing**: Clean and process dialogue text for analysis
2. **Statistical Analysis**: Calculate word counts, sentence lengths, and character statistics
3. **Visualization**: Create word clouds, charts, and interactive visualizations
4. **Topic Modeling**: Identify main topics/themes in character dialogues using NMF
5. **Clustering**: Group characters based on dialogue similarity
6. **Classification**: Build ML models to identify which character said a given dialogue
7. **Sentiment Analysis**: Analyze emotional polarity across films and characters
8. **Emotion Analysis**: Detect and visualize emotions using NRC Lexicon

---

## Tech Stack

**Language & Core Libraries:**
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- WordCloud
- NLTK
- TextBlob
- NRCLex
- spaCy
- SciPy

---

## Project Structure

This project follows a standard, production-like structure for data/ML work:

```bash
TextMining-ML-Analysis-Star-Wars/
│
├── README.md                    # Project description (this file)
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── raw/
│   ├   ├── star-wars-episode-iv-a-new-hope-1977.docx   # raw script files
│   ├   ├── star-wars-episode-vi-return-of-the-jedi-1983.docx
│   ├   └──star-wars-episode-v-the-empire-strikes-back-1980.docx                  
│   └── processed/                  # Processed script files
│       ├── new_hope.txt
│       ├── empire_strikes_back.txt
│       └── return_of_the_jedi.txt
│
├── notebooks/      
│   ├── Star_Wars_Sentiment_Analysis_PL.ipynb    # Jupyter notebook with analysis
│   └── star_wars_sentiment_analysis_pl.py       # Python version of notebook
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── preprocessing.py          # Text preprocessing functions
│   ├── data_loading.py          # Data loading and preparation
│   ├── statistics.py            # Statistical analysis functions
│   ├── visualization.py         # Visualization functions
│   ├── topic_modeling.py        # Topic modeling (NMF)
│   ├── clustering.py            # Clustering and similarity analysis
│   ├── classification.py        # Character classification models
│   ├── sentiment_analysis.py    # Sentiment and emotion analysis
│   └── main.py                  # Main script to run complete pipeline
│
└── output/                       # Generated outputs (plots, visualizations)
```

---

## Usage

### Running the Complete Pipeline

To run the entire analysis pipeline (preprocessing, visualization, modeling, and evaluation):

```bash
python -m src.main
```

Or from the project root:

```bash
python src/main.py
```

This will:
1. Load and preprocess the data
2. Calculate statistics
3. Create all visualizations (saved to `output/`)
4. Perform topic modeling
5. Perform clustering analysis
6. Train and evaluate classification models
7. Perform sentiment and emotion analysis
8. Generate all plots and visualizations in `output/`

### Using Individual Modules

You can also import and use individual modules in your own scripts:

```python
from src.data_loading import prepare_data
from src.preprocessing import preprocess_text_advanced
from src.sentiment_analysis import analyze_sentiment

# Load data
data = prepare_data()

# Analyze sentiment
sentiment = analyze_sentiment("May the Force be with you.")
```

### Jupyter Notebook

For interactive exploration, use the Jupyter notebook:

```bash
jupyter notebook notebooks/Star_Wars_Sentiment_Analysis_PL.ipynb
```

---

## Results

The pipeline generates the following outputs in the `output/` directory:

**Visualizations:**
- `top_5_characters_*.html` - Top 5 characters by word count for each film
- `character_comparison_across_films.html` - Comparative character analysis
- `sentence_length_distribution.html` - Sentence length distributions
- `wordcloud_*.png` - Word clouds for films and characters
- `unique_words_*.png` - Unique words per film
- `topic_distribution.html` - Topic distribution across characters
- `character_dendrogram.png` - Hierarchical clustering dendrogram
- `character_clusters.html` - Character clustering visualization

**Classification Results:**
- `confusion_matrix_*.png` - Confusion matrices for each model
- Model comparison metrics printed to console

**Sentiment Analysis:**
- `sentiment_films.html` - Sentiment distribution across films
- `sentiment_characters.html` - Sentiment distribution across characters
- `mean_sentiment_comparison.html` - Mean sentiment comparison
- `sentiment_evolution.html` - Sentiment evolution over time

**Emotion Analysis:**
- `emotions_characters.html` - Emotion distribution across characters
- `emotions_heatmap.png` - Emotion heatmap

**Expected Performance:**
- **TF-IDF Classification**: Typically achieves ~44% accuracy for character classification
- **Binary Vectorization**: ~40% accuracy
- **Logarithmic Vectorization**: ~43% accuracy
- **Word Embeddings**: ~28% accuracy (lower due to short dialogue sentences)

---

## Key Findings

1. **Character Dominance**: Luke, Han, and Threepio dominate dialogue across all three films
2. **Sentiment Trends**: Films maintain relatively neutral sentiment with subtle variations
3. **Topic Diversity**: Characters show distinct topic preferences reflecting their roles
4. **Clustering**: Characters cluster into groups based on dialogue style (formal vs. dynamic)
5. **Emotion Patterns**: Main characters show distinct emotional profiles using NRC Lexicon

---

## License

See LICENSE file for details.

---

## Acknowledgments

- Star Wars scripts used for analysis
- NLTK, TextBlob, and NRCLex for NLP capabilities
- Scikit-learn for machine learning tools
