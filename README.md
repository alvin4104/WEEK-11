# NLP Preprocessing Pipeline: NLTK vs spaCy
 
**Project Type:** NLP Preprocessing & Evaluation  
**Task:** Compare NLTK and spaCy preprocessing pipelines on English sentences  
**Libraries:** NLTK, spaCy  

---

## Project Overview

This project implements and compares two NLP preprocessing pipelines — **NLTK** and **spaCy** — across three English sentences. Both pipelines perform tokenization, lowercasing, punctuation removal, stopword removal, stemming, and lemmatization. The output lemmas are evaluated against a gold standard using standard classification metrics.

## Preprocessing Steps

| Step | Description |
|------|-------------|
| Tokenization | Split sentence into individual tokens |
| Lowercasing | Convert all tokens to lowercase |
| Punctuation Removal | Remove all punctuation characters |
| Stopword Removal | Remove common function words |
| Stemming | Reduce words to their root form (Porter Stemmer) |
| Lemmatization | Reduce words to dictionary base form |

## Libraries Used

| Library | Role |
|---------|------|
| NLTK | Tokenization, stopwords, stemming, lemmatization |
| spaCy (en_core_web_sm) | Full NLP pipeline with POS-aware lemmatization |
| scikit-learn | Evaluation metrics (Accuracy, Precision, Recall, F1) |

## Evaluation Results

| Metric | NLTK | spaCy |
|--------|------|-------|
| Accuracy | 0.7778 | **1.0000** |
| Precision | 0.8750 | **1.0000** |
| Recall | 0.8750 | **1.0000** |
| F1-Score | 0.8750 | **1.0000** |

## Test Sentences

```
1. "Artificial intelligence is transforming education."
2. "The quick brown fox jumps over the lazy dog."
3. "Apple opened a new office in Singapore in 2023."
```

## Gold Standard Lemmas

| Sentence | Gold Lemmas |
|----------|-------------|
| 1 | artificial, intelligence, transform, education |
| 2 | quick, brown, fox, jump, lazy, dog |
| 3 | apple, open, new, office, singapore, 2023 |

## Pipeline

```
Input Sentences
      ↓
Tokenization → Lowercasing → Punctuation Removal
      ↓
Stopword Removal
      ↓
Stemming + Lemmatization
      ↓
Compare against Gold Standard
      ↓
Evaluate (Accuracy, Precision, Recall, F1)
```

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/alvin4104/WEEK-11.git
cd WEEK-11

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Run
python main.py
```

## Install Dependencies

```bash
pip install nltk spacy scikit-learn
python -m spacy download en_core_web_sm
```

## Key Findings

- **spaCy** achieved perfect scores (1.0000) across all metrics by integrating POS tagging directly into its pipeline, enabling context-aware lemmatization
- **NLTK** scored 0.7778 accuracy and 0.8750 F1, failing on two inflected verb forms: *transforming* → *transforming* (expected: *transform*) and *opened* → *opened* (expected: *open*)
- NLTK's WordNetLemmatizer defaults to treating tokens as nouns without explicit POS input, causing incorrect lemmatization of verbs
- spaCy automatically infers grammatical context through dependency parsing, eliminating the need for manual POS tagging
- Evaluation is based on 3 sentences and 18 tokens; larger datasets are needed for broader generalization
