#IMPORT LIBRARIES 

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spacy
 
# ── Download required NLTK resources ──────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
 
# ── Load spaCy model ───────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")
 
# ── Input sentences ────────────────────────────────────────────────────────────
sentences = [
    "Artificial intelligence is transforming education.",
    "The quick brown fox jumps over the lazy dog.",
    "Apple opened a new office in Singapore in 2023.",
]
 
print("=" * 65)
print("         NLP PREPROCESSING: NLTK vs spaCy")
print("=" * 65)
 
# ══════════════════════════════════════════════════════════════════════════════
# NLTK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  NLTK PIPELINE")
print("─" * 65)
 
stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))
 
nltk_results = []   # store per-sentence results for evaluation
 
for i, sentence in enumerate(sentences, 1):
    print(f"\n[Sentence {i}] {sentence}")
 
    # 1. Input Text
    input_text = sentence
    print(f"  Input        : {input_text}")
 
    # 2. Tokenization
    tokens = word_tokenize(input_text)
    print(f"  Tokenized    : {tokens}")
 
    # 3. Lowercasing
    lower_tokens = [t.lower() for t in tokens]
    print(f"  Lowercased   : {lower_tokens}")
 
    # 4. Punctuation Removal
    no_punct = [t for t in lower_tokens if t not in string.punctuation]
    print(f"  No Punct     : {no_punct}")
 
    # 5. Stopword Removal
    no_stop = [t for t in no_punct if t not in stop_words]
    print(f"  No Stopwords : {no_stop}")
 
    # 6. Stemming
    stemmed = [stemmer.stem(t) for t in no_stop]
    print(f"  Stemmed      : {stemmed}")
 
    # 7. Lemmatization
    lemmatized = [lemmatizer.lemmatize(t) for t in no_stop]
    print(f"  Lemmatized   : {lemmatized}")
 
    nltk_results.append({
        "tokens":     no_stop,
        "stemmed":    stemmed,
        "lemmatized": lemmatized,
    })
 
# ══════════════════════════════════════════════════════════════════════════════
# spaCy PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  spaCy PIPELINE")
print("─" * 65)
 
spacy_results = []
 
for i, sentence in enumerate(sentences, 1):
    print(f"\n[Sentence {i}] {sentence}")
 
    doc = nlp(sentence)
 
    # 1. Tokenization
    tokens = [token.text for token in doc]
    print(f"  Tokenized    : {tokens}")
 
    # 2. Lowercasing
    lower_tokens = [t.lower() for t in tokens]
    print(f"  Lowercased   : {lower_tokens}")
 
    # 3. Punctuation Removal
    no_punct = [token for token in doc if not token.is_punct]
    print(f"  No Punct     : {[t.text for t in no_punct]}")
 
    # 4. Stopword Removal
    no_stop = [token for token in no_punct if not token.is_stop]
    print(f"  No Stopwords : {[t.text for t in no_stop]}")
 
    # 5. Stemming  (spaCy has no built-in stemmer; use NLTK's stemmer)
    stemmed = [stemmer.stem(t.text.lower()) for t in no_stop]
    print(f"  Stemmed      : {stemmed}")
 
    # 6. Lemmatization
    lemmatized = [token.lemma_.lower() for token in no_stop]
    print(f"  Lemmatized   : {lemmatized}")
 
    spacy_results.append({
        "tokens":     [t.text.lower() for t in no_stop],
        "stemmed":    stemmed,
        "lemmatized": lemmatized,
    })
 
# ══════════════════════════════════════════════════════════════════════════════
# GOLD DATA & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  GOLD DATA & EVALUATION")
print("─" * 65)
 
# Gold standard: expected lemmatized tokens after full preprocessing
gold_lemmas = [
    # Sentence 1: "Artificial intelligence is transforming education."
    ["artificial", "intelligence", "transform", "education"],
    # Sentence 2: "The quick brown fox jumps over the lazy dog."
    ["quick", "brown", "fox", "jump", "lazy", "dog"],
    # Sentence 3: "Apple opened a new office in Singapore in 2023."
    ["apple", "open", "new", "office", "singapore", "2023"],
]
 
def tokens_to_binary(predicted, gold):
    """Convert token lists to binary vectors over the union vocabulary."""
    vocab = sorted(set(predicted) | set(gold))
    pred_vec = [1 if w in predicted else 0 for w in vocab]
    gold_vec = [1 if w in gold      else 0 for w in vocab]
    return pred_vec, gold_vec
 
def evaluate(name, results, gold):
    print(f"\n  [{name}]")
    all_pred, all_gold = [], []
    for i, (res, g) in enumerate(zip(results, gold)):
        p, g_vec = tokens_to_binary(res["lemmatized"], g)
        all_pred.extend(p)
        all_gold.extend(g_vec)
        print(f"    Sentence {i+1} lemmas  : {res['lemmatized']}")
        print(f"    Gold lemmas         : {g}")
 
    acc  = accuracy_score(all_gold, all_pred)
    prec = precision_score(all_gold, all_pred, zero_division=0)
    rec  = recall_score(all_gold, all_pred, zero_division=0)
    f1   = f1_score(all_gold, all_pred, zero_division=0)
 
    print(f"\n    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    return acc, prec, rec, f1
 
nltk_scores  = evaluate("NLTK",  nltk_results,  gold_lemmas)
spacy_scores = evaluate("spaCy", spacy_results, gold_lemmas)
 
# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  COMPARISON SUMMARY")
print("─" * 65)
print(f"  {'Metric':<12} {'NLTK':>10} {'spaCy':>10}")
print(f"  {'─'*12} {'─'*10} {'─'*10}")
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
for m, n, s in zip(metrics, nltk_scores, spacy_scores):
    print(f"  {m:<12} {n:>10.4f} {s:>10.4f}")
print("=" * 65)