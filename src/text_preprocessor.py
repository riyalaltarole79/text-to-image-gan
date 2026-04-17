"""
text_preprocessor.py
====================
NLTK-based text preprocessing pipeline for flower captions.

Stages
------
1. Lowercasing & Unicode normalization
2. Contraction expansion
3. Punctuation & noise removal
4. Tokenization  (NLTK word_tokenize)
5. Stop-word removal  (configurable keep-list for flower-relevant words)
6. Lemmatization  (WordNetLemmatizer)
7. Part-of-speech filtering  (keep nouns, adjectives, verbs)
8. Caption augmentation  (synonym replacement via WordNet)
"""

import re
import string
import unicodedata
import random
from typing import List, Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag as _nltk_pos_tag

# ── Download required NLTK data (one-time) ─────────────────────────────────
_NLTK_PACKAGES = [
    "punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
    "omw-1.4", "punkt_tab",
]
for _pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

# ── Contraction map ────────────────────────────────────────────────────────
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "it's": "it is",
    "i'm":   "i am",   "i've":  "i have",   "i'll": "i will",
    "don't": "do not", "doesn't": "does not","didn't": "did not",
    "isn't": "is not", "aren't": "are not",  "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "mustn't": "must not", "let's": "let us",
    "that's": "that is", "who's": "who is", "what's": "what is",
}

# Flower / color / texture words we always keep even if stop-words
DOMAIN_KEEP = {
    "red", "blue", "yellow", "white", "pink", "purple", "orange", "green",
    "violet", "crimson", "golden", "pale", "bright", "dark", "light",
    "small", "large", "tiny", "big", "beautiful", "delicate", "fragrant",
    "petals", "petal", "leaf", "leaves", "stem", "stamen", "pistil",
    "bloom", "blossom", "flower", "floral", "garden", "field", "wild",
    "spring", "summer", "tropical", "alpine", "exotic", "rare",
}

# POS tags to keep after filtering
_KEEP_POS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBG", "VBD", "VBN"}

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS  = set(stopwords.words("english")) - DOMAIN_KEEP


def _expand_contractions(text: str) -> str:
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
                         re.IGNORECASE)
    def replacer(m):
        return CONTRACTIONS.get(m.group(0).lower(), m.group(0))
    return pattern.sub(replacer, text)


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _nltk_pos_to_wordnet(tag: str):
    """Map NLTK POS tag to WordNet POS."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _get_synonyms(word: str, pos_tag: str) -> List[str]:
    """Return WordNet synonyms for a word (same POS)."""
    wn_pos = _nltk_pos_to_wordnet(pos_tag)
    synonyms = set()
    for syn in wordnet.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word and len(candidate.split()) == 1:
                synonyms.add(candidate)
    return list(synonyms)


class TextPreprocessor:
    """
    Full NLP preprocessing pipeline for flower image captions.

    Parameters
    ----------
    remove_stopwords : bool
        Whether to remove stop-words (domain words are always kept).
    pos_filter : bool
        Keep only content POS tags (nouns, adjectives, verbs).
    augment : bool
        Enable synonym-based caption augmentation.
    aug_prob : float
        Probability of replacing each token with a synonym.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        pos_filter: bool = False,   # False keeps more words for SBERT
        augment: bool = True,
        aug_prob: float = 0.15,
        seed: int = 42,
    ):
        self.remove_stopwords = remove_stopwords
        self.pos_filter       = pos_filter
        self.augment          = augment
        self.aug_prob         = aug_prob
        random.seed(seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Return a single cleaned sentence (no augmentation)."""
        return self._preprocess(text, augment=False)

    def augment_captions(self, text: str, n: int = 4) -> List[str]:
        """
        Return *n* augmented variants of *text* (plus the cleaned original).
        """
        cleaned = self.clean(text)
        variants = {cleaned}
        attempts = 0
        while len(variants) < n + 1 and attempts < n * 3:
            variants.add(self._preprocess(text, augment=True))
            attempts += 1
        return list(variants)

    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    # ── Internal ────────────────────────────────────────────────────────────

    def _preprocess(self, text: str, augment: bool = False) -> str:
        # 1. Unicode normalise
        text = _normalize_unicode(text)
        # 2. Lowercase
        text = text.lower()
        # 3. Expand contractions
        text = _expand_contractions(text)
        # 4. Remove URLs, emails, special chars
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-z\s'-]", " ", text)
        # 5. Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 6. Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        # 7. POS tagging
        try:
            tagged = nltk.pos_tag(tokens)
        except Exception:
            tagged = [(t, "NN") for t in tokens]

        # 8. Filter & lemmatize
        processed = []
        for word, tag in tagged:
            if word in string.punctuation:
                continue
            if self.remove_stopwords and word in _STOP_WORDS and word not in DOMAIN_KEEP:
                continue
            if self.pos_filter and tag not in _KEEP_POS and word not in DOMAIN_KEEP:
                continue
            # Lemmatize
            wn_pos = _nltk_pos_to_wordnet(tag)
            lemma  = _LEMMATIZER.lemmatize(word, pos=wn_pos)

            # 9. Optional synonym augmentation
            if augment and self.augment and random.random() < self.aug_prob:
                syns = _get_synonyms(lemma, tag)
                if syns:
                    lemma = random.choice(syns)

            processed.append(lemma)

        return " ".join(processed) if processed else text


# ── Caption templates for flower dataset ───────────────────────────────────

COLOR_DESCRIPTORS = [
    "vibrant", "delicate", "beautiful", "striking", "magnificent",
    "elegant", "graceful", "lovely", "stunning", "radiant",
]

SETTING_DESCRIPTORS = [
    "in a garden",  "in a meadow",   "in the wild",
    "in sunlight",  "in soft light", "in full bloom",
    "with green leaves", "with slender stems", "against a blurred background",
]

STRUCTURE_DESCRIPTORS = [
    "with soft petals",    "with layered petals",  "with narrow petals",
    "with rounded petals", "with ruffled petals",  "with large petals",
    "with many petals",    "with colorful stamens","with a prominent center",
]


def generate_captions_for_class(class_name: str, n: int = 5) -> List[str]:
    """
    Generate *n* diverse natural-language captions for a flower class name.

    Example
    -------
    >>> generate_captions_for_class("fire lily", n=3)
    ['a vibrant fire lily in full bloom with soft petals',
     'a stunning fire lily in the wild with many petals',
     'a photograph of a beautiful fire lily in a garden']
    """
    templates = [
        "a {adj} {name} {setting} {struct}",
        "a close-up of a {adj} {name} {struct}",
        "a photograph of a {adj} {name} {setting}",
        "a {name} flower {struct} {setting}",
        "a {adj} {name} flower in full bloom",
        "beautiful {name} flowers {setting}",
        "{name} {struct} under natural light",
    ]

    captions = set()
    attempts  = 0
    while len(captions) < n and attempts < n * 5:
        tmpl   = random.choice(templates)
        adj    = random.choice(COLOR_DESCRIPTORS)
        setting = random.choice(SETTING_DESCRIPTORS)
        struct  = random.choice(STRUCTURE_DESCRIPTORS)
        cap    = tmpl.format(
            name=class_name,
            adj=adj,
            setting=setting,
            struct=struct,
        )
        captions.add(cap)
        attempts += 1

    return list(captions)[:n]


if __name__ == "__main__":
    # Quick smoke test
    preprocessor = TextPreprocessor(augment=True)

    test_sentences = [
        "A beautiful pink primrose with soft petals in the garden.",
        "Vibrant fire lily blooming wildly in sunlight!",
        "Delicate moon orchid with layered petals and a prominent center.",
    ]

    print("=" * 60)
    print("TEXT PREPROCESSOR — SMOKE TEST")
    print("=" * 60)

    for sent in test_sentences:
        print(f"\nOriginal : {sent}")
        print(f"Cleaned  : {preprocessor.clean(sent)}")
        variants = preprocessor.augment_captions(sent, n=2)
        for i, v in enumerate(variants, 1):
            print(f"  Aug {i}  : {v}")

    print("\nCaption generation:")
    for cap in generate_captions_for_class("rose", n=4):
        print(" •", cap)
