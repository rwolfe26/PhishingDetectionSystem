"""
Latent Semantic Analysis (LSA) extraction for email phishing detection.

This module defines a simple pipeline for converting raw email messages into
dense semantic vectors using term‑frequency/inverse‑document frequency (TF–IDF)
weighting followed by truncated singular value decomposition (SVD).  The
resulting vectors can be used as input features to train downstream machine
learning models for tasks such as email classification (safe, suspicious,
highly phishing).

The key components are:

* Text preprocessing: lower‑casing, removal of HTML tags, numbers and
  punctuation, and optional stop‑word filtering.  Emails are provided as
  plain strings and may include headers.  Preprocessing is deliberately kept
  simple to preserve semantic content; feel free to customise the
  ``_preprocess`` function according to your corpus.

* TF–IDF vectorisation: uses scikit‑learn's ``TfidfVectorizer`` with
  sub‑linear term frequency scaling and inverse document frequency weighting,
  as recommended for latent semantic analysis【607706025995768†L423-L457】.  English stop words
  are removed by default.  The vectoriser learns a vocabulary from the
  training set and converts each document to a sparse TF–IDF vector.

* Truncated SVD: implemented via scikit‑learn's ``TruncatedSVD``.  When
  applied to a term–document matrix, truncated SVD uncovers latent semantic
  structure by approximating the matrix with a lower‑rank representation.  In
  the context of text analysis this is often called latent semantic analysis
  (LSA)【607706025995768†L423-L456】.  The number of components (``n_components``) controls
  the dimensionality of the output vectors.  A value of 768 yields 768‑dimensional
  embeddings, matching the size produced by many modern transformer models.

Example:

>>> from lsa_tool import LSATextEncoder
>>> emails = ["Free money!!!", "Quarterly report attached"]
>>> encoder = LSATextEncoder(n_components=768)
>>> X = encoder.fit_transform(emails)
>>> X.shape
(2, 768)

After fitting the encoder on your training corpus, you can persist the model
using ``joblib.dump`` for later reuse, and feed the resulting vectors into
a classifier such as logistic regression, support vector machines, random
forest or any other supervised learning algorithm.
"""

import re
from pathlib import Path
import os
from typing import Iterable, List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

class LSATextEncoder:
    """Pipeline for computing latent semantic analysis (LSA) embeddings.

    Parameters
    ----------
    n_components : int, default=768
        Number of latent dimensions (SVD components) to retain.  Must be
        smaller than the number of features in the TF–IDF representation.  A
        dimension of 768 yields vector lengths comparable to BERT or GPT
        embeddings.

    max_features : Optional[int], default=None
        If provided, limit the vocabulary to the ``max_features`` most frequent
        terms in the training corpus.  Limiting features can reduce memory
        usage on very large datasets.  ``None`` keeps all tokens above
        ``min_df``.

    min_df : int or float, default=1
        Minimum document frequency for terms.  Terms that appear in fewer
        documents than ``min_df`` are ignored.  When specified as an integer
        ``n``, terms appearing in fewer than ``n`` documents are discarded.
        When a float is provided, it represents a proportion of the corpus.  A
        default of ``1`` includes all words that survive other filters; this
        avoids problems when the training corpus contains only a few
        documents.

    max_df : float, default=0.85
        Ignore terms that appear in more than ``max_df`` proportion of the
        documents.  High ``max_df`` values filter out extremely frequent words
        (such as “subject” or “from” in email headers) that do not carry
        semantic information.

    stop_words : str or list, default="english"
        Stop words to remove before vectorisation.  By default uses
        scikit‑learn’s built‑in English stop list.
    """

    def __init__(self,
                 n_components: int = 768,
                 max_features: Optional[int] = None,
                 min_df=1,
                 max_df=0.85,
                 stop_words='english'):
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words

        # Placeholders for fitted components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.normalizer: Optional[Normalizer] = None

    @staticmethod
    def _preprocess(text: str) -> str:
        """Basic text preprocessing.

        Performs simple normalisation: lower‑case, remove HTML tags, strip
        punctuation and digits.  You can extend this method to include more
        sophisticated tokenisation, stemming or lemmatisation.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text.
        """
        # Lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+', ' ', text)
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', text)
        # Remove non‑letter characters
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _build_vectorizer(self) -> TfidfVectorizer:
        """Instantiate a TF–IDF vectoriser configured for LSA.

        We enable sub‑linear TF scaling and IDF weighting, as recommended
        for latent semantic analysis on text corpora【607706025995768†L423-L456】.

        Returns
        -------
        TfidfVectorizer
            Configured vectoriser instance.
        """
        return TfidfVectorizer(
            preprocessor=self._preprocess,
            stop_words=self.stop_words,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            use_idf=True,
            ngram_range=(1, 2)
        )

    def _build_svd(self) -> TruncatedSVD:
        """Instantiate the truncated SVD transformer.

        Returns
        -------
        TruncatedSVD
            Configured truncated SVD instance.
        """
        return TruncatedSVD(n_components=self.n_components, random_state=42)

    def fit(self, documents: Iterable[str]) -> 'LSATextEncoder':
        """Fit the TF–IDF vectoriser and truncated SVD on a corpus.

        Parameters
        ----------
        documents : iterable of str
            Training documents.

        Returns
        -------
        self
            The fitted encoder.
        """
        # Fit vectoriser
        self.vectorizer = self._build_vectorizer()
        X_tfidf = self.vectorizer.fit_transform(documents)

        # Fit truncated SVD on TF–IDF features
        self.svd = self._build_svd()
        X_reduced = self.svd.fit_transform(X_tfidf)

        # L2 normalise the dense vectors.  Normalisation ensures that vector
        # magnitude does not scale with document length and is a common step in
        # LSA pipelines.  Use Normalizer to compute row‑wise L2 norms.
        self.normalizer = Normalizer(copy=False)
        self.normalizer.fit(X_reduced)
        return self

    def transform(self, documents: Iterable[str]) -> np.ndarray:
        """Transform new documents into LSA embeddings.

        The vectoriser and SVD must be fitted before calling this method.

        Parameters
        ----------
        documents : iterable of str
            Documents to transform.

        Returns
        -------
        numpy.ndarray
            2D array of shape (n_documents, n_components) containing the
            normalised LSA embeddings.
        """
        if self.vectorizer is None or self.svd is None or self.normalizer is None:
            raise RuntimeError("The encoder has not been fitted yet. Call 'fit' first.")

        X_tfidf = self.vectorizer.transform(documents)
        X_reduced = self.svd.transform(X_tfidf)
        X_norm = self.normalizer.transform(X_reduced)
        return X_norm

    def fit_transform(self, documents: Iterable[str]) -> np.ndarray:
        """Fit the encoder on the corpus and return the embeddings.

        Equivalent to calling ``fit`` followed by ``transform``.

        Parameters
        ----------
        documents : iterable of str
            Documents to fit on and transform.

        Returns
        -------
        numpy.ndarray
            Normalised LSA embeddings for each document.
        """
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> Optional[List[str]]:
        """Return the vocabulary learned by the TF–IDF vectoriser.

        Returns
        -------
        list of str or None
            Feature names if the vectoriser has been fitted; otherwise ``None``.
        """
        return None if self.vectorizer is None else self.vectorizer.get_feature_names_out()

    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Return the proportion of variance explained by each SVD component.

        This can help decide whether the chosen ``n_components`` captures an
        adequate amount of information.

        Returns
        -------
        numpy.ndarray or None
            Array of length ``n_components`` with explained variance ratios, or
            ``None`` if the encoder has not been fitted.
        """
        return None if self.svd is None else self.svd.explained_variance_ratio_

def load_emails_from_directory(directory: Path) -> List[str]:
    """Load email content from all files in a directory."""
    emails = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                emails.append(f.read())
    return emails

if __name__ == "__main__":
    # Example usage for demonstration purposes.  This block will not be
    # executed when the module is imported but can be run directly to test
    # functionality with a simple corpus.
    
    # Define the path to the dataset directory relative to this script file
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / 'Datasets' / 'easy_ham'

    # Load emails from the directory
    emails = load_emails_from_directory(dataset_dir)
    
    # Initialize and fit the encoder
    encoder = LSATextEncoder(n_components=100)  # Using 100 components for efficiency
    embeddings = encoder.fit_transform(emails)
    
    # Print results
    print(f"Loaded {len(emails)} emails.")
    print("Embeddings shape:", embeddings.shape)
    
    # Optionally, print explained variance to assess the model
    explained_variance = encoder.explained_variance_ratio()
    if explained_variance is not None:
        print("Total explained variance:", explained_variance.sum())
