Building a Stable Latent Semantic Analysis Tool for Phishing Email Detection

Motivation and constraints

The goal of this project is to design a semantic analysis tool that transforms
raw email messages into dense vector embeddings that downstream machine
learning models can use to decide whether an email is safe, suspicious or
highly phishing.  The user’s training data consists of labelled emails
(2 551 “easy‑ham” messages for the safe class) with all messages in English.
Computation must run on a local machine with 32 GB RAM, so the method must
be memory‑efficient, free to use and robust.  The desired output is a
768‑dimensional vector representing each email.

The user’s previous experiments with BERT encountered stability issues.  While
transformer‑based models such as BERT and GPT provide excellent semantic
representations, they are expensive to train and deploy on small machines.
Recent research comparing embedding models highlights that traditional
representations (TF–IDF, Word2Vec, FastText) remain competitive for many tasks
in resource‑constrained environments ￼.

Latent semantic analysis (LSA)

Latent Semantic Analysis (also called latent semantic indexing, LSI) is a
linear algebra technique that discovers relationships between words and
documents by applying truncated singular value decomposition (SVD) to a
term‑document matrix.  When truncated SVD is applied to a TF‑IDF matrix,
scikit‑learn notes that this transformation is known as latent semantic
analysis and can combat synonymy and polysemy in text data ￼.

The scikit‑learn documentation emphasises the following points when using
truncated SVD for LSA ￼:
	•	The method transforms a sparse term‑document matrix into a low‑dimensional
semantic space.  The number of dimensions (components) is user‑defined; the
output vectors reflect latent topics.
	•	It is important to use a TF–IDF matrix (rather than raw term counts) and
enable sublinear TF scaling and inverse document frequency weighting.
This choice helps approximate a Gaussian distribution of feature values and
improves LSA performance ￼.

Because LSA is a count‑based model, it works well on smaller corpora.
Discussion comparing Word2Vec and LSA stresses that LSA tends to perform
better when the training data is small, whereas Word2Vec requires large
datasets to train meaningful vectors ￼.  Our dataset
contains only a few thousand emails, so LSA is an appropriate choice.

Evidence of LSA effectiveness in phishing detection

In the security research community, LSA has been successfully applied to
phishing email classification.  A study by L’Huillier et al. proposed a
phishing detection pipeline that combined keyword extraction with LSA‐derived
features.  They demonstrated that LSA features combined with simple
classifiers (support‑vector machines, naïve Bayes and logistic regression)
achieved competitive results on benchmark datasets ￼.
The authors highlighted that performing singular value decomposition on the
TF‑IDF matrix reveals underlying semantic relationships between terms and
documents ￼, making it well‑suited for distinguishing
phishing from legitimate emails.

Advantages of LSA
	•	Deterministic and lightweight – Unlike neural embeddings, LSA uses linear
algebra and does not rely on gradient‑based training, making it fast and
stable.  It scales linearly with the number of components and operates on
sparse matrices, so memory usage is manageable for thousands of emails.
	•	Free and open source – Scikit‑learn and Gensim provide LSA implementations
under permissive licences.  The Gensim project markets itself as a free
library for training large‑scale semantic models, representing text as
semantic vectors and finding related documents ￼.  It
emphasises data streaming and the ability to process arbitrarily large
corpora ￼, although our current corpus is modest.
	•	Works with small datasets – Because LSA is count‑based and does not try to
learn context prediction parameters, it does not overfit small corpora.
	•	Produces interpretable embeddings – Each dimension corresponds to a latent
topic.  The explained variance ratio allows inspection of how much
information each component carries.

Implementation plan

1. Text preprocessing

Basic preprocessing should normalise case, remove HTML tags, strip URLs,
email addresses, numbers and punctuation, and collapse whitespace.  These
operations do not require heavy computation and reduce noise in the TF–IDF
matrix.  It is advisable to retain stop words only where they carry
important cues; however, for email classification the built‑in English stop
list in scikit‑learn is appropriate.

2. TF–IDF vectorisation

To convert preprocessed emails into numeric form, use TfidfVectorizer with
the following settings (informed by scikit‑learn’s guidelines ￼):