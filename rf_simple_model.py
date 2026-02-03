import tarfile
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# ------------- CONFIG ----------------
SCRIPT_DIR = Path(__file__).resolve().parent

EASY_HAM_TAR = SCRIPT_DIR / "Datasets" / "easy_ham.tar.bz2"
SPAM_TAR = SCRIPT_DIR / "Datasets" / "spam.tar.bz2"

OUT_DIR = SCRIPT_DIR / "spamassassin_data"  # where archives will be extracted
MODEL_PATH = SCRIPT_DIR / "logreg_spamassassin_email_model.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF settings
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
STOP_WORDS = "english"
# -------------------------------------


def extract_if_needed(tar_path: Path, extract_dir: Path) -> None:
    """Extract tar.bz2 archive into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract only if the target folder isn't already present
    with tarfile.open(tar_path, "r:bz2") as tar:
        # Python 3.14+ will change default filtering behavior; setting filter keeps it explicit.
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            # Older Python versions don't support `filter=...`
            tar.extractall(extract_dir)


def read_all_text_files(folder: Path, label: int):
    """Read every file in folder as text, assign the given label."""
    texts, labels = [], []

    if not folder.exists():
        raise FileNotFoundError(f"Expected folder not found: {folder}")

    for fp in folder.iterdir():
        if fp.is_file():
            text = fp.read_text(encoding="latin-1", errors="ignore")
            texts.append(text)
            labels.append(label)

    return texts, labels


def load_spamassassin(easy_ham_tar: Path, spam_tar: Path):
    """
    Load SpamAssassin easy_ham + spam datasets.
    Returns:
        texts: list[str]
        labels: list[int]  (0 = ham, 1 = spam)
    """
    base_dir = Path(OUT_DIR)
    ham_dir = base_dir / "easy_ham"
    spam_dir = base_dir / "spam"

    if not ham_dir.exists():
        extract_if_needed(easy_ham_tar, base_dir)
    if not spam_dir.exists():
        extract_if_needed(spam_tar, base_dir)

    ham_texts, ham_labels = read_all_text_files(ham_dir, label=0)
    spam_texts, spam_labels = read_all_text_files(spam_dir, label=1)

    return ham_texts + spam_texts, ham_labels + spam_labels


def train_model(texts, labels):
    """TF-IDF -> LogisticRegression training + evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words=STOP_WORDS,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC: {auc:.4f}")
    except Exception as e:
        print(f"\nCould not compute ROC AUC: {e}")

    return model, vectorizer


def main():
    print("Loading SpamAssassin easy_ham + spam ...")

    if not EASY_HAM_TAR.exists():
        raise FileNotFoundError(f"Missing: {EASY_HAM_TAR}")
    if not SPAM_TAR.exists():
        raise FileNotFoundError(f"Missing: {SPAM_TAR}")

    texts, labels = load_spamassassin(EASY_HAM_TAR, SPAM_TAR)

    print(f"Loaded {len(texts)} emails total")
    print(f"Ham: {labels.count(0)} | Spam: {labels.count(1)}")

    print("\nTraining Logistic Regression model ...")
    model, vectorizer = train_model(texts, labels)

    print(f"\nSaving model + vectorizer to {MODEL_PATH} ...")
    joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
