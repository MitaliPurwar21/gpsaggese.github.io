"""
weave_utils.py

Utilities for the Weave FOMC tone project. The notebooks call these functions
and classes so the notebook cells stay short and readable.

- `init_weave()` loads my API keys from `.env` and starts a Weave project
- `TfidfLogRegModel`, `FomcRobertaModel`, and `ZeroShotModel` are the three
  hawkish/dovish/neutral classifiers, each written as a `weave.Model`
- `predict_sample()` runs several models on the same sentences so I can compare
  them side by side

Import as:

import weave_utils as wutils
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import pandas as pd
import pydantic
import weave
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent

# My API keys (WANDB_API_KEY and HF_TOKEN) live in this git-ignored file.
dotenv.load_dotenv(PROJECT_DIR / ".env")

# Hugging Face models are large, so I cache them under the git-ignored data/
# folder. That way they survive container restarts and I only download them
# once.
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "data" / "hf_cache"))

# This is the label encoding used by the Trillion Dollar Words dataset.
LABEL_NAMES: Dict[int, str] = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_IDS: Dict[str, int] = {name: idx for idx, name in LABEL_NAMES.items()}


def init_loggers(notebook_log: logging.Logger) -> None:
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Weave setup
# #############################################################################


def init_weave(project_name: str = "msml610-weave-fomc") -> Any:
    """
    Start a Weave project so every `@weave.op` call after this gets traced.

    :param project_name: name of the W&B project that stores the traces
    :return: the Weave client returned by `weave.init()`
    """
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is missing. Add it to the .env file in the project "
            "folder (see https://wandb.ai/authorize)."
        )
    # By default Weave prints a link after every traced call, which floods the
    # notebook when a model predicts hundreds of sentences. I turn that off and
    # browse the calls on the Traces page instead. I use the environment
    # variable because passing `settings=` to `weave.init` did not stop the
    # links: they are printed from a background thread that doesn't see it.
    os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")
    # The notebook logs at INFO level, which makes the HTTP clients print one
    # line per request: `httpx` for Weave's calls to W&B and `httpx2` for the
    # Hugging Face downloads. I only want to see their warnings.
    for http_logger in ("httpx", "httpx2"):
        logging.getLogger(http_logger).setLevel(logging.WARNING)
    client = weave.init(project_name)
    return client


def _format_prediction(label_id: int, confidence: float) -> Dict[str, Any]:
    """
    Put every model's output in the same shape.

    All three models return the numeric label (to compare with the dataset),
    the readable name, and the model's confidence in that label.
    """
    return {
        "label": int(label_id),
        "label_name": LABEL_NAMES[int(label_id)],
        "confidence": round(float(confidence), 4),
    }


# #############################################################################
# Model 1: TF-IDF + logistic regression baseline
# #############################################################################


class TfidfLogRegModel(weave.Model):
    """
    Baseline classifier: TF-IDF features followed by logistic regression.

    The hyperparameters are model attributes, so Weave stores a new version of
    the model whenever I change one of them.
    """

    ngram_range: Tuple[int, int] = (1, 2)
    max_features: int = 10000
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    # The fitted sklearn pipeline is private because Weave doesn't need to
    # serialize it, only the settings above.
    _pipeline: Optional[Pipeline] = pydantic.PrivateAttr(default=None)

    def fit(self, sentences: List[str], labels: List[int]) -> "TfidfLogRegModel":
        """
        Fit the TF-IDF vectorizer and the classifier on the training split.

        :param sentences: training sentences
        :param labels: integer labels (0 dovish, 1 hawkish, 2 neutral)
        :return: the fitted model, so I can chain calls
        """
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=self.ngram_range,
                        lowercase=True,
                        max_features=self.max_features,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self._pipeline.fit(list(sentences), [int(y) for y in labels])
        return self

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Classify one sentence.

        :param sentence: FOMC sentence to classify
        :return: dict with `label`, `label_name`, and `confidence`
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() on the training data before predict().")
        probabilities = self._pipeline.predict_proba([sentence])[0]
        classes = self._pipeline.classes_
        best = probabilities.argmax()
        return _format_prediction(classes[best], probabilities[best])


# #############################################################################
# Model 2: pre-trained FOMC-RoBERTa
# #############################################################################


def _label_id_from_hf(raw_label: str) -> int:
    """
    Turn a Hugging Face label such as `LABEL_1` or `Hawkish` into 0, 1, or 2.
    """
    raw_label = raw_label.strip()
    if raw_label.upper().startswith("LABEL_"):
        return int(raw_label.split("_")[-1])
    return LABEL_IDS[raw_label.lower()]


class FomcRobertaModel(weave.Model):
    """
    RoBERTa-large fine-tuned on FOMC sentences by the Trillion Dollar Words
    authors.

    The model uses the same label order as the dataset (LABEL_0 dovish,
    LABEL_1 hawkish, LABEL_2 neutral). The repository is gated on Hugging
    Face, so HF_TOKEN in `.env` has to belong to an account with access.
    """

    model_name: str = "gtfintechlab/FOMC-RoBERTa"
    max_length: int = 256
    _pipe: Any = pydantic.PrivateAttr(default=None)

    def _load(self) -> Any:
        # I load the model on the first prediction, not when the object is
        # created, so building the object is instant and the download only
        # happens once.
        if self._pipe is None:
            import transformers

            _LOG.info("Loading %s", self.model_name)
            self._pipe = transformers.pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,
            )
        return self._pipe

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Classify one sentence with FOMC-RoBERTa.

        :param sentence: FOMC sentence to classify
        :return: dict with `label`, `label_name`, and `confidence`
        """
        pipe = self._load()
        output = pipe(sentence, truncation=True, max_length=self.max_length)[0]
        return _format_prediction(_label_id_from_hf(output["label"]), output["score"])


# #############################################################################
# Model 3: zero-shot NLI classifier
# #############################################################################


class ZeroShotModel(weave.Model):
    """
    Zero-shot classifier built on a natural language inference model.

    It never sees our training data. For each sentence, it scores how well
    each label description fits, and I take the best one.
    """

    model_name: str = "facebook/bart-large-mnli"
    # I describe each label in plain words because the NLI model does better
    # with a short explanation than with the bare word "hawkish".
    label_descriptions: Dict[str, str] = {
        "dovish": "dovish, favoring lower interest rates or easier monetary policy",
        "hawkish": "hawkish, favoring higher interest rates or tighter monetary policy",
        "neutral": "neutral, with no clear monetary policy stance",
    }
    hypothesis_template: str = "The monetary policy stance of this sentence is {}."
    _pipe: Any = pydantic.PrivateAttr(default=None)

    def _load(self) -> Any:
        if self._pipe is None:
            import transformers

            _LOG.info("Loading %s", self.model_name)
            self._pipe = transformers.pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,
            )
        return self._pipe

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Classify one sentence without any task-specific training.

        :param sentence: FOMC sentence to classify
        :return: dict with `label`, `label_name`, and `confidence`
        """
        pipe = self._load()
        descriptions = list(self.label_descriptions.values())
        output = pipe(
            sentence,
            candidate_labels=descriptions,
            hypothesis_template=self.hypothesis_template,
        )
        # The pipeline sorts the labels by score, so the first one is the best.
        best_description = output["labels"][0]
        best_name = next(
            name
            for name, text in self.label_descriptions.items()
            if text == best_description
        )
        return _format_prediction(LABEL_IDS[best_name], output["scores"][0])


# #############################################################################
# Side-by-side comparison
# #############################################################################


def predict_sample(
    models: Dict[str, weave.Model],
    sample_df: pd.DataFrame,
    *,
    text_col: str = "sentence",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Run each model on the same sentences and line up the predictions.

    Each `predict()` call is a Weave op, so this also creates one trace per
    model and sentence.

    :param models: display name -> model
    :param sample_df: sentences with their true labels
    :param text_col: column with the sentence text
    :param label_col: column with the true integer label
    :return: one row per sentence, with the true label and each model's label
    """
    rows = []
    for _, row in sample_df.iterrows():
        result = {
            "sentence": row[text_col],
            "true_label": LABEL_NAMES[int(row[label_col])],
        }
        for name, model in models.items():
            prediction = model.predict(row[text_col])
            result[name] = prediction["label_name"]
        rows.append(result)
    return pd.DataFrame(rows)
