"""
model_loader.py
----------------
Loads the fine-tuned ParsBERT model once (via AppConfig.ready) and exposes
`predict_sentiment()` for the view to call. This is the Django-adapted
version of the standalone predict.py script used earlier.
"""

import re

import torch
from django.conf import settings
from hazm import Normalizer, WordTokenizer, stopwords_list
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 64

# Populated by load_model(); left as None until the app is ready so an
# accidental import-time call fails loudly instead of silently doing nothing.
_tokenizer = None
_model = None
_device = None
_normalizer = None
_word_tokenizer = None
_stopwords = None


def load_model():
    """Load the tokenizer + model into memory. Called once at server startup."""
    global _tokenizer, _model, _device, _normalizer, _word_tokenizer, _stopwords

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = str(settings.SENTIMENT_MODEL_DIR)
    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device)
    _model.eval()

    _normalizer = Normalizer()
    _word_tokenizer = WordTokenizer()
    _stopwords = set(stopwords_list())

    print(f'[classifier] ParsBERT model loaded from {model_dir} on {_device}')


def _light_clean(text: str) -> str:
    """Same lightweight cleaning used before feeding text to ParsBERT during training."""
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', str(text))
    text = _normalizer.normalize(text)
    tokens = [t for t in _word_tokenizer.tokenize(text) if t not in _stopwords]
    return ' '.join(tokens)


def predict_sentiment(text: str) -> dict:
    """Return {'label': 'Happy' | 'Sad', 'confidence': float} for a raw Persian sentence."""
    if _model is None:
        raise RuntimeError('Model is not loaded yet — check apps.py / ready().')

    cleaned = _light_clean(text)
    inputs = _tokenizer(
        cleaned, padding='max_length', truncation=True,
        max_length=MAX_LEN, return_tensors='pt',
    ).to(_device)

    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=1)[0]

    label_id = probs.argmax().item()
    label = 'Happy' if label_id == 0 else 'Sad'
    return {'label': label, 'confidence': round(probs[label_id].item(), 3)}
