# snappfood-sentiment-analysis
Persian Sentiment Analysis for SnappFood Customer Reviews using Fine-Tuned ParsBERT and Django.
# SnappFood Persian Sentiment Analysis — Django App

A small Django web app that classifies Persian text as **Happy** (positive)
or **Sad** (negative) using a fine-tuned ParsBERT model. Type a sentence into
the form, get back the sentiment label and a confidence score.

The model was fine-tuned on the SnappFood review dataset — see
`snappfood_sentiment_analysis.ipynb` for the full training pipeline.

## Project Structure

```
sentiment_site/
├── manage.py
├── requirements.txt
│
├── sentiment_site/                        Project settings
│   ├── settings.py
│   └── urls.py
│
├── classifier/                            Sentiment-analysis app
│   ├── apps.py                            Loads the model once at server startup
│   ├── model_loader.py                    Model loading + predict_sentiment()
│   ├── views.py                           Handles the form (GET / POST)
│   ├── urls.py
│   └── templates/classifier/index.html    The form + result display
│
└── parsbert_finetuned/                    Trained model (see below)
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    └── tokenizer_config.json
```
## Installation

```bash
git clone <repo-url>
cd sentiment_site

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Add the Trained Model

The fine-tuned model (~600 MB) is not stored in this repo. Download it and
place it at the project root, next to `manage.py`:

```
sentiment_site/
├── parsbert_finetuned/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── tokenizer_config.json
└── manage.py
```

## Run

```bash
python manage.py migrate
python manage.py runserver
```

Open **http://127.0.0.1:8000/**, type a Persian sentence, and submit.


