# Dialogue-Bot

tested with Python3.8

## Installation (for development only)

Install package:
    
    pip install .
    
Download models (if needed):

    python -c "import stanza; stanza.download('en')"
    python -c "import stanza; stanza.download('de')"

    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm

    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords

## Setup

### Set up a running MongoDB instance

The chatbot needs access to a running MongoDB instance.
You can start a MongoDB instance using the provided Docker-Compose file:

    docker-compose -f mongo-db.yml up --build
    
If you need an admin interface for the MongoDB, you can use

    docker-compose -f mongo-express.yml up --build
    
### Create Bot

See `example_bot.py`

## Installation Troubles

Ubuntu issues

    sudo apt-get install libpq-dev
    pip install --upgrade pip