import nltk
from nltk.tokenize import word_tokenize

# Force NLTK to use the data directory in your virtual environment
nltk.data.path.append('/var/opinionway-webhook/venv/nltk_data')

# Test text for tokenization
text = "This is a test sentence."

# Perform tokenization
try:
    tokens = word_tokenize(text)
    print("Tokenized words:", tokens)
except Exception as e:
    print("An error occurred:", str(e))
