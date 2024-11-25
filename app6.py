import json
import logging
import string
from typing import List, Dict

import nltk
from gensim.summarization import summarize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from stop_words import get_stop_words

# Configure logging
logging.basicConfig(
    filename='./log/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Force NLTK to use the data directory in the virtual environment
nltk.data.path.append('/var/opinionway-webhook/venv/nltk_data')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")

# Initialize LSA summarizer
try:
    summarizer = LsaSummarizer(Stemmer('english'))
    summarizer.stop_words = get_stop_words('english')
except Exception as e:
    logging.error(f"Failed to initialize summarizer: {str(e)}")
    summarizer = None


def clean_data(data: Dict[str, any]) -> Dict[str, any]:
    """
    Clean the data by replacing 'null' strings with None.

    Args:
        data (dict): The data to clean.

    Returns:
        dict: Cleaned data with 'null' replaced by None.
    """
    if isinstance(data, dict):
        return {key: clean_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    return None if data == 'null' else data


def normalize_text(text: str) -> str:
    """
    Normalize text by removing stop words and punctuation.

    Args:
        text (str): Input text to normalize.

    Returns:
        str: Normalized text.
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


def extract_main_clause(text: str) -> str:
    """
    Extract the main clause from text for a focused summary.

    Args:
        text (str): Input text to process.

    Returns:
        str: Extracted main clause or a shortened fallback.
    """
    tokens = word_tokenize(text)
    return " ".join(tokens[:12]) if tokens else text


def summarize_opinion_with_gensim(opinion: str, word_count: int = 12) -> str:
    """
    Create a concise summary of the opinion using Gensim or fallback.

    Args:
        opinion (str): Opinion text to summarize.
        word_count (int): Desired word count for the summary.

    Returns:
        str: Summarized text.
    """
    if not opinion.strip():
        return ""

    try:
        summary = summarize(opinion, word_count=word_count)
        if not summary.strip():
            return extract_main_clause(opinion)
        return extract_main_clause(summary)
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}")
        return extract_main_clause(opinion)


def summarize_theme(theme_description: str, max_words: int = 3) -> str:
    """
    Summarize a theme to a specific number of words.

    Args:
        theme_description (str): Theme description text.
        max_words (int): Maximum words for the summary.

    Returns:
        str: Summarized theme.
    """
    return " ".join(theme_description.split()[:max_words])


def classify_opinion(opinion: str) -> str:
    """
    Classify opinion sentiment as Positive, Negative, or Neutral.

    Args:
        opinion (str): Text of the opinion.

    Returns:
        str: Sentiment classification.
    """
    sentiment_markers = {
        "positive": ["smooth", "accessible", "appealing", "good", "great", "excellent"],
        "negative": ["misleading", "complex", "bad", "difficult", "complicated"]
    }

    opinion = opinion.lower()
    positive_score = sum(word in opinion for word in sentiment_markers["positive"])
    negative_score = sum(word in opinion for word in sentiment_markers["negative"])

    if positive_score > negative_score:
        return "Positive"
    elif negative_score > positive_score:
        return "Negative"
    return "Neutral"


def extract_themes_and_classify(data: Dict[str, any]) -> List[Dict[str, str]]:
    """
    Extract themes and classify opinions in transcript data.

    Args:
        data (dict): Input transcript data.

    Returns:
        list: List of themes, summarized opinions, and classifications.
    """
    themes_of_interest = [
        {
            "theme": "Centralized exchanges",
            "keywords": ["exchange", "trading fee"],
            "context_words": ["platform", "trading"]
        }
    ]

    results = []
    transcript_segments = data.get("payload", {}).get("transcript_segments", [])
 
    for segment in transcript_segments:
        text = segment.get("text", "")
        normalized_text = normalize_text(text)

        for theme in themes_of_interest:
            if any(kw in normalized_text for kw in theme["keywords"]):
                opinion = text.strip()
                results.append({
                    "theme": summarize_theme(theme["theme"]),
                    "opinion": summarize_opinion_with_gensim(opinion),
                    "classification": classify_opinion(opinion)
                })

    return results


if __name__ == "__main__":
    try:
        with open('input.json', 'r') as f:
            input_data = json.load(f)
        cleaned_data = clean_data(input_data)
        results = extract_themes_and_classify(cleaned_data)
        print(json.dumps(results, indent=2))
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
