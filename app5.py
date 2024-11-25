import json
import spacy
import logging
from typing import List, Dict, Optional, Union
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from gensim.summarization import summarize
import nltk

# Force NLTK to use the data directory in your virtual environment
nltk.data.path.append('/var/opinionway-webhook/venv/nltk_data')

# Initialize SpaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(filename='./log/app.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")

# Initialize summarizer
try:
    summarizer = LsaSummarizer(Stemmer('english'))
    summarizer.stop_words = get_stop_words('english')
except Exception as e:
    logging.error(f"Failed to initialize summarizer: {str(e)}")
    summarizer = None

def clean_data(data: Dict[str, any]) -> Dict[str, any]:
    """Clean the data by replacing 'null' with None."""
    if isinstance(data, dict):
        return {key: clean_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    return None if data == 'null' else data

def normalize_text(text: str) -> str:
    """Normalize text by removing stop words and punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    return " ".join([token.text for token in doc if token.text not in STOP_WORDS])

def extract_main_clause(text: str) -> str:
    """Extract the main clause from text for a clear, focused summary."""
    doc = nlp(text)
    
    main_parts = []
    for token in doc:
        if token.dep_ == "ROOT":
            # Get subject
            subject = next((t for t in token.lefts if t.dep_ == "nsubj"), None)
            if subject:
                main_parts.extend([t.text for t in subject.subtree])
            
            # Get verb
            main_parts.append(token.text)
            
            # Get object
            for right_token in token.rights:
                if right_token.dep_ in ["dobj", "attr", "ccomp"]:
                    main_parts.extend([t.text for t in right_token.subtree])
                    break
            break
    
    if not main_parts:
        return " ".join(text.split()[:8])
    
    summary = " ".join(main_parts)
    return " ".join(summary.split()[:12])

#########################################################################3333
def summarize_opinion_with_llm(opinion: str) -> str:
    """Create a clear, single-sentence summary of the opinion."""
    if not opinion.strip():
        return ""
    
    try:
        if not summarizer:
            return extract_main_clause(opinion)
        
        parser = PlaintextParser.from_string(opinion, Tokenizer('english'))
        summary = summarizer(parser.document, sentences_count=1)
        
        if not summary:
            return extract_main_clause(opinion)
        
        return extract_main_clause(str(summary[0]))
        
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}")
        return extract_main_clause(opinion)
##############################################################################33
def summarize_theme(theme_description: str, max_words: int = 3) -> str:
    """Summarize theme to specified number of words."""
    words = theme_description.split()
    return " ".join(words[:max_words])

def classify_opinion(opinion: str) -> str:
    """Classify opinion sentiment with improved accuracy."""
    opinion = opinion.lower()
    
    sentiment_markers = {
        "positive": {
            "smooth": 1,
            "accessible": 1,
            "appealing": 1,
            "convenient": 1,
            "positive": 1,
            "easy": 1,
            "stress-free": 1,
            "innovative": 1,
            "good": 1,
            "great": 1,
            "excellent": 1
        },
        "negative": {
            "misleading": 2,
            "risk": 1.5,
            "liquidation": 1.5,
            "complex": 1,
            "risky": 1.5,
            "stressful": 1,
            "bad": 1,
            "dangerous": 1.5,
            "difficult": 1,
            "complicated": 1,
            "confusing": 1.5
        }
    }
    
    positive_score = sum(weight for word, weight in sentiment_markers["positive"].items() 
                        if word in opinion)
    negative_score = sum(weight for word, weight in sentiment_markers["negative"].items() 
                        if word in opinion)
    
    # Check for negation
    negation_words = ["not", "no", "never", "neither", "nor", "without", "isn't", "aren't"]
    contains_negation = any(word in opinion.split() for word in negation_words)
    
    if contains_negation:
        positive_score, negative_score = negative_score, positive_score
    
    # Determine classification
    if positive_score > negative_score and positive_score > 0:
        return "Positive"
    elif negative_score > positive_score and negative_score > 0:
        return "Negative"
    return "Neutral"

def extract_themes_and_classify(data: Dict[str, any]) -> List[Dict[str, str]]:
    """Extract and classify themes from transcript data."""
    themes_of_interest = [
        {
            "theme": "Centralized exchanges",
            "keywords": ["centralized exchange", "smooth user experience", "trading fee", "bybit", "exchange"],
            "context_words": ["platform", "trading", "exchange"]
        },
        {
            "theme": "Lending products",
            "keywords": ["lending", "loan", "stake", "unstake", "savings"],
            "context_words": ["interest", "yield", "term"]
        },
        {
            "theme": "Dual asset strategies",
            "keywords": ["dual asset", "options trading", "derivatives"],
            "context_words": ["APR", "premium", "trading"]
        }
    ]

    result = []
    transcript_segments = data.get("payload", {}).get("transcript_segments", [])
    
    for segment in transcript_segments:
        text = segment.get("text", "")
        normalized_text = normalize_text(text)
        
        for theme in themes_of_interest:
            keyword_match = any(keyword.lower() in normalized_text for keyword in theme['keywords'])
            context_match = any(word.lower() in normalized_text for word in theme['context_words'])
            
            if keyword_match and context_match:
                opinion = text.strip()
                summarized_theme = summarize_theme(theme["theme"])
                summarized_opinion = summarize_opinion_with_llm(opinion)
                classification = classify_opinion(opinion)
                
                if len(summarized_opinion.split()) >= 3:
                    log_entry = {
                        "theme": summarized_theme,
                        "opinion": summarized_opinion,
                        "classification": classification
                    }
                    log_json_data(log_entry)
                    result.append(log_entry)

    return result

def log_json_data(log_entry: Dict[str, str]) -> None:
    """Log the analysis results."""
    try:
        logging.info(json.dumps(log_entry))
    except Exception as e:
        logging.error(f"Failed to log entry: {str(e)}")

if __name__ == "__main__":
    try:
        # Load and process input data
        with open('input.json', 'r') as f:
            input_data = json.load(f)
        
        # Clean and process the data
        cleaned_data = clean_data(input_data)
        results = extract_themes_and_classify(cleaned_data)
        
        # Print results
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        print(f"Error: {str(e)}")