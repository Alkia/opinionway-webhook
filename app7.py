import logging
import string
import json
import nltk
import warnings
from typing import List, Dict, Tuple, Optional
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter
from textblob import TextBlob

# Filter out specific syntax warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="textblob")

# Configure logging with more detailed error reporting
logging.basicConfig(
    filename='./log/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpinionAnalyzer:
    def __init__(self):
        """Initialize the OpinionAnalyzer with error handling for NLTK downloads."""
        try:
            self.stop_words = set(stopwords.words('english'))
            self.initialize_sentiment_lexicon()
            self._ensure_nltk_downloads()
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            # Initialize with empty stop words if NLTK fails
            self.stop_words = set()
            self.initialize_sentiment_lexicon()

    def _ensure_nltk_downloads(self) -> None:
        """Ensure all required NLTK data is downloaded."""
        required_nltk_data = [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words'
        ]
        for item in required_nltk_data:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data {item}: {str(e)}")

    def initialize_sentiment_lexicon(self) -> None:
        """Initialize domain-specific sentiment lexicon with crypto/finance terms."""
        self.domain_lexicon = {
            'positive': {
                'accessible': 1.5,
                'secure': 1.5,
                'profitable': 1.8,
                'innovative': 1.2,
                'efficient': 1.3,
                'reliable': 1.4,
                'trustworthy': 1.6,
                'transparent': 1.4,
                'user-friendly': 1.3,
                'liquid': 1.2,
                'gain': 1.3,
                'growth': 1.4,
                'adoption': 1.2,
                'success': 1.5
            },
            'negative': {
                'risky': -1.5,
                'volatile': -1.2,
                'complex': -1.3,
                'expensive': -1.4,
                'centralized': -0.8,
                'hacked': -1.8,
                'scam': -1.9,
                'suspicious': -1.6,
                'unreliable': -1.5,
                'complicated': -1.3,
                'loss': -1.4,
                'crash': -1.7,
                'fraud': -1.9,
                'failure': -1.6
            }
        }

    def clean_data(self, data: Dict[str, any]) -> Dict[str, any]:
        """Clean the data by replacing 'null' strings with None."""
        try:
            if isinstance(data, dict):
                return {key: self.clean_data(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [self.clean_data(item) for item in data]
            return None if data == 'null' else data
        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            return data

    def extract_complete_sentence(self, text: str, keyword: str) -> str:
        """Extract complete sentences containing the keyword with error handling."""
        try:
            sentences = sent_tokenize(text)
            relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
            return ' '.join(relevant_sentences) if relevant_sentences else text
        except Exception as e:
            logger.warning(f"Sentence extraction failed: {str(e)}")
            return text

    def get_context_window(self, text: str, keyword: str, window_size: int = 20) -> str:
        """Get a context window around the keyword with error handling."""
        try:
            words = text.split()
            keyword_indices = [i for i, word in enumerate(words) 
                             if keyword.lower() in word.lower()]
            
            if not keyword_indices:
                return text
                
            # Use the first occurrence of the keyword
            keyword_index = keyword_indices[0]
            start = max(0, keyword_index - window_size)
            end = min(len(words), keyword_index + window_size + 1)
            return ' '.join(words[start:end])
        except Exception as e:
            logger.warning(f"Context window extraction failed: {str(e)}")
            return text

    def calculate_domain_sentiment(self, text: str) -> float:
        """Calculate sentiment score using domain-specific lexicon with error handling."""
        try:
            words = word_tokenize(text.lower())
            score = 0.0
            word_count = 0

            for word in words:
                if word in self.domain_lexicon['positive']:
                    score += self.domain_lexicon['positive'][word]
                    word_count += 1
                elif word in self.domain_lexicon['negative']:
                    score += self.domain_lexicon['negative'][word]
                    word_count += 1

            # Use TextBlob with error handling
            try:
                blob_sentiment = TextBlob(text).sentiment.polarity
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {str(e)}")
                blob_sentiment = 0.0

            # Weight domain-specific sentiment more heavily
            if word_count > 0:
                final_score = (0.7 * (score / word_count)) + (0.3 * blob_sentiment)
            else:
                final_score = blob_sentiment

            return max(min(final_score, 1.0), -1.0)  # Ensure score is between -1 and 1
        except Exception as e:
            logger.error(f"Sentiment calculation failed: {str(e)}")
            return 0.0

    def classify_opinion(self, text: str) -> Tuple[str, float]:
        """Classify opinion with confidence score and error handling."""
        try:
            sentiment_score = self.calculate_domain_sentiment(text)
            
            if sentiment_score > 0.3:
                classification = "Positive"
            elif sentiment_score < -0.3:
                classification = "Negative"
            else:
                classification = "Neutral"
                
            confidence = min(abs(sentiment_score) * 1.5, 1.0)  # Scale confidence but cap at 1.0
            return classification, confidence
        except Exception as e:
            logger.error(f"Opinion classification failed: {str(e)}")
            return "Neutral", 0.0

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using POS tagging with error handling."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            phrases = []
            current_phrase = []
            
            for word, tag in pos_tags:
                if tag.startswith(('NN', 'JJ', 'VB')):
                    current_phrase.append(word)
                else:
                    if current_phrase:
                        phrases.append(' '.join(current_phrase))
                        current_phrase = []
                        
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                
            return phrases
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {str(e)}")
            return []

    def summarize_opinion(self, text: str, theme_keywords: List[str]) -> str:
        """Create a meaningful summary of the opinion with error handling."""
        try:
            # Get relevant context for each keyword
            contexts = []
            for keyword in theme_keywords:
                context = self.extract_complete_sentence(text, keyword)
                if context:
                    contexts.append(context)

            if not contexts:
                return text[:100]  # Fallback to first 100 chars if no context found

            # Extract key phrases from the contexts
            all_phrases = []
            for context in contexts:
                phrases = self.extract_key_phrases(context)
                all_phrases.extend(phrases)

            # Count phrase frequencies
            phrase_counter = Counter(all_phrases)
            
            # Build summary using most relevant phrases
            summary_phrases = [phrase for phrase, _ in phrase_counter.most_common(5)]
            summary = ' '.join(summary_phrases)

            # Ensure the summary includes at least one theme keyword
            if not any(keyword.lower() in summary.lower() for keyword in theme_keywords):
                summary = f"{theme_keywords[0]} - {summary}"

            return summary[:150]  # Limit length but keep it meaningful
        except Exception as e:
            logger.error(f"Opinion summarization failed: {str(e)}")
            return text[:100]  # Fallback to simple truncation

    def extract_themes_and_classify(self, data: Dict[str, any]) -> List[Dict[str, any]]:
        """Extract themes and classify opinions with improved error handling."""
        try:
            themes_of_interest = [
                {
                    "theme": "Centralized exchanges",
                    "keywords": ["exchange", "trading fee", "centralized", "cex"],
                    "context_words": ["platform", "trading", "market", "liquidity", "fee", "deposit", "withdrawal"]
                }
            ]

            results = []
            transcript_segments = data.get("payload", {}).get("transcript_segments", [])
            
            for segment in transcript_segments:
                text = segment.get("text", "")
                if not text:  # Skip empty segments
                    continue
                
                for theme in themes_of_interest:
                    # Check for theme relevance
                    theme_keywords = theme["keywords"] + theme["context_words"]
                    if any(kw.lower() in text.lower() for kw in theme_keywords):
                        # Generate opinion summary
                        summary = self.summarize_opinion(text, theme["keywords"])
                        
                        # Classify with confidence
                        classification, confidence = self.classify_opinion(text)
                        
                        results.append({
                            "theme": theme["theme"],
                            "opinion": summary,
                            "classification": classification,
                            "confidence": round(confidence, 2),
                            "full_context": text
                        })

            return results
        except Exception as e:
            logger.error(f"Theme extraction and classification failed: {str(e)}")
            return []

def main():
    try:
        analyzer = OpinionAnalyzer()
        with open('input.json', 'r') as f:
            input_data = json.load(f)
        cleaned_data = analyzer.clean_data(input_data)
        results = analyzer.extract_themes_and_classify(cleaned_data)
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    main()
