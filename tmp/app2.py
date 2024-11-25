import json
import spacy
import logging
from typing import List, Dict

# Initialize SpaCy model
nlp = spacy.load("en_core_web_sm")

# Configure logging to write into ./log/app.log
logging.basicConfig(filename='./log/app.log', level=logging.INFO, format='%(message)s')

def extract_themes_and_classify(data: dict) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Extract themes, opinions, and classify them as Negative, Neutral, or Positive.
    Also log the result as a JSON stream into a log file.
    
    :param data: Dictionary containing the input transcript data.
    :return: List of themes with their respective opinions and classifications.
    """
    themes = {}
    transcript_segments = data.get("payload", {}).get("transcript_segments", [])
    
    for segment in transcript_segments:
        text = segment.get("text", "")
        doc = nlp(text)
        
        # Identify themes using noun chunks
        for chunk in doc.noun_chunks:
            theme = chunk.text.strip().capitalize()
            if theme not in themes:
                themes[theme] = []
        
        # Extract opinions and classify them
        for sent in doc.sents:
            opinion = sent.text.strip()
            classification = classify_opinion(opinion)
            
            # Log each theme and opinion as JSON
            log_json_data(theme, opinion, classification)
            
            themes.setdefault("General", []).append({"opinion": opinion, "classification": classification})
    
    # Format themes and their respective opinions for return
    result = [{"theme": theme, "opinions": opinions} for theme, opinions in themes.items()]
    return result

def classify_opinion(opinion: str) -> str:
    """
    Classify the sentiment of an opinion as Negative, Neutral, or Positive.
    
    :param opinion: Opinion text.
    :return: Sentiment classification as a string.
    """
    opinion = opinion.lower()
    positive_keywords = ["smooth", "accessible", "innovative", "positive", "appealing", "convenient"]
    negative_keywords = ["misleading", "risk", "liquidation", "complex", "stressful", "inaccurate"]
    
    # Classification based on keyword presence
    if any(word in opinion for word in positive_keywords):
        return "Positive"
    elif any(word in opinion for word in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

def log_json_data(theme: str, opinion: str, classification: str):
    """
    Logs the theme, opinion, and classification as a JSON object to the log file.
    
    :param theme: The extracted theme from the transcript.
    :param opinion: The opinion text associated with the theme.
    :param classification: The sentiment classification of the opinion.
    """
    log_entry = {
        "Theme": theme,
        "Opinion": opinion,
        "Classification": classification
    }
    
    # Write JSON data as a log entry
    logging.info(json.dumps(log_entry))

def sort_and_process(themes_of_interest, sentiment: str, score: float):
    """
    Function that processes and sorts the themes based on sentiment.
    
    :param themes_of_interest: List of themes to process.
    :param sentiment: The sentiment to filter by.
    :param score: The score to associate with the sentiment.
    :return: Matching theme, sentiment, and score.
    """
    for theme in themes_of_interest:
        if sentiment.lower() in theme['omiSentiment'].lower():
            return theme, sentiment, score
    return None, sentiment, score

# Example input payload
input_data = {
    "timestamp": "2024-11-24T13:29:44.313171",
    "uid": "user8888?uid=JYow9cHgC2SdkqV3jINGNJ9RNDQ2",
    "ip_address": "34.96.46.36",
    "payload": {
        "id": "443eebd5-4158-4b23-8f6b-cac574c59724",
        "created_at": "2024-11-24T13:25:59.353348+00:00",
        "started_at": "2024-11-24T13:25:59.353348+00:00",
        "finished_at": "2024-11-24T13:27:42.784900+00:00",
        "transcript_segments": [
            {"text": "Centralized exchanges provide smooth user experiences and offer appealing deals.", "speaker": "SPEAKER_0", "is_user": False},
            {"text": "However, lending products often use misleading terms and come with risks.", "speaker": "SPEAKER_1", "is_user": False},
            {"text": "Dual asset strategies are complex and risky, not suitable for stress-free passive income.", "speaker": "SPEAKER_2", "is_user": False}
        ]
    }
}

# Run the function and display the results
themes_and_opinions = extract_themes_and_classify(input_data)

# Example usage of the sort_and_process function with mock data
themes_of_interest = [{"omiSentiment": "Positive", "theme": "Centralized exchanges"}]
sentiment = "Positive"
score = 5.0

theme, sentiment, score = sort_and_process(themes_of_interest, sentiment, score)

print(f"Processed Theme: {theme}, Sentiment: {sentiment}, Score: {score}")
