import json
import spacy
import logging
from typing import List, Dict

# Initialize SpaCy model for text processing
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
    themes_of_interest = [
        {"theme": "Centralized exchanges", "keywords": ["centralized exchange", "smooth user experience", "trading fee discounts"]},
        {"theme": "Lending products", "keywords": ["lending", "misleading terms", "risk", "liquidation"]},
        {"theme": "Dual asset strategies", "keywords": ["dual asset", "complex", "risky", "stress-free", "passive income"]}
    ]

    result = []

    transcript_segments = data.get("payload", {}).get("transcript_segments", [])
    
    # Process each segment in the transcript
    for segment in transcript_segments:
        text = segment.get("text", "")
        doc = nlp(text)
        
        # Check if the text matches any of the themes of interest
        for theme in themes_of_interest:
            if any(keyword.lower() in text.lower() for keyword in theme['keywords']):
                opinion = text.strip()
                classification = classify_opinion(opinion)
                log_json_data(theme['theme'], opinion, classification)
                result.append({"theme": theme['theme'], "opinion": opinion, "classification": classification})

    return result

def classify_opinion(opinion: str) -> str:
    """
    Classify the sentiment of an opinion as Negative, Neutral, or Positive.
    
    :param opinion: Opinion text.
    :return: Sentiment classification as a string.
    """
    opinion = opinion.lower()
    positive_keywords = ["smooth", "accessible", "appealing", "convenient", "positive"]
    negative_keywords = ["misleading", "risk", "liquidation", "complex", "risky", "stressful"]
    
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

# Print the results
for entry in themes_and_opinions:
    print(f"Theme: {entry['theme']}")
    print(f" - Opinion: {entry['opinion']}")
    print(f" - Classification: {entry['classification']}")
