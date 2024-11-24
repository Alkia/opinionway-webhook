import json
import spacy
import logging
from typing import List, Dict
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Initialize SpaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Configure logging to write into ./log/app.log
logging.basicConfig(filename='./log/app.log', level=logging.INFO, format='%(message)s')

def clean_data(data: dict) -> dict:
    """
    Recursively clean the data by replacing 'null' with None.
    
    :param data: The original JSON dictionary to clean.
    :return: A cleaned version of the dictionary with 'null' replaced by None.
    """
    # Check if data is a dictionary
    if isinstance(data, dict):
        return {key: clean_data(value) for key, value in data.items()}
    
    # Check if data is a list
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    
    # If data is 'null', replace it with None (this is done implicitly as there's no 'null' type in Python)
    elif data is None:
        return None
    
    # Otherwise, return the data as is
    return data

def normalize_text(text: str) -> str:
    """
    Normalize the text by removing stop words, punctuation, and normalizing case.
    
    :param text: The input text to normalize.
    :return: A normalized version of the text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize and remove stop words using SpaCy
    doc = nlp(text)
    return " ".join([token.text for token in doc if token.text not in STOP_WORDS])

def summarize_theme(theme_description: str, max_words: int = 5) -> str:
    """
    Summarizes the theme description to a maximum number of words.
    
    :param theme_description: The full theme description.
    :param max_words: The maximum number of words to keep.
    :return: A summarized version of the theme description.
    """
    # Tokenize and return the first `max_words` from the description
    words = theme_description.split()
    return " ".join(words[:max_words])

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
        {"theme": "Dual asset strategies", "keywords": ["dual asset", "complex", "risky", "stress-free", "passive income"]},
    ]

    result = []

    transcript_segments = data.get("payload", {}).get("transcript_segments", [])
    
    # Process each segment in the transcript
    for segment in transcript_segments:
        text = segment.get("text", "")
        normalized_text = normalize_text(text)
        
        # Check if the normalized text matches any of the themes of interest
        for theme in themes_of_interest:
            summarized_theme = summarize_theme(theme["theme"])
            if any(keyword.lower() in normalized_text for keyword in theme['keywords']):
                opinion = text.strip()
                classification = classify_opinion(opinion)
                log_json_data(summarized_theme, opinion, classification)
                result.append({"theme": summarized_theme, "opinion": opinion, "classification": classification})

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

# Example input payload (the same as your input_data)
input_data = {
    "timestamp": "2024-11-24T13:21:23.022982",
    "uid": "user8888?uid=JYow9cHgC2SdkqV3jINGNJ9RNDQ2",
    "ip_address": "34.96.46.36",
    "payload": {
        "id": "a85d91bd-17cf-4078-b072-78c034e70532",
        "created_at": "2024-11-24T13:14:10.446403+00:00",
        "started_at": "2024-11-24T13:14:10.446403+00:00",
        "finished_at": "2024-11-24T13:19:07.976724+00:00",
        "transcript": "",
        "transcript_segments": [
            {
                "text": "Of centralized exchanges too. This generally means a more accessible and smooth user experience. So if that's more your speed, then you head to the Conduero deals page, you can get trading fee discounts of up to ฿5,000. Lending, for example, is possible on most BEDs. On Vidbit, it's called Vidbit savings. Now bybit uses terms like state and unstake This is a bit misleading because this product does not delegate your crypto to a validator. Stake blockchain. It's just lending your crypto to other traders on FID in exchange for interest. You can go for a flexible loan term that allows you to withdraw your assets Or if you want a slightly higher yield, you can go for a fixed term loan, your crypto will be locked up and inaccessible. Duration. When lending on an exchange, aside from the fundamental risk of keeping your asset on an exchange, you're also relying on the liquidation engine not having any bad debts that can't be repaid.",
                "speaker": "SPEAKER_0",
                "speakerId": 1,
                "is_user": False,
                "start": 0,
                "end": 83.76925299999999
            },
            {
                "text": "risk of taking off. Moving on. Like many other exchanges, bybit has a due product will show you some eye popping APRs. Without knowing anything else, this already tells you That's because dual asset is crypto options trading by another name. Now options are complex and risky derivatives, and trading them is not exactly the most passive strategy out there either. But since Bridget put it on their passive income platform, That's explained. The API you see comes from other market participants on Vivid paying you a premium in exchange for the option to trade a future debt. This date is often only a few days into the future, meaning that the annualized returns you see can be misleading. Look. I'm gonna give it to you straight. Options trading is not If you want a relatively easy and stress free way to bring in passive income in the long term, then well, dual asset is not for you. Not financial advice, of course. Next up, the Christie Monmouth. Now this sounds cool, but it's worth pointing out that it's, well, just LP and by another name. These exchanges love to rebrand existing strategies, and I have to admit it, It sounds better than our previous. So Findit provides you with a really convenient interface to provide liquidity to an automated market maker You earn a yield at a cruise day, and it varies depending on trading volume. At the time of making this video, they're advertising at a 23% annualized USDT. So if I add $1,000 worth of This year for passive income I could expect about 200 bucks. Nice."
                "speaker": "SPEAKER_0",
                "speakerId": 1,
                "is_user": False,
                "start": 84,
                "end": 163.76925299999999
            }
        ]
    }
}

# Run the extraction and classification function
results = extract_themes_and_classify(input_data)

# Print the results
for entry in results:
    print(f"Theme: {entry['theme']}")
    print(f" - Opinion: {entry['opinion']}")
    print(f" - Classification: {entry['classification']}")
