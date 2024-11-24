#
from flask import Flask, request, jsonify
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from transformers import pipeline
from textblob import TextBlob
import os
import zipfile
from datetime import datetime

app = Flask(__name__)

# Create log directory if it doesn't exist
log_directory = "log"
os.makedirs(log_directory, exist_ok=True)

print(os.path.exists('point-of-interest.json'))
print(os.path.getsize('point-of-interest.json'))

# Load point-of-interest themes from a JSON file
with open('point-of-interest.json', 'r') as f:
    themes_of_interest = json.load(f)["themesOfInterest"]

# Initialize sentiment analysis model (using HuggingFace)
sentiment_analyzer = pipeline("sentiment-analysis")

# Set up logging
log_filename = os.path.join(log_directory, "app.log")
handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1)
handler.suffix = "%Y-%m-%d"
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Helper function to perform sentiment analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Helper function to match the theme based on sentiment
def match_theme(transcript):
    sentiment, score = analyze_sentiment(transcript)
    for theme in themes_of_interest:
        if sentiment.lower() in theme['omiSentiment'].lower():
            return theme, sentiment, score
    return None, sentiment, score

# Define the endpoint to receive the webhook data
@app.route('/opinionway', methods=['POST'])
def opinionway():
    data = request.get_json()
    transcript = data.get('transcript', '')
    matched_theme, sentiment, score = match_theme(transcript)

    if matched_theme:
        response = {
            "message": "Theme matched",
            "theme": matched_theme['theme'],
            "sentiment": sentiment,
            "score": score,
            "bet_outcomes": matched_theme['betOutcome']
        }
    else:
        response = {
            "message": "No theme matched",
            "sentiment": sentiment,
            "score": score
        }

    # Log the request and response
    log_request_response(request, data, response)

    return jsonify(response)

# Log function
def log_request_response(req, payload, response):
    uid = payload.get("uid", "unknown")
    ip_address = req.remote_addr
    log_message = {
        "timestamp": datetime.now().isoformat(),
        "uid": uid,
        "ip_address": ip_address,
        "payload": payload,
        "response": response
    }
    logging.info(json.dumps(log_message))
    rotate_logs()

# Log rotation
def rotate_logs():
    log_files = sorted([f for f in os.listdir(log_directory) if f.endswith(".log")])

    if len(log_files) > 7:
        oldest_log = os.path.join(log_directory, log_files[0])
        with zipfile.ZipFile(f"{oldest_log}.zip", 'w') as zipf:
            zipf.write(oldest_log, arcname=log_files[0])  # Ensure the original name is preserved
        os.remove(oldest_log)
@app.route('/log', methods=['GET'])
def get_logs():
    with open(log_filename, 'r') as f:
        logs = f.readlines()
    return jsonify({"logs": logs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
