from flask import Flask, request, jsonify
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from transformers import pipeline # for sentiment analysis
import os
import zipfile
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Create log directory if it doesn't exist
log_directory = "log"
os.makedirs(log_directory, exist_ok=True)

# Load point-of-interest themes from a JSON file
with open('point-of-interest.json', 'r') as f:
    themes_of_interest = json.load(f)["themesOfInterest"]

# Initialize sentiment analysis model (using HuggingFace) from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Set up logging
log_filename = os.path.join(log_directory, "app.log")
handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1)
handler.suffix = "%Y-%m-%d"
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Log start of the application
start_message = f"####### start date = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} #######"
logging.info(start_message)

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
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Parse and log individual fields
        payload_fields = parse_payload(data)
        uid = request.args.get("uid", "unknown")  # Extract UID from query params
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
        log_request_response(request, payload_fields, response, uid)

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

def parse_payload(payload):
    """Parse and structure payload."""
    parsed_fields = {
        "id": payload.get("id"),
        "created_at": payload.get("created_at"),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "transcript": payload.get("transcript", ""),
        "transcript_segments": [
            {
                "text": seg.get("text"),
                "speaker": seg.get("speaker"),
                "speakerId": seg.get("speakerId"),
                "is_user": seg.get("is_user"),
                "start": seg.get("start"),
                "end": seg.get("end"),
            } for seg in payload.get("transcript_segments", [])
        ],
        "photos": payload.get("photos", []),
        "structured": payload.get("structured", {}),
        "apps_response": payload.get("apps_response", []),
        "discarded": payload.get("discarded", False),
    }
    return parsed_fields

def log_request_response(req, payload, response, uid):
    """Log the incoming request and outgoing response."""
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
            zipf.write(oldest_log, arcname=log_files[0])
        os.remove(oldest_log)

@app.route('/log', methods=['GET'])
def get_logs():
    """Return logs for debugging."""
    try:
        with open(log_filename, 'r') as f:
            logs = f.readlines()
        return jsonify({"logs": logs})
    except FileNotFoundError:
        return jsonify({"error": "Log file not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
