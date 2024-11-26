from flask import Flask, request, jsonify
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from transformers import pipeline
import os
import zipfile
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Create log directory if it doesn't exist
log_directory = "log"
os.makedirs(log_directory, exist_ok=True)

# Load point-of-interest themes from a JSON file
def load_themes_of_interest():
    try:
        with open('point-of-interest.json', 'r') as f:
            return json.load(f)["themesOfInterest"]
    except FileNotFoundError:
        logging.warning("point-of-interest.json not found. Proceeding with empty themes.")
        return []

# Initialize models and configurations
themes_of_interest = load_themes_of_interest()
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Set up logging
log_filename = os.path.join(log_directory, "app.log")
handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1)
handler.suffix = "%Y-%m-%d"
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# Log start of the application
start_message = f"####### Application Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} #######"
logging.info(start_message)

def summarize_opinion(text: str, max_length: int = 12) -> str:
    """
    Summarize an opinion using a Hugging Face summarization model.

    Args:
        text (str): Input text.
        max_length (int): Maximum length of the summary.

    Returns:
        str: Summarized text.
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=20, do_sample=False)
        return summary[0]['summary_text'] if summary else text[:100]
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return text[:100]

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): Input text for sentiment analysis.

    Returns:
        tuple: Sentiment label and confidence score.
    """
    try:
        result = sentiment_analyzer(text)
        return result[0]['label'], result[0]['score']
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return "NEUTRAL", 0.5

def parse_payload(payload):
    """
    Parse and structure the incoming payload.

    Args:
        payload (dict): Incoming JSON payload.

    Returns:
        dict: Parsed and structured payload.
    """
    try:
        parsed_fields = {
            "id": payload.get("id"),
            "created_at": payload.get("created_at"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "transcript_segments": [
                {
                    "text": seg.get("text", ""),
                    "speaker": seg.get("speaker", "Unknown"),
                    "speakerId": seg.get("speakerId"),
                    "is_user": seg.get("is_user", False),
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
    except Exception as e:
        logging.error(f"Payload parsing error: {e}")
        return {}

def log_request_response(req, payload, response, uid):
    """
    Log the incoming request and outgoing response.

    Args:
        req (flask.Request): Flask request object.
        payload (dict): Parsed payload.
        response (str): Response type.
        uid (str): User ID.
    """
    try:
        ip_address = req.remote_addr
        log_message = {
            "timestamp": datetime.now().isoformat(),
            "uid": uid,
            "ip_address": ip_address,
            "payload_id": payload.get("id"),
            "response_type": response
        }
        logging.info(json.dumps(log_message))
        rotate_logs()
    except Exception as e:
        logging.error(f"Logging error: {e}")

def rotate_logs():
    """
    Rotate log files, keeping only the most recent 7 days.
    """
    try:
        log_files = sorted([f for f in os.listdir(log_directory) if f.endswith(".log")])

        if len(log_files) > 7:
            oldest_log = os.path.join(log_directory, log_files[0])
            with zipfile.ZipFile(f"{oldest_log}.zip", 'w') as zipf:
                zipf.write(oldest_log, arcname=log_files[0])
            os.remove(oldest_log)
    except Exception as e:
        logging.error(f"Log rotation error: {e}")

@app.route('/opinionway', methods=['POST'])
def opinionway():
    """
    Process incoming webhook data for opinion analysis.
    
    Returns:
        JSON response with analyzed transcript segments.
    """
    try:
        # Validate incoming JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Parse payload and extract metadata
        payload_fields = parse_payload(data)
        uid = request.args.get("uid", "unknown")

        # Log the request
        log_request_response(request, payload_fields, "processing", uid)
        
        # Initialize results list
        opinion_list = []
        
        # Process each transcript segment
        for segment in payload_fields.get('transcript_segments', []):
            transcript_text = segment.get('text', '').strip()
            
            # Skip empty transcripts
            if not transcript_text:
                continue
            
            try:
                # Analyze sentiment and summarize
                sentiment, score = analyze_sentiment(transcript_text)
                opinion = summarize_opinion(transcript_text)
                
                # Create detailed segment analysis
                segment_analysis = {
                    "text": transcript_text,
                    "sentiment": {
                        "label": sentiment,
                        "score": score
                    },
                    "summary": opinion,
                    "metadata": {
                        "speaker": segment.get('speaker', 'Unknown'),
                        "is_user": segment.get('is_user', False),
                        "start_time": segment.get('start'),
                        "end_time": segment.get('end')
                    }
                }
                
                opinion_list.append(segment_analysis)
                
                # Log processing details
                logging.info(f"Processed segment - Sentiment: {sentiment}, Score: {score}")
            
            except Exception as segment_error:
                logging.error(f"Error processing transcript segment: {segment_error}")
                continue

        # Return comprehensive analysis results
        return jsonify({
            "request_id": payload_fields.get("id"),
            "uid": uid,
            "total_segments": len(opinion_list),
            "opinions": opinion_list,
            "structured_data": payload_fields.get('structured', {})
        })

    except Exception as e:
        logging.error(f"Critical error processing request: {e}")
        return jsonify({
            "error": "Internal Server Error", 
            "details": str(e)
        }), 500

@app.route('/log', methods=['GET'])
def get_logs():
    """
    Retrieve application logs for debugging.
    
    Returns:
        JSON response with log contents.
    """
    try:
        with open(log_filename, 'r') as f:
            logs = f.readlines()
        return jsonify({"logs": logs[-100:]})  # Return last 100 log entries
    except FileNotFoundError:
        return jsonify({"error": "Log file not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)