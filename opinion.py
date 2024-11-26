import asyncio
import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure the current directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opinion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('logs', exist_ok=True)

# Disable CUDA if not available
if not torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Pydantic Models for Request Validation
class TranscriptSegment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str = Field(default="", description="Transcript segment text")
    speaker: str = Field(default="Unknown", description="Speaker identifier")
    speakerId: Optional[int] = None
    is_user: bool = False
    start: Optional[float] = None
    end: Optional[float] = None

class StructuredData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    title: Optional[str] = None
    overview: Optional[str] = None
    category: Optional[str] = None
    emoji: Optional[str] = None

class OpinionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    transcript_segments: List[TranscriptSegment] = []
    structured: Optional[StructuredData] = None

class SentimentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    label: str
    score: float

class SegmentAnalysis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str
    sentiment: SentimentResult
    summary: str
    metadata: Dict[str, Any]

# Initialize ML Pipelines
class MLModels:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.init_models()
        return cls._instance

    def init_models(self):
        try:
            # Check if models can be loaded without GPU
            torch.cuda.set_device(-1)
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn"
            )
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def analyze_sentiment(self, text: str) -> SentimentResult:
        try:
            result = self.sentiment_analyzer(text)[0]
            return SentimentResult(
                label=result['label'], 
                score=result['score']
            )
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentResult(label="NEUTRAL", score=0.5)

    def summarize_text(self, text: str, max_length: int = 50) -> str:
        try:
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=20, 
                do_sample=False
            )
            return summary[0]['summary_text'] if summary else text[:100]
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return text[:100]

# Initialize FastAPI App
def create_app():
    app = FastAPI(
        title="Opinion Analysis API",
        description="Real-time transcript analysis with sentiment and summary",
        version="1.0.0"
    )
    return app

app = create_app()

# Global ML Models
ml_models = MLModels()

async def process_transcript_segment(segment: TranscriptSegment) -> SegmentAnalysis:
    """
    Asynchronously process a single transcript segment
    """
    # Remove empty or very short segments
    if not segment.text or len(segment.text.strip()) < 3:
        raise ValueError("Segment text is too short")

    # Analyze sentiment
    sentiment = ml_models.analyze_sentiment(segment.text)
    
    # Generate summary
    summary = ml_models.summarize_text(segment.text)

    # Create detailed segment analysis
    return SegmentAnalysis(
        text=segment.text,
        sentiment=sentiment,
        summary=summary,
        metadata={
            "speaker": segment.speaker,
            "is_user": segment.is_user,
            "start_time": segment.start,
            "end_time": segment.end
        }
    )

@app.post("/opinionway", response_model=Dict[str, Any])
async def process_opinion(
    request: OpinionRequest, 
    background_tasks: BackgroundTasks
):
    """
    Process incoming opinion data with async segment analysis
    """
    try:
        # Validate input
        if not request.transcript_segments:
            raise HTTPException(
                status_code=400, 
                detail="No transcript segments provided"
            )

        # Process segments concurrently
        segment_analyses = await asyncio.gather(
            *[process_transcript_segment(segment) 
              for segment in request.transcript_segments]
        )

        # Log processing details
        background_tasks.add_task(
            log_request, 
            request_id=request.id, 
            total_segments=len(segment_analyses)
        )

        # Prepare response
        return {
            "request_id": request.id,
            "total_segments": len(segment_analyses),
            "opinions": [
                analysis.model_dump() for analysis in segment_analyses
            ],
            "structured_data": request.structured.model_dump() if request.structured else {}
        }

    except Exception as e:
        logger.error(f"Error processing opinion: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal processing error: {str(e)}"
        )

async def log_request(request_id: str, total_segments: int):
    """
    Background logging task
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "total_segments": total_segments
        }
        
        # Log to file
        with open('logs/request_log.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Rotate logs if necessary
        rotate_logs()
    except Exception as e:
        logger.error(f"Logging error: {e}")

def rotate_logs(max_log_files: int = 7):
    """
    Rotate log files, keeping only the most recent entries
    """
    try:
        log_files = sorted(
            [f for f in os.listdir('logs') 
             if f.endswith('.json') and f.startswith('request_log')]
        )

        if len(log_files) > max_log_files:
            oldest_log = os.path.join('logs', log_files[0])
            with zipfile.ZipFile(f"{oldest_log}.zip", 'w') as zipf:
                zipf.write(oldest_log, arcname=log_files[0])
            os.remove(oldest_log)
    except Exception as e:
        logger.error(f"Log rotation error: {e}")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

def main():
    """
    Main entry point for the application
    """
    try:
        uvicorn.run(
            "opinion:app", 
            host="0.0.0.0", 
            port=8888, 
            reload=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()