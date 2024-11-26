import asyncio
import dataclasses
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

# Third-party imports
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, validator
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcript_analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class AnalysisConfig:
    """Configuration for transcript analysis"""
    max_segment_length: int = 512
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    summarization_model: str = "facebook/bart-large-cnn"
    device: str = "cpu"

class TranscriptSegment(BaseModel):
    """Represents a single segment of a transcript"""
    text: str = Field(min_length=3, max_length=1024)
    speaker: str = "Unknown"
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = {}

    @validator('text')
    def clean_text(cls, v):
        """Sanitize and validate text input"""
        return v.strip()

class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    label: str
    confidence: float

class SegmentAnalysis(BaseModel):
    """Comprehensive analysis of a transcript segment"""
    original_text: str
    sentiment: SentimentResult
    summary: str
    keywords: List[str]
    language: str

class TranscriptAnalyzer:
    """Core ML-powered transcript analysis engine"""
    _instance = None

    def __new__(cls, config: AnalysisConfig = AnalysisConfig()):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, config: AnalysisConfig):
        """Initialize ML pipelines with robust error handling"""
        try:
            # Ensure CPU usage
            torch.cuda.set_device(-1)

            self.config = config
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=config.sentiment_model,
                device=-1  # Force CPU
            )
            self.summarization_pipeline = pipeline(
                "summarization", 
                model=config.summarization_model, 
                device=-1  # Force CPU
            )
            self.keyword_extractor = pipeline(
                "token-classification", 
                model="dslim/bert-base-NER"
            )

            logger.info("ML pipelines initialized successfully")
        except Exception as e:
            logger.error(f"ML pipeline initialization failed: {e}")
            raise RuntimeError(f"Could not load ML models: {e}")

    async def analyze_segment(self, segment: TranscriptSegment) -> SegmentAnalysis:
        """
        Comprehensive async analysis of a transcript segment
        
        Args:
            segment (TranscriptSegment): Input transcript segment
        
        Returns:
            SegmentAnalysis: Detailed analysis result
        """
        try:
            # Sentiment Analysis
            sentiment_result = self._analyze_sentiment(segment.text)
            
            # Summarization
            summary = self._generate_summary(segment.text)
            
            # Keyword Extraction
            keywords = self._extract_keywords(segment.text)

            return SegmentAnalysis(
                original_text=segment.text,
                sentiment=sentiment_result,
                summary=summary,
                keywords=keywords,
                language='en'  # Default to English
            )
        except Exception as e:
            logger.error(f"Segment analysis error: {e}")
            raise

    def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Internal sentiment analysis method"""
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            return SentimentResult(
                label=result['label'], 
                confidence=result['score']
            )
        except Exception as e:
            logger.warning(f"Sentiment analysis fallback: {e}")
            return SentimentResult(label="NEUTRAL", confidence=0.5)

    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate concise text summary"""
        try:
            summary = self.summarization_pipeline(
                text[:1024], 
                max_length=max_length, 
                min_length=30, 
                do_sample=False
            )
            return summary[0]['summary_text'] if summary else text[:max_length]
        except Exception as e:
            logger.warning(f"Summary generation fallback: {e}")
            return text[:max_length]

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract significant keywords/entities"""
        try:
            ner_results = self.keyword_extractor(text[:512])
            keywords = [
                entity['word'] for entity in ner_results 
                if entity['score'] > 0.7
            ]
            return list(set(keywords))[:top_k]
        except Exception as e:
            logger.warning(f"Keyword extraction fallback: {e}")
            return []

class TranscriptAnalysisApp:
    """FastAPI Application for Transcript Analysis"""
    def __init__(self, analyzer: TranscriptAnalyzer):
        self.app = FastAPI(
            title="Intelligent Transcript Analyzer",
            description="Advanced ML-powered transcript processing",
            version="1.0.0"
        )
        self.analyzer = analyzer
        self._configure_middleware()
        self._setup_routes()

    def _configure_middleware(self):
        """Configure application middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Define API routes"""
        @self.app.post("/analyze")
        async def analyze_transcript(
            request: Request, 
            background_tasks: BackgroundTasks
        ):
            try:
                payload = await request.json()
                segments = [TranscriptSegment(**seg) for seg in payload.get('segments', [])]
                
                if not segments:
                    raise HTTPException(status_code=400, detail="No transcript segments provided")

                # Concurrent segment analysis
                analyses = await asyncio.gather(
                    *[self.analyzer.analyze_segment(segment) for segment in segments]
                )

                return {
                    "total_segments": len(analyses),
                    "results": [analysis.dict() for analysis in analyses]
                }

            except Exception as e:
                logger.error(f"Analysis request error: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

def main():
    """Application entry point"""
    try:
        config = AnalysisConfig()
        analyzer = TranscriptAnalyzer(config)
        app_wrapper = TranscriptAnalysisApp(analyzer)
        
        uvicorn.run(
            app_wrapper.app, 
            host="0.0.0.0", 
            port=8000
        )
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()