import os
import io
import uuid
import numpy as np
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
)
import uvicorn
from supabase.client import create_client
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import sys
import traceback
import logging

# Import Sentry SDK
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configure better exception handling
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_exception

# Load environment variables - force override to ensure latest values
load_dotenv(override=True)

# Initialize Sentry SDK
def setup_sentry():
    env = os.environ.get("ENVIRONMENT", "development")
    sentry_dsn = os.environ.get("SENTRY_DSN")
    
    if sentry_dsn:
        logger.info(f"Initializing Sentry monitoring in {env} environment")
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=env,
            # Add data like request headers and IP for users
            send_default_pii=True,
            # Set traces_sample_rate to capture transactions for performance monitoring
            # Lower this in production for high-traffic apps
            traces_sample_rate=1.0 if env == "development" else 0.2,
            # Profile sessions for performance insights
            profile_session_sample_rate=1.0 if env == "development" else 0.2,
            # Automatically run the profiler on active transactions
            profile_lifecycle="trace",
            # Enable FastAPI and AsyncIO integrations
            integrations=[
                FastApiIntegration(),
                AsyncioIntegration(),
            ],
            # Set a custom release identifier
            release="speech-emotion-recognition@1.0.0",
        )
        logger.info("Sentry monitoring initialized successfully")
    else:
        logger.warning("Sentry DSN not provided in environment variables, monitoring disabled")

# Setup Sentry before application start
setup_sentry()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
bucket_name = os.environ.get("SUPABASE_BUCKET", "speech-emotion-recognition")

logger.info(f"Using Supabase URL: {supabase_url}")
logger.info(f"Using bucket: {bucket_name}")

supabase = None
if supabase_url and supabase_key:
    try:
        # Initialize client with the service role key
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Supabase client: {e}")
else:
    logger.error("Supabase credentials not found in environment variables")

# Label mapping
e_labels = {"marah": 0, "jijik": 1, "takut": 2, "bahagia": 3, "netral": 4, "sedih": 5}
inv_map = {v: k for k, v in e_labels.items()}

# Model checkpoint - update to the new model repository
checkpoint = os.environ.get("HF_MODEL_REPO", "Miracle12345/Speech-Emotion-Recognition")
model_path = os.environ.get("LOCAL_MODEL_PATH", "checkpoints/emotion_model.pth")

# Define lifespan context manager for FastAPI (modern way to handle startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Log application startup to Sentry
    logger.info("API server starting up...")
    with sentry_sdk.start_transaction(op="startup", name="Application Startup"):
        # No preloading of model - will be loaded on first request
        pass
    yield
    # Shutdown: Clean up resources
    logger.info("API server shutting down...")
    with sentry_sdk.start_transaction(op="shutdown", name="Application Shutdown"):
        # Release model resources
        global model
        model = None  # Free memory

# Create FastAPI app with lifespan
app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for predicting emotions from speech audio files",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting with slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

# Create limiter instance with default key function
limiter = Limiter(key_func=get_remote_address)

# Add limiter to app state
app.state.limiter = limiter

# Add rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests, please try again later."}
    )

# Enable CORS with more restrictive settings
# Read allowed origins from environment variable
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

# For development/initial deployment, temporarily allow broader CORS
# This helps during the transition before you know the exact frontend URL
allow_all = os.environ.get("ALLOW_ALL_ORIGINS", "false").lower() == "true"

if allow_all or os.environ.get("ENVIRONMENT") == "development":
    logger.warning("⚠️ CORS configured to allow all origins or in development mode.")
    origins = ["*"] if allow_all else allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
else:
    logger.info(f"CORS configured to allow specific origins: {allowed_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

# Add HTTPS redirect in production
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
if os.environ.get("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# Create a cached model singleton class
class ModelCache:
    _instance = None
    _model = None
    _feature_extractor = None
    _is_loading = False
    
    @classmethod
    async def get_model(cls):
        if cls._model is None and not cls._is_loading:
            cls._is_loading = True
            try:
                # Check if we should try loading from a local path first
                use_local = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"
                
                # First, try to load the feature extractor
                try:
                    logger.info("Loading feature extractor...")
                    # Use the fallback checkpoint if configured
                    fallback_checkpoint = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                    
                    try:
                        cls._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
                        logger.info(f"Feature extractor loaded from {checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to load feature extractor from {checkpoint}: {e}")
                        logger.info(f"Attempting to load feature extractor from fallback: {fallback_checkpoint}")
                        cls._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(fallback_checkpoint)
                        logger.info(f"Feature extractor loaded from fallback {fallback_checkpoint}")
                except Exception as e:
                    logger.error(f"Error loading feature extractor: {e}")
                    raise
                
                # Now, handle model loading with priority based on USE_LOCAL_MODEL
                if use_local:
                    logger.info("USE_LOCAL_MODEL is set to true, trying local model first")
                    
                    if os.path.exists(model_path):
                        try:
                            # Load model architecture (try both checkpoints)
                            try:
                                logger.info(f"Loading model architecture from {checkpoint}")
                                cls._model = AutoModelForAudioClassification.from_pretrained(
                                    checkpoint,
                                    num_labels=6  # 6 emotions
                                )
                            except Exception as e:
                                logger.warning(f"Failed to load model architecture from {checkpoint}: {e}")
                                logger.info(f"Attempting to load model architecture from fallback: {fallback_checkpoint}")
                                cls._model = AutoModelForAudioClassification.from_pretrained(
                                    fallback_checkpoint,
                                    num_labels=6  # 6 emotions
                                )
                            
                            # Load state dictionary from local file
                            logger.info(f"Loading weights from local file: {model_path}")
                            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                            cls._model.load_state_dict(state_dict)
                            logger.info(f"Successfully loaded model from local file {model_path}")
                        except Exception as e:
                            logger.error(f"Error loading local model: {e}")
                            logger.info("Falling back to Hugging Face model")
                            # Try loading from Hugging Face (both checkpoints)
                            try:
                                cls._model = AutoModelForAudioClassification.from_pretrained(
                                    checkpoint,
                                    num_labels=6  # 6 emotions
                                )
                            except Exception as hf_error:
                                logger.warning(f"Failed to load from {checkpoint}: {hf_error}")
                                logger.info(f"Attempting to load from fallback: {fallback_checkpoint}")
                                cls._model = AutoModelForAudioClassification.from_pretrained(
                                    fallback_checkpoint,
                                    num_labels=6  # 6 emotions
                                )
                    else:
                        logger.warning(f"Local model path {model_path} not found, loading from Hugging Face")
                        # Try loading from Hugging Face (both checkpoints)
                        try:
                            cls._model = AutoModelForAudioClassification.from_pretrained(
                                checkpoint,
                                num_labels=6  # 6 emotions
                            )
                        except Exception as hf_error:
                            logger.warning(f"Failed to load from {checkpoint}: {hf_error}")
                            logger.info(f"Attempting to load from fallback: {fallback_checkpoint}")
                            cls._model = AutoModelForAudioClassification.from_pretrained(
                                fallback_checkpoint,
                                num_labels=6  # 6 emotions
                            )
                else:
                    # Try loading directly from Hugging Face (both checkpoints)
                    logger.info("Attempting to load model from Hugging Face")
                    try:
                        cls._model = AutoModelForAudioClassification.from_pretrained(
                            checkpoint,
                            num_labels=6  # 6 emotions
                        )
                    except Exception as hf_error:
                        logger.warning(f"Failed to load from {checkpoint}: {hf_error}")
                        logger.info(f"Attempting to load from fallback: {fallback_checkpoint}")
                        cls._model = AutoModelForAudioClassification.from_pretrained(
                            fallback_checkpoint,
                            num_labels=6  # 6 emotions
                        )
                
                # Set model to evaluation mode
                cls._model.eval()
                logger.info("Model and feature extractor loaded successfully!")
            except Exception as e:
                logger.error(f"Fatal error loading model: {e}")
                cls._is_loading = False
                raise e
            cls._is_loading = False
        
        # Wait for model to load if it's currently loading
        while cls._is_loading:
            await asyncio.sleep(0.1)
            
        return cls._model, cls._feature_extractor

# Update the response model with modern type annotations
class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    file_id: Optional[str] = ""
    storage_path: Optional[str] = ""

# Health check endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Health check endpoint"""
    return {"status": "online", "message": "Speech Emotion Recognition API is running"}

# Simple latency test endpoint with rate limiting
@app.get("/ping")
@limiter.limit("10/minute")  # Allow 10 pings per minute
async def ping(request: Request) -> Dict[str, Any]:
    """Simple endpoint to test API latency"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

async def save_to_supabase(file_content: bytes, file_name: str, content_type: str) -> tuple:
    """Save file to Supabase Storage and record in database"""
    if not supabase:
        logger.error("Supabase client not initialized")
        return None, None
    
    try:
        # Generate a unique ID for the file
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file_name)[1]
        storage_path = f"{file_id}{file_extension}"
        
        # Upload file to Supabase Storage
        try:
            logger.info(f"Uploading file to Supabase bucket: {bucket_name}")
            supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": content_type}
            )
            logger.info("File uploaded successfully")
            
            # Get public URL
            file_url = supabase.storage.from_(bucket_name).get_public_url(storage_path)
            logger.info(f"File URL: {file_url}")
        except Exception as e:
            logger.error(f"Error uploading file to Supabase storage: {e}")
            return None, None
        
        # Insert record into database
        try:
            timestamp = datetime.now().isoformat()
            data = {
                "id": file_id,
                "file_name": file_name,
                "storage_path": storage_path,
                "content_type": content_type,
                "created_at": timestamp,
                "file_url": file_url
            }
            
            logger.info(f"Inserting record into audio_files table")
            response = supabase.table("audio_files").insert(data).execute()
            logger.info("Database record created successfully")
        except Exception as e:
            logger.error(f"Error inserting record into database: {e}")
            # If database insert fails but file upload succeeded, return file info anyway
            return file_id, storage_path
        
        logger.info(f"File saved to Supabase with ID: {file_id}")
        return file_id, storage_path
        
    except Exception as e:
        logger.error(f"Error saving to Supabase: {e}")
        return None, None

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")  # Allow 30 predictions per minute
async def predict_emotion(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict emotion from an audio file
    
    - **file**: Audio file (WAV format recommended)
    
    Returns predicted emotion, confidence score, and probability distribution
    """
    # Enhanced input validation
    # 1. Check content type
    valid_content_types = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/x-wav"]
    if file.content_type and file.content_type not in valid_content_types:
        raise HTTPException(status_code=400, 
                           detail=f"Invalid content type: {file.content_type}. Expected audio file.")
    
    # 2. Check file extension more precisely
    valid_extensions = ('.wav', '.mp3', '.ogg')
    if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
        raise HTTPException(status_code=400, 
                          detail=f"Invalid file extension. Supported formats: {', '.join(valid_extensions)}")
    
    # 3. Read file content ONLY ONCE
    content = await file.read()
    file_size = len(content)
    max_size = 10 * 1024 * 1024  # 10 MB
    
    if file_size > max_size:
        raise HTTPException(status_code=413, 
                          detail=f"File too large. Maximum size is {max_size // (1024 * 1024)}MB")
    
    # Create a Sentry transaction for monitoring this endpoint's performance
    with sentry_sdk.start_transaction(op="http.server", name="POST /predict"):
        # Add context information to Sentry
        sentry_sdk.set_context("file_info", {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file_size
        })
        
        # Get the model and feature extractor (loads on first request)
        try:
            with sentry_sdk.start_span(op="model.load", description="Load ML model"):
                model, feature_extractor = await ModelCache.get_model()
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
        try:
            # Make a fresh BytesIO object for processing
            audio_bytes = io.BytesIO(content)
            audio_bytes.seek(0)  # Important: reset position to beginning of file
            
            # Try different approaches to load the audio
            try:
                # First try using soundfile
                speech, sr = sf.read(audio_bytes)
            except Exception as sf_error:
                logger.warning(f"Failed to read with soundfile: {sf_error}. Trying librosa...")
                
                # If soundfile fails, try librosa (more robust but slower)
                # We need to save to a temporary file for librosa
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(content)
                
                try:
                    speech, sr = librosa.load(temp_path, sr=None)
                    logger.info(f"Successfully loaded audio with librosa, sr={sr}")
                    # Clean up temp file
                    os.unlink(temp_path)
                except Exception as librosa_error:
                    # Clean up temp file if it exists
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    logger.error(f"Failed to read with librosa: {librosa_error}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Could not process audio file. Make sure it's a valid audio format."
                    )
            
            # Limit audio length to 10 seconds max for faster processing
            max_samples = 10 * 16000  # 10 seconds at 16kHz
            if len(speech) > max_samples:
                speech = speech[:max_samples]
            
            # Resample if needed
            if sr != 16000:
                speech = librosa.resample(
                    speech, 
                    orig_sr=sr, 
                    target_sr=16000
                )
                sr = 16000
            
            # Convert stereo to mono if needed
            if len(speech.shape) > 1:
                speech = np.mean(speech, axis=1)
            
            # Process audio with feature extractor
            inputs = feature_extractor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Track model inference with Sentry
            with sentry_sdk.start_span(op="model.inference", description="Run inference"):
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get predictions
                logits = outputs.logits.cpu().numpy()[0]
                probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
                pred_idx = np.argmax(probabilities)
                pred_emotion = inv_map[pred_idx]
                confidence = float(probabilities[pred_idx])
        
            # Create probability dictionary
            prob_dict = {inv_map[i]: float(probabilities[i]) for i in range(len(probabilities))}
            
            # Save to Supabase - Use the original content
            file_id = None
            storage_path = None
            
            if supabase:
                logger.info("Attempting to save file to Supabase...")
                file_id, storage_path = await save_to_supabase(
                    content,  # Use the original content directly 
                    file.filename, 
                    file.content_type
                )
                
                logger.info(f"Supabase save result: file_id={file_id}, storage_path={storage_path}")
                
                # If we have a file_id, save the prediction too
                if file_id:
                    try:
                        prediction_data = {
                            "audio_file_id": file_id,
                            "emotion": pred_emotion,
                            "confidence": confidence,
                            "probabilities": prob_dict,
                            "created_at": datetime.now().isoformat()
                        }
                        logger.info("Saving prediction to database")
                        prediction_response = supabase.table("predictions").insert(prediction_data).execute()
                        logger.info("Prediction saved successfully")
                    except Exception as e:
                        logger.error(f"Error saving prediction to database: {e}")
            
            # Add prediction result to Sentry breadcrumbs
            sentry_sdk.add_breadcrumb(
                category="prediction",
                message=f"Predicted emotion: {pred_emotion} with confidence {confidence:.2f}",
                level="info"
            )
            
            # Return the response with file_id and storage_path if available
            return PredictionResponse(
                emotion=pred_emotion,
                confidence=confidence,
                probabilities=prob_dict,
                file_id=file_id if file_id is not None else "",
                storage_path=storage_path if storage_path is not None else ""
            )
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            sentry_sdk.capture_exception(e)
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    try:
        # Configure uvicorn with appropriate settings
        uvicorn.run(
            "app:app", 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            reload=True
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)
