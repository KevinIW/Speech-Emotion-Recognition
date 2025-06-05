import os
import io
import uuid
import numpy as np
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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

# Model checkpoint
checkpoint = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model_path = "checkpoints/emotion_model.pth"  

# Define lifespan context manager for FastAPI (modern way to handle startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and feature extractor
    logger.info("API server starting up...")
    # No preloading of model - will be loaded on first request
    yield
    # Shutdown: Clean up resources
    logger.info("API server shutting down...")
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                logger.info("Loading feature extractor...")
                cls._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
                
                logger.info(f"Loading model from {model_path}...")
                # Check if model file exists
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found at {model_path}. Using base model.")
                    # First create model instance with the right architecture
                    cls._model = AutoModelForAudioClassification.from_pretrained(
                        checkpoint,
                        num_labels=6  # 6 emotions
                    )
                else:
                    # First create model instance with the right architecture
                    cls._model = AutoModelForAudioClassification.from_pretrained(
                        checkpoint,
                        num_labels=6  # 6 emotions
                    )
                    
                    try:
                        # Load state dictionary
                        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                        cls._model.load_state_dict(state_dict)
                        logger.info(f"Loaded saved model parameters from {model_path}")
                    except Exception as e:
                        logger.error(f"Error loading model state: {e}")
                        logger.info("Using base model without fine-tuning")
                
                # Set model to evaluation mode
                cls._model.eval()
                logger.info("Model and feature extractor loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
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

# Simple latency test endpoint
@app.get("/ping")
async def ping() -> Dict[str, Any]:
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
async def predict_emotion(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict emotion from an audio file
    
    - **file**: Audio file (WAV format recommended)
    
    Returns predicted emotion, confidence score, and probability distribution
    """
    # Get the model and feature extractor (loads on first request)
    try:
        model, feature_extractor = await ModelCache.get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
        
    # Validate file
    if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Only WAV, MP3, and OGG audio files are supported")
    
    try:
        # Read audio file - just read once
        content = await file.read()
        audio_bytes = io.BytesIO(content)
        
        # Process for inference first, then save to Supabase
        audio_bytes_copy = io.BytesIO(content)  # Make a copy for Supabase
        
        # Load audio using soundfile
        speech, sr = sf.read(audio_bytes)
        
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
        
        # Save to Supabase
        file_id = None
        storage_path = None
        
        if supabase:
            logger.info("Attempting to save file to Supabase...")
            audio_bytes_copy = io.BytesIO(content)
            file_id, storage_path = await save_to_supabase(
                audio_bytes_copy.getvalue(), 
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
        sys.exit(1)
