import os
import io
import numpy as np
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
)
import uvicorn


app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for predicting emotions from speech audio files",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Label mapping
e_labels = {"marah": 0, "jijik": 1, "takut": 2, "bahagia": 3, "netral": 4, "sedih": 5}
inv_map = {v: k for k, v in e_labels.items()}

# Model checkpoint
checkpoint = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model_path = "checkpoints/emotion_model.pth"  

model = None
feature_extractor = None

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict

@app.on_event("startup")
async def startup_event():
    """Load model and feature extractor on startup"""
    global model, feature_extractor
    
    try:
        print("Loading feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        
        print(f"Loading model from {model_path}...")
        # First create model instance with the right architecture
        model = AutoModelForAudioClassification.from_pretrained(
            checkpoint,
            num_labels=6  # 6 emotions
        )
        
        # Check if model file exists
        if os.path.exists(model_path):
            # Load state dictionary
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Loaded saved model parameters from {model_path}")
        else:
            print(f"Model file not found at {model_path}. Using base model.")
        
        # Set model to evaluation mode
        model.eval()
        print("Model and feature extractor loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Continue without model - we'll handle this in the endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Speech Emotion Recognition API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from an audio file
    
    - **file**: Audio file (WAV format recommended)
    
    Returns predicted emotion, confidence score, and probability distribution
    """
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
    
    # Validate file
    if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Only WAV, MP3, and OGG audio files are supported")
    
    try:
        # Read audio file
        content = await file.read()
        audio_bytes = io.BytesIO(content)
        
        # Load audio using soundfile
        speech, sr = sf.read(audio_bytes)
        
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
        
        return {
            "emotion": pred_emotion,
            "confidence": confidence,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
