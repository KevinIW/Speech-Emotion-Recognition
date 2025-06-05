"""
Test script to verify that local model can be loaded correctly.
This helps diagnose issues with model loading when Hugging Face access fails.
"""
import os
import sys
import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Model path
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "emotion_model.pth")
# Fallback for base architecture
fallback_checkpoint = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

def check_local_model():
    """Test that the local model can be loaded without errors"""
    print(f"Looking for model file at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    try:
        print(f"Loading model architecture from {fallback_checkpoint}")
        model = AutoModelForAudioClassification.from_pretrained(
            fallback_checkpoint,
            num_labels=6  # 6 emotions for Indonesian
        )
        
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        print("Model loaded successfully!")
        
        # Try loading feature extractor
        print(f"Loading feature extractor from {fallback_checkpoint}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(fallback_checkpoint)
        print("Feature extractor loaded successfully!")
        
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("======== LOCAL MODEL TEST ========")
    success = check_local_model()
    
    if success:
        print("\n✅ Local model test PASSED. The model can be loaded correctly.")
        print("\nTo use this model in your app:")
        print("1. Make sure your .env file has USE_LOCAL_MODEL=true")
        print("2. Ensure the checkpoint path in app.py points to the correct location")
    else:
        print("\n❌ Local model test FAILED.")
        print("\nPossible solutions:")
        print("1. Check that the model file exists at the expected location")
        print("2. Verify that the model architecture matches the weights file")
        print("3. Try downloading the model architecture from HuggingFace manually")
        print("   using: model = AutoModelForAudioClassification.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')")
        print("4. Make sure you have enough RAM to load the model")
    
    print("\n======== TEST COMPLETE ========")
