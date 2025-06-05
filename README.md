# Speech Emotion Recognition System

This project implements a Speech Emotion Recognition (SER) system that can detect emotions in Indonesian speech. It consists of a machine learning model trained on audio data, a FastAPI backend for serving predictions, and a Next.js frontend for user interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Supabase Integration](#supabase-integration)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
  - [Deploying Backend to Render](#deploying-backend-to-render)
  - [Deploying Frontend to Vercel](#deploying-frontend-to-vercel)
- [Troubleshooting](#troubleshooting)

## Project Overview

This system can identify six different emotions in Indonesian speech:
- marah (angry)
- jijik (disgust)
- takut (fear)
- bahagia (happy)
- netral (neutral)
- sedih (sad)

The model is based on Wav2Vec2, fine-tuned on a dataset of Indonesian speech samples.

## Project Structure

```
speech-emotion-recognition-dl-genap-2024-2025/
├── app.py                     # FastAPI backend application
├── test_api.py                # Script to test the API
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── ser-ehcalabres.ipynb       # Jupyter notebook for model training
├── checkpoints/               # Directory for storing model checkpoints
│   └── emotion_model.pth      # Trained model weights
├── supabase/                  # Supabase configuration files
│   └── tables.sql             # SQL script to create Supabase tables
├── frontend/                  # Next.js frontend application
└── POSTMAN_TUTORIAL.md        # Guide for testing API with Postman
```

## Setup and Installation

### Prerequisites

- Python 3.8+ 
- Node.js 14+
- npm or yarn
- Git
- Supabase account

### Clone the Repository

```bash
git clone https://github.com/yourusername/speech-emotion-recognition-dl-genap-2024-2025.git
cd speech-emotion-recognition-dl-genap-2024-2025
```

### Python Environment Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training the Model

### Option 1: Run the Jupyter Notebook

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook ser-ehcalabres.ipynb
   ```

2. Follow the notebook cells to:
   - Load and analyze the dataset
   - Train the model
   - Evaluate the model
   - Export the model weights

3. Make sure the trained model is saved to `checkpoints/emotion_model.pth`

### Option 2: Use Pre-trained Model

1. Create a `checkpoints` directory if it doesn't exist:
   ```bash
   mkdir -p checkpoints
   ```

2. Download the pre-trained model and place it in the checkpoints directory:
   ```bash
   # If you have a download link
   wget -O checkpoints/emotion_model.pth https://your-model-download-link.com/emotion_model.pth
   ```

## Supabase Integration

This project uses Supabase for storing audio files and prediction results.

### Setting Up Supabase

1. **Create a Supabase account**
   - Sign up at [supabase.com](https://supabase.com/)
   - Create a new project

2. **Create a storage bucket**
   - Go to Storage in the Supabase dashboard
   - Create a new bucket called `audio_files`
   - Set the bucket's privacy to private

3. **Create database tables**
   - Go to the SQL Editor in the Supabase dashboard
   - Run the SQL script from `supabase/tables.sql`
   - Or manually create the `audio_files` and `predictions` tables

4. **Get your API credentials**
   - Go to Project Settings → API
   - Copy the URL and the anon/public key

5. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add the following variables:
     ```
     SUPABASE_URL=your_supabase_url
     SUPABASE_KEY=your_supabase_anon_key
     SUPABASE_BUCKET=audio_files
     ```

### Testing Supabase Integration

To verify that Supabase is properly configured:

1. Run the FastAPI server:
   ```bash
   python app.py
   ```

2. Use the `/predict` endpoint to upload an audio file
   - The file should be saved to Supabase storage
   - File metadata should be stored in the `audio_files` table
   - Prediction results should be stored in the `predictions` table

3. Check the Supabase dashboard to confirm the data was saved correctly

## Running the Backend

1. Make sure the model file exists at `checkpoints/emotion_model.pth`

2. Start the FastAPI server:
   ```bash
   python app.py
   ```

3. The API will be available at http://localhost:8000

4. You can check the API documentation at http://localhost:8000/docs

5. Test the API with the provided test script:
   ```bash
   python test_api.py http://localhost:8000 path/to/audio/file.wav
   ```

## Running the Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. The frontend will be available at http://localhost:3000

5. Make sure the backend API is running at http://localhost:8000 (or update the API URL in `frontend/app/page.tsx`)

## API Documentation

The API provides the following endpoints:

- `GET /`: Health check endpoint
- `POST /predict`: Predict emotion from an audio file
  - Input: Form data with a file field named "file" containing the audio file (WAV, MP3, OGG)
  - Output: JSON response with predicted emotion, confidence, probabilities, and file storage information

For detailed API documentation, visit http://localhost:8000/docs when the server is running.

## Deployment

### Deploying Backend to Render

1. **Create a Render account**
   - Sign up at [render.com](https://render.com/)

2. **Prepare your GitHub repository**
   - Make sure your code is pushed to a GitHub repository
   - Ensure the repository contains:
     - `app.py`
     - `requirements.txt`
     - Model file in `checkpoints/emotion_model.pth`

3. **Create a new Web Service**
   - In the Render dashboard, click "New" and select "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Name**: Choose a name for your service
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Choose Free (for testing) or other plan as needed

4. **Add Environment Variables**
   - Add your Supabase environment variables:
     - `SUPABASE_URL`
     - `SUPABASE_KEY`
     - `SUPABASE_BUCKET`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the build and deployment to complete

6. **Verify Deployment**
   - Once deployed, you'll get a URL like `https://your-app-name.onrender.com`
   - Test the API with:
     ```bash
     python test_api.py https://your-app-name.onrender.com path/to/audio/file.wav
     ```

7. **Handle Model Storage**
   - For production, consider storing your model in cloud storage (S3, Google Cloud Storage)
   - Update your code to download the model from cloud storage during startup

### Deploying Frontend to Vercel

1. **Create a Vercel account**
   - Sign up at [vercel.com](https://vercel.com/)

2. **Prepare your frontend**
   - Update the API URL in `frontend/app/page.tsx`:
     ```typescript
     const API_URL = 'https://your-backend-on-render.onrender.com';
     ```
   - Push the changes to your GitHub repository

3. **Import your project**
   - In the Vercel dashboard, click "Add New" → "Project"
   - Import your GitHub repository
   - Configure the project:
     - **Framework Preset**: Next.js
     - **Root Directory**: `frontend` (important!)

4. **Environment Variables (if needed)**
   - Add any environment variables your frontend needs

5. **Deploy**
   - Click "Deploy"
   - Wait for the build and deployment to complete

6. **Verify Deployment**
   - Once deployed, you'll get a URL like `https://your-frontend.vercel.app`
   - Open the URL in your browser and test the application

7. **Custom Domain (optional)**
   - In your project settings, you can configure a custom domain

## Troubleshooting

### Backend Issues

- **Model loading fails**: Ensure the model file exists at `checkpoints/emotion_model.pth`
- **CORS errors**: Make sure the frontend URL is added to the CORS allowed origins in `app.py`
- **Memory issues on Render**: Use a paid plan with more memory if you encounter OOM errors
- **Supabase connection issues**: Verify your environment variables are correct

### Frontend Issues

- **API connection fails**: Check that the API_URL in `frontend/app/page.tsx` is correct
- **File upload issues**: Make sure you're using the correct field name ("file") in the upload form

### Supabase Issues

- **Storage errors**: Verify the bucket exists and has the correct permissions
- **Database errors**: Check that the tables are created correctly
- **Authentication errors**: Verify your API key has the necessary permissions

### Testing API with Postman

See [POSTMAN_TUTORIAL.md](POSTMAN_TUTORIAL.md) for detailed instructions on testing the API with Postman.

## Using Pre-trained Models from Hugging Face

The application can load models directly from Hugging Face, which is especially useful for deployment environments with memory constraints like Render.

### Using Public Models

To use a public pre-trained model from Hugging Face:

1. Find a suitable emotion recognition model on [Hugging Face](https://huggingface.co/models)
   - Recommended model: [Miracle12345/Speech-Emotion-Recognition](https://huggingface.co/Miracle12345/Speech-Emotion-Recognition)

2. Set the `HF_MODEL_REPO` environment variable to the model path:
   ```
   HF_MODEL_REPO=Miracle12345/Speech-Emotion-Recognition
   ```

3. Ensure `USE_LOCAL_MODEL` is set to `false` to always load from Hugging Face:
   ```
   USE_LOCAL_MODEL=false
   ```

4. The application will automatically download and use the model from Hugging Face.

### Using Private Models

For private models, you'll need to:

1. Generate a read token at https://huggingface.co/settings/tokens
2. Add your token to the environment variables:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

### Deploying with Hugging Face Models

When deploying to platforms like Render:

1. Set the environment variables in your deployment platform:
   - `HF_MODEL_REPO`: Your chosen model path
   - `USE_LOCAL_MODEL`: Set to "false"
   - `HF_TOKEN`: Only if using a private model

2. The application will load the model directly from Hugging Face, avoiding the need to store large model files in your deployment.