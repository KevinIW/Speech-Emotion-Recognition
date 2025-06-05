-- =============================================
-- Speech Emotion Recognition - Secure Supabase Setup
-- =============================================
-- This script sets up the database with:
-- 1. UUID extension
-- 2. Tables for audio files and predictions
-- 3. Appropriate RLS (Row Level Security) policies
-- 4. Private bucket configuration with service role access
-- =============================================

-- Enable UUID extension for generating unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- Create Tables
-- =============================================

-- Create audio_files table to store metadata about uploaded audio files
CREATE TABLE IF NOT EXISTS audio_files (
    id UUID PRIMARY KEY,               -- Unique identifier for each audio file
    file_name TEXT NOT NULL,           -- Original filename
    storage_path TEXT NOT NULL,        -- Path in Supabase Storage
    content_type TEXT NOT NULL,        -- MIME type (e.g., audio/wav)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),  -- When the file was uploaded
    file_url TEXT                      -- Public URL to access the file
);

-- Create predictions table to store emotion recognition results
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),  -- Unique identifier for each prediction
    audio_file_id UUID REFERENCES audio_files(id),   -- Reference to the audio file
    emotion TEXT NOT NULL,                           -- Predicted emotion
    confidence REAL NOT NULL,                        -- Confidence score
    probabilities JSONB NOT NULL,                    -- Full probability distribution
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() -- When the prediction was made
);

-- =============================================
-- Setup Row Level Security (RLS)
-- =============================================

-- Enable Row Level Security on tables
ALTER TABLE audio_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Drop any existing policies to ensure clean setup
DROP POLICY IF EXISTS "Allow service role full access to audio_files" ON audio_files;
DROP POLICY IF EXISTS "Allow service role full access to predictions" ON predictions;
DROP POLICY IF EXISTS "Allow authenticated users to select audio_files" ON audio_files;
DROP POLICY IF EXISTS "Allow authenticated users to insert audio_files" ON audio_files;
DROP POLICY IF EXISTS "Allow authenticated users to select predictions" ON predictions;
DROP POLICY IF EXISTS "Allow authenticated users to insert predictions" ON predictions;

-- =============================================
-- Create RLS Policies for Service Role
-- =============================================
-- These policies allow the service role (used by the backend) to have full access

-- Allow service role to do anything with audio_files
CREATE POLICY "Allow service role full access to audio_files"
ON audio_files FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Allow service role to do anything with predictions
CREATE POLICY "Allow service role full access to predictions"
ON predictions FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- =============================================
-- Storage Bucket Policies (Private Bucket)
-- =============================================
-- Drop any existing bucket policies
DROP POLICY IF EXISTS "Allow service role access to speech-emotion-recognition bucket" ON storage.objects;
DROP POLICY IF EXISTS "Allow public access to speech-emotion-recognition bucket" ON storage.objects;
DROP POLICY IF EXISTS "Allow authenticated access to speech-emotion-recognition bucket" ON storage.objects;

-- Create policy for service role to access the storage bucket
CREATE POLICY "Allow service role access to speech-emotion-recognition bucket"
ON storage.objects FOR ALL
TO service_role
USING (bucket_id = 'speech-emotion-recognition')
WITH CHECK (bucket_id = 'speech-emotion-recognition');

-- =============================================
-- Verify Setup
-- =============================================
-- List all tables
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- List all policies
SELECT 
    schemaname, 
    tablename, 
    policyname, 
    permissive, 
    roles, 
    cmd, 
    qual
FROM 
    pg_policies 
WHERE 
    schemaname = 'public' OR schemaname = 'storage';

-- =============================================
-- Instructions for Use
-- =============================================
/*
After running this script:

1. Create a bucket named 'speech-emotion-recognition' in the Storage section
2. Set the bucket to "Private" in the bucket settings
3. Use the service_role key in your backend application
4. Make sure your .env file contains:
   SUPABASE_URL=your_project_url
   SUPABASE_KEY=your_service_role_key
   SUPABASE_BUCKET=speech-emotion-recognition

This secure setup:
- Uses a private storage bucket for security
- Enforces Row Level Security on all tables
- Only allows access via the service role key
- Prevents unauthorized access to your data

You MUST use the service role key with this configuration, as the anon key
will not have sufficient permissions.
*/
