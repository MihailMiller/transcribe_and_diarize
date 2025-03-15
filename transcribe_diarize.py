import argparse
import logging
import subprocess
import os
from typing import List

import pandas as pd
import torch
from dotenv import dotenv_values
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
HF_TOKEN = dotenv_values(".env").get("HF_TOKEN", "")

def create_transcribed_segment(start: int, end: int, text: str):
    return {"start": start, "end": end, "text": text}

def create_speaker_segment(start: int, end: int, speaker: str):
    return {"start": start, "end": end, "speaker": speaker}

def convert_to_wav(input_file: str, output_file: str):
    """Convert an audio file to WAV format using ffmpeg."""
    logging.info(f"Converting {input_file} to WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_file
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def transcribe_audio(wav_file: str, model: str):
    """Transcribe an audio file using Whisper with a specified model."""
    logging.info(f"Transcribing with Whisper using model {model}...")
    subprocess.run([
        "../whisper.cpp/build/bin/Release/main.exe", "-m", f"../whisper.cpp/models/{model}", "-ml", "50", "-l", "de", "-ocsv", "-f", wav_file
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return f"{os.path.splitext(wav_file)[0]}.wav.csv"

def diarize_audio(wav_file: str):
    """Perform speaker diarization using pyannote with GPU support if available."""
    logging.info("Performing speaker diarization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    pipeline.to(device)  # Move the model to GPU if available

    diarization = pipeline(wav_file)
    
    return [
        create_speaker_segment(int(segment.start * 1000), int(segment.end * 1000), label)
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]

def parse_transcript(csv_file: str) -> List[dict]:
    """Parse the Whisper-generated transcript CSV."""
    df = pd.read_csv(csv_file)
    return [create_transcribed_segment(int(row["start"]), int(row["end"]), row["text"]) for _, row in df.iterrows()]

def match_speakers(transcribed_segments: List[dict], speaker_segments: List[dict]):
    """Match transcribed segments to speaker segments based on overlap, grouping consecutive lines per speaker."""
    matched = []
    last_speaker = None
    speaker_text = []
    speaker_start_time = None

    for transcribed_segment in transcribed_segments:
        # Find the best matching speaker segment
        best_match = max(
            speaker_segments,
            key=lambda s: max(0, min(transcribed_segment["end"], s["end"]) - max(transcribed_segment["start"], s["start"]))
        )
        current_speaker = best_match["speaker"]
        start_time = transcribed_segment["start"]
        text = transcribed_segment["text"]

        if current_speaker == last_speaker:
            speaker_text.append(text)  # Append to ongoing text for the same speaker
        else:
            # If speaker changes, save the previous speaker's dialogue
            if last_speaker is not None:
                matched.append((last_speaker, speaker_start_time, "".join(speaker_text)))

            # Start a new speaker segment
            last_speaker = current_speaker
            speaker_start_time = start_time
            speaker_text = [text]

    # Append last recorded segment
    if last_speaker is not None and speaker_text:
        matched.append((last_speaker, speaker_start_time, "".join(speaker_text)))

    return matched

def format_time(ms):
    """Format milliseconds into HH:MM:SS format."""
    seconds = ms // 1000
    minutes = (seconds // 60) % 60
    hours = (seconds // 3600)
    return f"{hours:02}:{minutes:02}:{seconds % 60:02}"

def process_audio(file_to_process: str, model: str, output_dir: str):
    """Process an individual audio file: convert, transcribe, diarize, and match speakers."""
    base_filename = os.path.splitext(os.path.basename(file_to_process))[0]
    wav_path = os.path.join(output_dir, f"{base_filename}.wav")
    transcript_path = os.path.join(output_dir, f"{base_filename}.wav.csv")  # Expected transcript file

    if not file_to_process.endswith(".wav"):
        convert_to_wav(file_to_process, wav_path)

    # Check if transcription already exists
    if os.path.exists(transcript_path):
        logging.info(f"Transcription already exists for {file_to_process}, skipping transcription.")
    else:
        transcript_path = transcribe_audio(wav_path, model)

    transcribed_segments = parse_transcript(transcript_path)

    speaker_segments = diarize_audio(wav_path)
    matched_segments = match_speakers(transcribed_segments, speaker_segments)

    # Write to output
    output_file = os.path.join(output_dir, f"{base_filename}-output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for speaker, start_time, text in matched_segments:
            f.write(f"{speaker}:{text} #{format_time(start_time)}#\n\n")
    logging.info(f"Processing complete. Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process audio files for transcription and speaker diarization.")
    parser.add_argument("--input", type=str, help="Input directory containing audio files")
    parser.add_argument("--output", type=str, help="Output directory for processed files")
    parser.add_argument("--model", type=str, default="ggml-tiny.bin", help="Whisper model to use for transcription")
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and file.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):  # Supported formats
            process_audio(file_path, args.model, output_dir)

if __name__ == "__main__":
    main()