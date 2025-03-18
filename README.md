# Transcription and Speaker Diarization

## Overview
This script processes audio files by converting them to WAV format, transcribing them, and automatically assigning the speaker sections using **Whisper** (an open-source speech-to-text tool) and **Pyannote.audio** (a Python library for speaker diarization). It is particularly suitable for analyzing conversations or interviews.

## Features
- **Audio conversion**: Converting MP3, M4A, FLAC, and other formats into WAV.
- **Transcription**: Using Whisper for speech recognition and text extraction.
- **Speaker diarisation**: Identifying and marking speakers using Pyannote speaker diarisation.
- **Merging text and speakers**: Assigning the transcribed segments to the recognized speakers.
- **Storing the result**: Output of the transcription including speaker information in a text file.

## Prerequisites

### Software:
- Python 3.x
- `ffmpeg` (for audio conversion)
- `whisper.cpp` (compiled Whisper implementation)

### Python Dependencies:
Install the required Python packages with:

```bash
pip install pandas torch python-dotenv pyannote.audio
```

Additionally, **Pyannote** requires a Hugging Face token for authentication. This should be stored in an `.env` file as `HF_TOKEN`.

## Usage

### 1. Preparation
Store your Hugging Face token in an `.env` file:

```bash
HF_TOKEN=your_huggingface_token
```

### 2. Running the Script
Run the script with the desired parameters:

```bash
python script.py --input /path/to/input --output /path/to/output --model ggml-large-v3-turbo.bin
```

#### Arguments:
- `--input` (path to the input folder containing the audio files)
- `--output` (path to the output folder for the processed files)
- `--model` (name of the Whisper model, default: `ggml-tiny.bin`)

### 3. Processing
The script performs the following steps:
1. Converts the file to WAV format if necessary.
2. Performs transcription with Whisper.
3. Performs speaker diarisation with Pyannote.
4. Links the transcribed segments to the speaker labels.
5. Saves the result in a `.txt` file in the output folder.

### 4. Example Output
```
SPEAKER_00: Hello, how are you? #00:00:01#
SPEAKER_01: I'm fine, thank you! #00:00:03#
```

### Assumptions

The script assumes the following setup for running `whisper.cpp`:

- The compiled executable (Windows) is located at:
  ```
  ../whisper.cpp/build/bin/Release/main.exe
  ```
- The model files should be stored in:
  ```
  ../whisper.cpp/models/{model}
  ```
- The script will execute `whisper.cpp` using Python's `subprocess` module.

#### Adjusting for Your Environment
If your environment differs:
- Update the path to `main.exe` accordingly.
- Ensure that the model files are placed in the correct directory.
- Modify the script to reflect your directory structure.

## Troubleshooting
If you encounter any issues:
- Ensure that `ffmpeg` is installed correctly.
- Check that `whisper.cpp` compiles and the model files are present.
- Verify that the `.env` file contains a valid Hugging Face token.

## License
This script is released under the **MIT License**.

## Funding
Parts of the work were funded by grants of the German Ministry of Education and Research in the context of the joint research projects "MANGAN" (01IS22011C) under the supervision of the PT-DLR.
