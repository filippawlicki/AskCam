# AskCam - Ask Your Camera, Answered by LLM

**AskCam** is an intelligent voice-enabled local assistant that lets you ask questions 
about what your webcam sees. It listens for a hotword (like “hi” or “hey”), 
captures a frame from your camera, runs it through a vision-language model to 
understand the scene, and responds to your question with spoken audio.

## Features

- **Voice Activation** - Just say "hi" or "hey" to trigger a question.
- **Live Camera Capture** - Captures real-time frames from your webcam.
- **Vision + Language Integration** - Uses LLaVA model to answer questions using not just text but also visual context.
- **Text-to-Speech** - Reads the answer back to you using a high-quality TTS engine.
- **Multithreaded Architecture** - Ensures responsive operation with background threads for camera, audio, hotword detection, and TTS.

## Demo

https://github.com/user-attachments/assets/042a6991-95bf-448b-9c81-8755c1b80e03

## Tech Stack

- **Python**
- **Gradio** - GUI
- **OpenCV** - Real-time camera handling
- **Whisper** - Speech-to-Text for transcribing questions
- **LLaVA** - Vision-Language model for question answering
- **TTS (kokoro / pydub)** - Audio output
- **pyaudio** - Hotword and question recording
- **Threading** - For concurrent audio/video/TTS

## Installation

```bash
git clone https://github.com/filippawlicki/askcam.git
cd askcam
pip install -r requirements.txt
```

## Usage

1. **Start the Application**: Run `python app.py` to launch the AskCam interface.
2. **Open the Web Interface**: Navigate to `http://localhost:7860` in your web browser.
3. **Ask a Question**: Say "hi" or "hey" to activate the assistant, then ask your question.
4. **Listen for the Answer**: The assistant will process your question and respond with spoken audio.

Enjoy!

