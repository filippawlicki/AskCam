import torch
import numpy as np
import sounddevice as sd
from parler_tts import ParlerTTSForConditionalGeneration # pip install git+https://github.com/huggingface/parler-tts.git
from transformers import AutoTokenizer
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
import io
import os
from TTS.api import TTS


class myTTS:
    def __init__(self, device=None):
        """Initialize the TTS system."""
        self.engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

    def generate_audio(self, text):
        audio_arr = self.engine.tts(text)
        return audio_arr

    def play_audio(self, audio: np.ndarray):
        """Play the generated audio using sounddevice and wait for it to end speaking."""
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if outside [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio /= max_val

        try:
            sd.play(audio, samplerate=22050)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Audio playback failed: {e}")

    def speak(self, text: str):
        """Generate and play audio for the given text."""
        print(f"[TTS] Speaking: {text}")
        audio = self.generate_audio(text)
        self.play_audio(audio)

if __name__ == "__main__":
    tts = myTTS()
    tts.speak("Hello, this is a test of the text-to-speech system.")
