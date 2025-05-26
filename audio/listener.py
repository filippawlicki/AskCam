import pyaudio
import numpy as np
import whisper
import queue
import time
import threading

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS_QUESTION = 5

# Whisper model
model = whisper.load_model("tiny.en")

# Shared queue for audio
audio_queue = queue.Queue()


def start_audio_stream(callback):
  """Start a PyAudio stream with the given callback."""
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paFloat32,
                  channels=CHANNELS,
                  rate=SAMPLE_RATE,
                  input=True,
                  frames_per_buffer=CHUNK,
                  stream_callback=callback)
  return stream, p


def callback_pyaudio(in_data, frame_count, time_info, status):
  """PyAudio callback to continuously feed audio chunks."""
  audio = np.frombuffer(in_data, dtype=np.float32)
  audio_queue.put(audio)
  return (in_data, pyaudio.paContinue)


def transcribe_audio(audio_np):
  """Transcribe using Whisper and normalize if needed."""
  audio_float32 = audio_np.astype(np.float32)

  result = model.transcribe(audio_float32, fp16=False, language="en")
  return result["text"].strip().lower()


def listen_hotword(hotword="hey ai"):
  """Continuously listen for the hotword in real time."""
  print(f"Listening for hotword: '{hotword}'...")
  stream, pa = start_audio_stream(callback_pyaudio)
  stream.start_stream()

  buffer = np.zeros(0, dtype=np.float32)
  try:
    while True:
      while not audio_queue.empty():
        data = audio_queue.get()
        buffer = np.concatenate([buffer, data])

        if len(buffer) > SAMPLE_RATE:
          snippet = buffer[-SAMPLE_RATE:] # Get last 1 second of audio
          text = transcribe_audio(snippet)
          print(f"Transcribed: {text}")
          if hotword in text:
            #print("Hotword detected!")
            stream.stop_stream()
            stream.close()
            pa.terminate()
            return
          # Keep last 1.5s of buffer
          if len(buffer) > int(SAMPLE_RATE * 1.5):
            buffer = buffer[-int(SAMPLE_RATE * 1.5):]
      time.sleep(0.1)
  except KeyboardInterrupt:
    print("Hotword listening stopped.")
    stream.stop_stream()
    stream.close()
    pa.terminate()


def record_audio_pyaudio(duration_sec=5):
  """Record audio using PyAudio for a fixed duration."""
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paFloat32,
                  channels=CHANNELS,
                  rate=SAMPLE_RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

  print(f"Recording question for {duration_sec} seconds...")
  frames = []

  for _ in range(0, int(SAMPLE_RATE / CHUNK * duration_sec)):
    data = stream.read(CHUNK)
    audio = np.frombuffer(data, dtype=np.float32)
    frames.append(audio)

  stream.stop_stream()
  stream.close()
  p.terminate()

  audio_np = np.concatenate(frames)
  return audio_np


def listen_hotword_and_get_question():
  """Wait for hotword and then record and transcribe a question."""
  listen_hotword()
  audio_question = record_audio_pyaudio(RECORD_SECONDS_QUESTION)
  #print("Transcribing question...")
  question_text = transcribe_audio(audio_question)
  #print(f"Question: {question_text}")
  return question_text


if __name__ == "__main__":
  question = listen_hotword_and_get_question()
  #print(f"\nFinal question: {question}")