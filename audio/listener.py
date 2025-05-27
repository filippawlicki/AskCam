import numpy as np
import threading
import pyaudio
import whisper
import time


class AudioListener:
    def __init__(self, model_name="tiny.en", sample_rate=16000, channels=1, chunk=1024, buffer_size=32000):
        self.SAMPLE_RATE = sample_rate
        self.CHANNELS = channels
        self.CHUNK = chunk
        self.BUFFER_SIZE = buffer_size

        self.audio_buffer = np.zeros(self.BUFFER_SIZE, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.write_index = 0

        self.model = whisper.load_model(model_name)
        self.stream = None
        self.pa = None

    def start_audio_stream(self):
        """Start a PyAudio stream with the given callback."""
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paFloat32,
                                   channels=self.CHANNELS,
                                   rate=self.SAMPLE_RATE,
                                   input=True,
                                   frames_per_buffer=self.CHUNK,
                                   stream_callback=self.callback_pyaudio)
        self.stream.start_stream()

    def callback_pyaudio(self, in_data, frame_count, time_info, status):
        """PyAudio callback to update the circular buffer."""
        audio = np.frombuffer(in_data, dtype=np.float32)

        with self.buffer_lock:
            end_index = (self.write_index + len(audio)) % self.BUFFER_SIZE
            if end_index < self.write_index:
                self.audio_buffer[self.write_index:] = audio[:self.BUFFER_SIZE - self.write_index]
                self.audio_buffer[:end_index] = audio[self.BUFFER_SIZE - self.write_index:]
            else:
                self.audio_buffer[self.write_index:end_index] = audio
            self.write_index = end_index

        return in_data, pyaudio.paContinue

    def listen_hotword(self):
        """Continuously transcribe audio from the circular buffer and wait for hotword."""
        while True:
            with self.buffer_lock:
                start_index = (self.write_index - self.SAMPLE_RATE) % self.BUFFER_SIZE
                if start_index < 0:
                    snippet = np.concatenate((self.audio_buffer[start_index:], self.audio_buffer[:self.write_index]))
                else:
                    snippet = self.audio_buffer[start_index:self.write_index]

            text = self.model.transcribe(snippet, fp16=False, language="en")["text"].strip().lower()
            if any(word in text for word in ["hi", "hey"]):
                break
            time.sleep(0.1)

    def listen_question(self, duration_sec=5, silence_threshold=0.01, silence_duration_sec=1.5):
      """Listen for a question after detecting the hotword."""
      p = pyaudio.PyAudio()
      stream = p.open(format=pyaudio.paFloat32,
                      channels=self.CHANNELS,
                      rate=self.SAMPLE_RATE,
                      input=True,
                      frames_per_buffer=self.CHUNK)

      frames = []
      silence_frames = 0
      max_silence_frames = int(self.SAMPLE_RATE / self.CHUNK * silence_duration_sec)

      for _ in range(0, int(self.SAMPLE_RATE / self.CHUNK * duration_sec)):
        data = stream.read(self.CHUNK)
        audio = np.frombuffer(data, dtype=np.float32)
        frames.append(audio)

        # Calculate RMS and check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < silence_threshold:
          silence_frames += 1
          if silence_frames >= max_silence_frames:
            break
        else:
          silence_frames = 0

      stream.stop_stream()
      stream.close()
      p.terminate()

      full_audio = np.concatenate(frames)
      question_text = self.model.transcribe(full_audio, fp16=False, language="en")["text"].strip().lower()
      return question_text

    def listen_hotword_and_get_question(self):
        """Listen for hotword and then get the question."""
        self.listen_hotword()
        question = self.listen_question()
        return question


if __name__ == "__main__":
    listener = AudioListener()
    listener.start_audio_stream()
    print("Listening for hotword...")
    question = listener.listen_hotword_and_get_question()
    print(f"Detected question: {question}")
    listener.stream.stop_stream()
    listener.stream.close()
    listener.pa.terminate()
