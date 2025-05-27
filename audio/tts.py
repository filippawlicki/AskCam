import numpy as np
from kokoro import KPipeline
from pydub import AudioSegment
from pydub.playback import play



class myTTS:
    def __init__(self):
        """Initialize the TTS system."""
        self.engine = KPipeline(lang_code='a')

    def generate_audio(self, text):
        audio_res = self.engine(text, voice='af_heart')
        return audio_res

    def play_audio(self, generator):
        """Play the generated audio using sounddevice and wait for it to end speaking."""
        for i, (gs, ps, audio) in enumerate(generator):
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            audio = audio.astype(np.float32)

            audio_int16 = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=24000,
                sample_width=2,  # 2 bytes = 16-bit PCM
                channels=1
            )

            normalized = audio_segment.apply_gain(-audio_segment.max_dBFS)

            play(normalized)

    def speak(self, text: str):
        """Generate and play audio for the given text."""
        generator = self.generate_audio(text)
        self.play_audio(generator)

if __name__ == "__main__":
    tts = myTTS()
    tts.speak("Hello, this is a test of the text-to-speech system.")
