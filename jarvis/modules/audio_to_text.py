import wave

import openai
import pyaudio


class AudioToText:
    pyaudio_instance: pyaudio.PyAudio
    CHANNELS = 1
    CHUNK = 1024
    RATE = 44100
    FORMAT = pyaudio.paInt16
    RECORD_SECONDS = 5

    def __init__(self) -> None:
        self.pyaudio_instance = pyaudio.PyAudio()

    def use(self) -> str:
        write_buffer: wave.Wave_write = wave.open("temp_recording.wav", "wb")
        write_buffer.setnchannels(self.CHANNELS)
        write_buffer.setsampwidth(self.pyaudio_instance.get_sample_size(self.FORMAT))
        write_buffer.setframerate(self.RATE)

        stream = self.pyaudio_instance.open(
            self.RATE, self.CHANNELS, self.FORMAT, input=True
        )
        print("[recording-start]")
        for _ in range(0, self.RATE // self.CHUNK * self.RECORD_SECONDS):
            write_buffer.writeframes(stream.read(self.CHUNK))
        print("[recording-end]")

        stream.close()
        self.pyaudio_instance.terminate()  # Terminate the pyaudio instance

        write_buffer.close()
        read_buffer = open("temp_recording.wav", "rb")

        audio_transcribe = openai.audio.transcriptions.create(
            file=read_buffer, model="whisper-1"
        )
        transcribed_audio = audio_transcribe.text
        read_buffer.close()

        print("[transcription] - {}".format(transcribed_audio))
        return transcribed_audio
