import wave
from typing import Literal

import openai
import pyaudio
from pydantic import BaseModel, Field

from jarvis.tools._tool import AnthropicTool


class RecordVoiceInput(BaseModel):
    control_type: Literal["record_voice"] = "record_voice"
    record_intervals: int


class RecordVoiceOutput(BaseModel):
    control_type: Literal["record_voice"] = "record_voice"
    text: str


class OutputVoiceInput(BaseModel):
    control_type: Literal["output_voice"] = "output_voice"
    text: str


class OutputVoiceOutput(BaseModel):
    control_type: Literal["output_voice"] = "output_voice"


class AudioTranscieverControls(BaseModel):
    input: RecordVoiceInput | OutputVoiceInput = Field(discriminator="control_type")


class AudioTranscieverOutput(BaseModel):
    output: RecordVoiceOutput | OutputVoiceOutput = Field(discriminator="control_type")


class AudioTransciever(AnthropicTool[AudioTranscieverControls, AudioTranscieverOutput]):
    pyaudio_instance: pyaudio.PyAudio
    CHANNELS = 1
    CHUNK = 1024
    RATE = 44100
    FORMAT = pyaudio.paInt16

    def __init__(self) -> None:
        self.pyaudio_instance = pyaudio.PyAudio()

    @classmethod
    def get_name(cls) -> str:
        return "AudioToText"

    @classmethod
    def get_description(cls) -> str:
        return """This tool can be used to to record voice audio input which 
        is then converted to text by leveraging openai's whisper model. You can
        utilize this tool to collect a users response to a query you asked the
        user to better help them with their request.
        """

    def use(self, control_request: AudioTranscieverControls) -> AudioTranscieverOutput:
        if isinstance(control_request.input, RecordVoiceInput):
            return self._record_voice(control_request.input.record_intervals)
        elif isinstance(control_request.input, OutputVoiceInput):
            return self._output_voice(control_request.input.text)

    def _record_voice(self, record_intervals: int) -> AudioTranscieverOutput:
        write_buffer: wave.Wave_write = wave.open("temp_recording.wav", "wb")
        write_buffer.setnchannels(self.CHANNELS)
        write_buffer.setsampwidth(self.pyaudio_instance.get_sample_size(self.FORMAT))
        write_buffer.setframerate(self.RATE)

        stream = self.pyaudio_instance.open(
            self.RATE, self.CHANNELS, self.FORMAT, input=True
        )
        print("[recording-start]")
        for _ in range(0, self.RATE // self.CHUNK * record_intervals):
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
        return AudioTranscieverOutput(
            output=RecordVoiceOutput(
                control_type="record_voice", text=transcribed_audio
            )
        )

    def _output_voice(self, text: str) -> AudioTranscieverOutput:
        raise NotImplementedError()
