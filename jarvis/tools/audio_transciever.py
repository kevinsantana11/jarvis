import json
import logging
import os
import tempfile
import typing
import uuid
import wave
from typing import Literal

import openai
import pyaudio
from pydantic import BaseModel, Field
from pydub import AudioSegment

from jarvis.tools._tool import AnthropicTool

_logger = logging.getLogger(__name__)


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
    status: Literal["success"] | Literal["failure"]
    reason: str | None = None


class AudioTranscieverControls(BaseModel):
    input: RecordVoiceInput | OutputVoiceInput = Field(
        ..., discriminator="control_type"
    )


class AudioTranscieverOutput(BaseModel):
    output: RecordVoiceOutput | OutputVoiceOutput = Field(
        ..., discriminator="control_type"
    )


class AudioTransciever(AnthropicTool[AudioTranscieverControls, AudioTranscieverOutput]):
    controls = AudioTranscieverControls
    openai_client: openai.Client
    pyaudio_instance: pyaudio.PyAudio
    CHANNELS = 1
    CHUNK = 1024
    RATE = 44100
    FORMAT = pyaudio.paInt16

    def __init__(self, openai_client: openai.Client) -> None:
        self.pyaudio_instance = pyaudio.PyAudio()
        self.openai_client = openai_client

    @classmethod
    def get_name(cls) -> str:
        return "AudioTransciever"

    @classmethod
    def get_description(cls) -> str:
        return """This tool can be used to to convert a users voice and words
         from audio into text by leveragings openai's whisper tts model. This 
         tool will enable the user to communicate with you, using their voice.
        """

    @classmethod
    @typing.override
    def transform(cls, input: dict):
        input_obj_str = input.get("input", "{}")
        input_obj: dict[str, typing.Any] = json.loads(input_obj_str)
        return AudioTranscieverControls.model_validate({"input": input_obj})

    @typing.override
    def _use(self, control_request: AudioTranscieverControls) -> AudioTranscieverOutput:
        if isinstance(control_request.input, RecordVoiceInput):
            return self._record_voice(control_request.input.record_intervals)
        elif isinstance(control_request.input, OutputVoiceInput):
            return self._output_voice(control_request.input.text)

    def _get_tmp_fp(self, filename: str, ext: str):
        temp_dir_base = tempfile.gettempdir()
        jarvis_tmp_dir = os.path.join(temp_dir_base, "jarvis", "tmp")
        os.makedirs(jarvis_tmp_dir, exist_ok=True)
        filepath = os.path.join(jarvis_tmp_dir, f"$output_voice_{uuid.uuid4().hex}.{ext}")
        return filepath

    def _record_voice(self, record_intervals: int) -> AudioTranscieverOutput:
        record_fp = self._get_tmp_fp("temp_recording", "wav")
        write_buffer: wave.Wave_write = wave.open(record_fp, "wb")
        write_buffer.setnchannels(self.CHANNELS)
        write_buffer.setsampwidth(self.pyaudio_instance.get_sample_size(self.FORMAT))
        write_buffer.setframerate(self.RATE)

        stream = self.pyaudio_instance.open(
            self.RATE, self.CHANNELS, self.FORMAT, input=True
        )
        _logger.info("[recording-start]")
        for _ in range(0, self.RATE // self.CHUNK * record_intervals):
            write_buffer.writeframes(stream.read(self.CHUNK))
        _logger.info("[recording-end]")

        stream.close()

        write_buffer.close()
        read_buffer = open(record_fp, "rb")

        audio_transcribe = self.openai_client.audio.transcriptions.create(
            file=read_buffer, model="whisper-1"
        )
        transcribed_audio = audio_transcribe.text
        read_buffer.close()

        _logger.info("[transcription] - {}".format(transcribed_audio))
        return AudioTranscieverOutput(
            output=RecordVoiceOutput(
                control_type="record_voice", text=transcribed_audio
            )
        )

    def _output_voice(self, text: str) -> AudioTranscieverOutput:
        voice_output_fp = self._get_tmp_fp("voice_output", "mp3")
        response = self.openai_client.audio.speech.create(
            input=text, model="tts-1", voice="alloy", response_format="mp3"
        )
        with open(voice_output_fp, mode="wb") as audio_file:
            audio_file.write(response.content)

        audio = AudioSegment.from_mp3(voice_output_fp)
        stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(audio.sample_width),
            channels=audio.channels,
            rate=audio.frame_rate,
            output=True,
        )
        raw_data = audio.raw_data
        stream.write(raw_data)

        stream.stop_stream()
        stream.close()

        return AudioTranscieverOutput(
            output=OutputVoiceOutput(
                status="success",
            )
        )
