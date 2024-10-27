import contextlib
import logging
import os
import tempfile
import typing
import uuid
import wave
from typing import Literal

import openai
import pyaudio
import webrtcvad
from pydantic import BaseModel, Field
from pydub import AudioSegment

from jarvis import utils
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


class RecordAdaptiveVoiceInput(BaseModel):
    control_type: Literal["record_voice_adaptive"] = "record_voice_adaptive"
    frame_duration: int = 5 # in ms (milliseconds)


class RecordAdaptiveVoiceOutput(BaseModel):
    control_type: Literal["record_voice_adaptive"] = "record_voice_adaptive"
    text: str


class AudioTranscieverControls(BaseModel):
    input: RecordVoiceInput | OutputVoiceInput | RecordAdaptiveVoiceInput = Field(
        ..., discriminator="control_type"
    )


class AudioTranscieverOutput(BaseModel):
    output: RecordVoiceOutput | OutputVoiceOutput | RecordAdaptiveVoiceOutput | None = Field(
        ..., discriminator="control_type"
    )
    error: bool = False
    reason: str | None = None


class AudioTransciever(AnthropicTool[AudioTranscieverControls, AudioTranscieverOutput]):
    controls = AudioTranscieverControls
    vad: webrtcvad.Vad
    openai_client: openai.Client
    pyaudio_instance: pyaudio.PyAudio
    CHANNELS = 1
    RATE = 32_000
    FORMAT = pyaudio.paInt16

    def __init__(self, openai_client: openai.Client, vad: webrtcvad.Vad) -> None:
        self.pyaudio_instance = pyaudio.PyAudio()
        self.openai_client = openai_client
        self.vad = vad

    @classmethod
    def get_name(cls) -> str:
        return "AudioTransciever"

    @classmethod
    def get_description(cls) -> str:
        return """This tool can be used to to convert a users voice and words
         from audio into text by leveragings openai's whisper tts model. This 
         tool will enable the user to communicate with you, using their voice.
         Always try to use the adaptive tools first.
        """

    @typing.override
    def _use(self, control_request: AudioTranscieverControls) -> AudioTranscieverOutput:
        try:
            if isinstance(control_request.input, RecordVoiceInput):
                return self._record_voice_manual(control_request.input.record_intervals)
            elif isinstance(control_request.input, OutputVoiceInput):
                return self._output_voice(control_request.input.text)
            elif isinstance(control_request.input, RecordAdaptiveVoiceInput):
                return self._record_voice_adaptively(control_request.input.frame_duration)
        except Exception as e:
            return AudioTranscieverOutput(output=None, error=True, reason=str(e))

    def _get_tmp_fp(self, filename: str, ext: str):
        temp_dir_base = tempfile.gettempdir()
        jarvis_tmp_dir = os.path.join(temp_dir_base, "jarvis", "tmp")
        os.makedirs(jarvis_tmp_dir, exist_ok=True)
        filepath = os.path.join(jarvis_tmp_dir, f"{filename}_{uuid.uuid4().hex}.{ext}")
        return filepath

    def _record_voice_adaptively(self, base_recording_time: int):
        record = True
        voice_segments = list[bytes]()
        frame_length = 20
        chunksize = self.RATE * (frame_length / 1000)
        frames_per_buffer = int(self.RATE // chunksize * base_recording_time)

        try:
            stream = self.pyaudio_instance.open(
                self.RATE, self.CHANNELS, self.FORMAT, input=True, frames_per_buffer=frames_per_buffer
            )
            while record:
                if (stream.is_stopped()):
                    stream.start_stream()

                audiodata_chunks = list[bytes]()
                for _ in range(0, frames_per_buffer):
                    audiodata_chunks.append(stream.read(frames_per_buffer))
                stream.stop_stream()
                frames = utils.frame_generator(20, b"".join(audiodata_chunks), self.RATE)
                new_voice_segments, cont = utils.vad_collector(
                    self.RATE, 20, 400, self.vad, frames
                )
                voice_segments.extend(new_voice_segments)
                record = cont
        except Exception as e:
            _logger.error(f"error occured: {e}")
        finally:
            stream.stop_stream()
            stream.close()

        if len(voice_segments) == 0:
            return AudioTranscieverOutput(
                output=RecordAdaptiveVoiceOutput(text="")
            )

        voiceonly_fp = self._get_tmp_fp("temp_recording_voiceonly", "wav")
        with wave.open(voiceonly_fp, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(voice_segments))

        with open(voiceonly_fp, "rb") as read_buffer:
            audio_transcribe =self.openai_client.audio.transcriptions.create(
                file=read_buffer, model="whisper-1"
            )
            transcribed_audio = audio_transcribe.text
            _logger.info("[transcription] - {}".format(transcribed_audio))
            return AudioTranscieverOutput(
                output=RecordAdaptiveVoiceOutput(
                    text=transcribed_audio
                )
            )

    def _record_voice_manual(self, record_intervals: int) -> AudioTranscieverOutput:
        record_fp = self._get_tmp_fp("temp_recording", "wav")
        with wave.open(record_fp, "wb") as write_buffer:
            stream = self.pyaudio_instance.open(
                self.RATE, self.CHANNELS, self.FORMAT, input=True
            )
            write_buffer.setnchannels(self.CHANNELS)
            write_buffer.setsampwidth(
                self.pyaudio_instance.get_sample_size(self.FORMAT)
            )
            write_buffer.setframerate(self.RATE)

            _logger.info("[recording-start]")
            for _ in range(0, self.RATE // 1024 * record_intervals):
                write_buffer.writeframes(stream.read(1024))
            _logger.info("[recording-end]")

        with wave.open(record_fp, "rb") as read_buffer:
            audio_transcribe = self.openai_client.audio.transcriptions.create(
                file=read_buffer, model="whisper-1"
            )
            transcribed_audio = audio_transcribe.text
            _logger.info("[transcription] - {}".format(transcribed_audio))
            return AudioTranscieverOutput(
                output=RecordVoiceOutput(
                    control_type="record_voice", text=transcribed_audio
                )
            )

    def _output_voice(self, text: str) -> AudioTranscieverOutput:
        voice_output_fp = self._get_tmp_fp("voice_output", "mp3")

        with contextlib.closing(
            self.openai_client.audio.speech.create(
                input=text, model="tts-1", voice="alloy", response_format="mp3"
            )
        ) as response, open(voice_output_fp, mode="wb") as audio_file:
            audio_file.write(response.content)

        audio = AudioSegment.from_mp3(voice_output_fp)
        raw_data = audio.raw_data

        stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(audio.sample_width),
            channels=audio.channels,
            rate=audio.frame_rate,
            output=True,
        )
        stream.write(raw_data)

        return AudioTranscieverOutput(
            output=OutputVoiceOutput(
                status="success",
            )
        )
