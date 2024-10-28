from unittest.mock import Mock, patch

import openai
import pyaudio
import pytest
import webrtcvad

from jarvis.tools.audio_transciever import (
    AudioTransciever,
    AudioTranscieverControls,
    OutputVoiceInput,
    RecordAdaptiveVoiceInput,
    RecordVoiceInput,
)


@pytest.fixture
def mock_openai_client():
    client = Mock(spec=openai.Client)

    # Create the nested audio structure
    client.audio = Mock()
    client.audio.transcriptions = Mock()
    client.audio.speech = Mock()

    # Mock the transcription response
    transcription = Mock()
    transcription.text = "test transcription"
    client.audio.transcriptions.create.return_value = transcription

    # Mock the speech response
    speech_response = Mock()
    speech_response.content = b"mock audio content"
    client.audio.speech.create.return_value = speech_response

    return client


@pytest.fixture
def mock_vad():
    return Mock(spec=webrtcvad.Vad)


@pytest.fixture
def mock_pyaudio():
    with patch("pyaudio.PyAudio") as mock:
        mock_stream = Mock()
        mock_stream.read.return_value = b"mock audio data"
        mock_stream.is_stopped.return_value = False
        mock_stream.write = Mock()

        mock_instance = mock.return_value
        mock_instance.open.return_value = mock_stream
        mock_instance.get_sample_size.return_value = 2
        mock_instance.get_format_from_width.return_value = pyaudio.paInt16

        yield mock_instance


@pytest.fixture
def audio_transciever(mock_openai_client, mock_vad, mock_pyaudio):
    return AudioTransciever(mock_openai_client, mock_vad)


def test_audio_transciever_initialization(audio_transciever):
    """Test that AudioTransciever initializes correctly"""
    assert isinstance(audio_transciever, AudioTransciever)
    assert audio_transciever.CHANNELS == 1
    assert audio_transciever.RATE == 32_000
    assert audio_transciever.FORMAT == pyaudio.paInt16


def test_record_voice_manual(audio_transciever):
    """Test manual voice recording functionality"""
    control = AudioTranscieverControls(
        input=RecordVoiceInput(control_type="record_voice", record_intervals=5)
    )

    result = audio_transciever._use(control)

    assert result.error is False
    assert result.output is not None
    assert result.output.text == "test transcription"


@patch("jarvis.tools.audio_transciever.AudioSegment")
def test_output_voice(mock_audio_segment, audio_transciever):
    """Test voice output functionality"""
    # Create a mock audio segment
    mock_audio = Mock()
    mock_audio.raw_data = b"mock raw audio data"
    mock_audio.sample_width = 2
    mock_audio.channels = 1
    mock_audio.frame_rate = 44100

    # Set up the from_mp3 class method
    mock_audio_segment.from_mp3.return_value = mock_audio

    control = AudioTranscieverControls(
        input=OutputVoiceInput(
            control_type="output_voice", text="Hello, this is a test"
        )
    )

    result = audio_transciever._use(control)

    assert result.error is False
    assert result.output is not None
    assert result.output.status == "success"


@patch("jarvis.tools.audio_transciever.utils.frame_generator")
@patch("jarvis.tools.audio_transciever.utils.vad_collector")
def test_record_voice_adaptive(
    mock_vad_collector, mock_frame_generator, audio_transciever
):
    """Test adaptive voice recording functionality"""
    mock_frame_generator.return_value = [b"frame1", b"frame2"]
    mock_vad_collector.return_value = ([b"voice segment"], False)

    control = AudioTranscieverControls(
        input=RecordAdaptiveVoiceInput(
            control_type="record_voice_adaptive", frame_duration=5
        )
    )

    result = audio_transciever._use(control)

    assert result.error is False
    assert result.output is not None
    assert result.output.text == "test transcription"


def test_error_handling(audio_transciever):
    """Test error handling when recording fails"""
    # Create a new mock stream that raises the expected exception
    error_stream = Mock()
    error_stream.read = Mock(side_effect=Exception("Recording failed"))

    # Replace the PyAudio instance's open method to return our error stream
    audio_transciever.pyaudio_instance.open = Mock(return_value=error_stream)

    control = AudioTranscieverControls(
        input=RecordVoiceInput(control_type="record_voice", record_intervals=5)
    )

    result = audio_transciever._use(control)

    assert result.error is True
    assert result.reason == "Recording failed"
    assert result.output is None


def test_get_name():
    """Test get_name class method"""
    assert AudioTransciever.get_name() == "AudioTransciever"


def test_get_description():
    """Test get_description class method"""
    description = AudioTransciever.get_description()
    assert isinstance(description, str)
    assert "voice" in description.lower()
    assert "whisper" in description.lower()
