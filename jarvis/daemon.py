from __future__ import annotations

import logging

import anthropic
import dotenv
import openai
import webrtcvad
from anthropic.types import MessageParam, ToolParam
from pydantic import BaseModel

from jarvis.agents import AnthropicAgent
from jarvis.config import Config
from jarvis.tools import AudioTransciever
from jarvis.tools.audio_transciever import (
    AudioTranscieverControls,
    RecordAdaptiveVoiceInput,
)
from jarvis.tools.google_search import GoogleSearch
import googleapiclient

_logger = logging.getLogger(__name__)


dotenv.load_dotenv()


class DaemonControl(BaseModel):
    activate: bool


class Daemon:
    audio_transciever: AudioTransciever
    anthropic_client: anthropic.Client
    jarvis_agent: AnthropicAgent
    config: Config

    def __init__(
        self,
        audio_transciever: AudioTransciever,
        jarvis_agent: AnthropicAgent,
        config: Config,
        anthropic_client: anthropic.Client,
    ) -> None:
        self.audio_transciever = audio_transciever
        self.jarvis_agent = jarvis_agent
        self.anthropic_client = anthropic_client
        self.config = config
        logging.basicConfig(level=config.loglevel)

    @classmethod
    def default(cls) -> Daemon:
        config = Config()
        audio_transciever = AudioTransciever(
            openai_client=openai.Client(
                api_key=config.openai_api_key,
            ),
            vad=webrtcvad.Vad(mode=2),
        )
        google_search = GoogleSearch(
            googleapiclient.discovery.build(
                "customsearch",
                "v1",
                developerKey=config.google_api_key,
            ),
            config.google_search_engine_id,
        )

        anthropic_client = anthropic.Client(api_key=config.anthropic_api_key)
        jarvis_agent = AnthropicAgent(
            directive="""
            You are an agent and your goal is to engage with the user and chat
            with to learn more about them. Use the transciever to prompt the
            user with questions and to hear their feedback. Tool use blocks
            should be kept separately from text blocks.
            """,
            max_tokens=4096,
            anthropic_client=anthropic_client,
        )
        jarvis_agent.register_tool(audio_transciever)
        jarvis_agent.register_tool(google_search)
        daemon = Daemon(
            audio_transciever=audio_transciever,
            jarvis_agent=jarvis_agent,
            config=config,
            anthropic_client=anthropic_client,
        )
        _logger.info("Successfully started daemon")
        return daemon

    def activation_func(self, activation_phrase, sensitivity, input_):
        """Scans through the input and checks if a a word
        similar to the activation phrase can be found. Activation is triggered
        based on the configured sensitivity. Higher sensitivity will match words
        with only a few similar characters while lower sensitivity requires matching
        more characters.
        """

        ap_idx = 0
        left = 0
        activation_phrase_map = dict()
        for char in activation_phrase.lower():
            cnt = activation_phrase_map.get(char, 0)
            activation_phrase_map[char] = cnt + 1

        input_ = input_.lower()
        matches = 0
        for right, char in enumerate(input_):
            if (
                input_[right] in activation_phrase_map
                and activation_phrase_map[input_[right]] == 0
            ):
                while activation_phrase_map[input_[right]] == 0:
                    if input_[left] in activation_phrase_map:
                        activation_phrase_map[input_[left]] += 1
                    left += 1
            elif input_[right] not in activation_phrase_map:
                while left <= right:
                    if input_[left] in activation_phrase_map:
                        activation_phrase_map[input_[left]] += 1
                    left += 1
            else:
                matches = max(matches, right - left + 1)
                activation_phrase_map[input_[right]] -= 1

        portion = matches / len(activation_phrase)
        return portion > (1.0 - sensitivity)

    def run(self) -> None:
        activation_phrase = "jarvis"
        while True:
            tool_output = self.audio_transciever.use(
                AudioTranscieverControls(input=RecordAdaptiveVoiceInput())
            )

            if tool_output.output.text != "" and self.activation_func(
                activation_phrase, 0.50, tool_output.output.text
            ):
                self.jarvis_agent.act(tool_output.output.text)
