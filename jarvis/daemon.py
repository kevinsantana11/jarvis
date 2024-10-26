from __future__ import annotations

import logging

import dotenv
import openai

from jarvis.agents import BaseAgent
from jarvis.clients.llm_client import AnthropicLLMClient
from jarvis.config import Config
from jarvis.tools import AudioTransciever

_logger = logging.getLogger(__name__)


dotenv.load_dotenv()


class WakeDaemon:
    audio_transciever: AudioTransciever
    jarvis_agent: BaseAgent
    config: Config

    def __init__(
        self,
        audio_transciever: AudioTransciever,
        jarvis_agent: BaseAgent,
        config: Config,
    ) -> None:
        self.audio_transciever = audio_transciever
        self.jarvis_agent = jarvis_agent
        self.config = config
        logging.basicConfig(level=config.loglevel)

    @classmethod
    def default(cls) -> WakeDaemon:
        config = Config()
        audio_transciever = AudioTransciever(
            openai_client=openai.Client(
                api_key=config.openai_api_key,
            )
        )
        jarvis_agent = BaseAgent(
            # directive=""""
            # You are a meta agent and your goal is to help users with their
            # requests. You will have several tools and agents that can be
            # leveraged to complete the users request. If the request cannot
            # be completed using the available tools, short circuit the
            # process and mention the limitation to the user.
            # """,
            directive="""
            You are an agent and your goal is to engage with the user and chat
            with to learn more about them. Use the transciever to prompt the
            user with questions and to hear their feedback. Tool use blocks
            should be kept separately from text blocks.
            """,
            anthropic_client=AnthropicLLMClient(
                api_key=config.anthropic_api_key,
                max_tokens=1024,
                model="claude-3-5-sonnet-20241022",
            ),
        )
        jarvis_agent.register_tool(audio_transciever)
        daemon = WakeDaemon(
            audio_transciever=audio_transciever,
            jarvis_agent=jarvis_agent,
            config=config,
        )
        _logger.info("Successfully started daemon")
        return daemon

    def run(self) -> None:
        self.jarvis_agent.act("Initiate conversation")
