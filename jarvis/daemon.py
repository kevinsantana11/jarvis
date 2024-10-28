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
        daemon = Daemon(
            audio_transciever=audio_transciever,
            jarvis_agent=jarvis_agent,
            config=config,
            anthropic_client=anthropic_client,
        )
        _logger.info("Successfully started daemon")
        return daemon

    def run(self) -> None:
        activation_phrase = "jarvis"
        while True:
            tool_output = self.audio_transciever.use(
                AudioTranscieverControls(input=RecordAdaptiveVoiceInput())
            )

            if tool_output.output.text != "":
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2048,
                    system=f"""
                    You are an activation agent, and your goal is to detect whether the user
                    input includes the activation phrase.

                    The activation phrase is: `{activation_phrase}`
                    """,
                    messages=[
                        MessageParam(role="user", content=tool_output.output.text)
                    ],
                    tools=[
                        ToolParam(
                            name="DaemonController",
                            input_schema=DaemonControl.model_json_schema(),
                            description="Utilize this tool to control the daemon.",
                        )
                    ],
                )

                for cb in response.content:
                    if cb.type == "tool_use":
                        tool_name = cb.name
                        if tool_name == "DaemonController":
                            input = DaemonControl(**cb.input)

                            if input.activate:
                                self.jarvis_agent.act(tool_output.output.text)
