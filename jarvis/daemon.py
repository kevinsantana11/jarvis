from __future__ import annotations

from jarvis.agents import BaseMetaAgent, MetaAgent
from jarvis.clients.llm_client import LLMClient
from jarvis.tools import AudioTransciever
from jarvis.tools.audio_to_text import AudioTranscieverControls, RecordVoiceInput


class WakeDaemon:
    audio_transciever: AudioTransciever
    jarvis_agent: MetaAgent

    def __init__(
        self, audio_transciever: AudioTransciever, jarvis_agent: MetaAgent
    ) -> None:
        self.audio_transciever = audio_transciever
        self.jarvis_agent = jarvis_agent

    def default(cls) -> WakeDaemon:
        return WakeDaemon(
            audio_transciever=AudioTransciever(),
            jarvis_agent=BaseMetaAgent(
                directive=""""
                You are a meta agent and your goal is to help users with their
                requests. You will have several tools and agents that can be 
                leveraged to complete the users request. If the request cannot
                be completed using the available tools, short circuit the
                process and mention the limitation to the user. 
                """,
                llm_client=LLMClient(
                    api_key="",
                    max_tokens=1024,
                    model="sonnet...",
                ),
            ),
        )

    def run(self) -> None:
        user_request = self.audio_transciever.use(
            AudioTranscieverControls(input=RecordVoiceInput(record_intervals=5))
        )
        if user_request.output.control_type == "record_voice":
            self.jarvis_agent.act(user_request.output.text)
