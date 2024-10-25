import abc
import typing

from anthropic.types import MessageParam

from jarvis.clients.llm_client import LLMClient
from jarvis.tools._tool import AnthropicTool


class Agent(abc.ABC):
    _directive: str
    _tools: dict[str, AnthropicTool[typing.Any, typing.Any]]
    _llm_client: LLMClient
    _memory: list[MessageParam]

    @abc.abstractmethod
    def __init__(self, directive: str, llm_client: LLMClient): ...

    @abc.abstractmethod
    def act(self, request: str) -> None: ...

    def directive(self) -> str:
        return self._directive

    def register_tool(
        self, name: str, tool: AnthropicTool[typing.Any, typing.Any]
    ) -> None:
        self._tools[name] = tool

    def unregister_tool(self, name: str) -> None:
        del self._tools[name]


class MetaAgent(Agent, abc.ABC):
    _agents: dict[str, Agent]

    def register_agent(self, name: str, agent: Agent) -> None:
        self._agents[name] = agent

    def unregister_agent(self, name: str) -> None:
        del self._agents[name]
