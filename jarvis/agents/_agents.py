import abc
import typing

from anthropic.types import MessageParam

from jarvis.clients.llm_client import AnthropicLLMClient
from jarvis.tools._tool import AnthropicTool


class Agent(abc.ABC):
    _directive: str
    _tools: dict[str, AnthropicTool[typing.Any, typing.Any]]
    _anthropic_client: AnthropicLLMClient
    _memory: list[MessageParam]

    @abc.abstractmethod
    def __init__(self, directive: str, anthropic_client: AnthropicLLMClient): ...

    @abc.abstractmethod
    def act(self, request: str) -> None: ...

    def directive(self) -> str:
        return self._directive

    def register_tool(self, tool: AnthropicTool[typing.Any, typing.Any]) -> None:
        self._tools[tool.get_name()] = tool

    def unregister_tool(self, name: str) -> None:
        del self._tools[name]
