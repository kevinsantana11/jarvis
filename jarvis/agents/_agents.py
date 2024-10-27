import abc
import logging
import typing

from anthropic.types import MessageParam, ToolParam

from jarvis.clients.llm_client import AnthropicLLMClient
from jarvis.exceptions import RequestComplete
from jarvis.tools._tool import AnthropicTool

_logger = logging.getLogger(__name__)


class Agent(abc.ABC):
    _directive: str
    _tools: dict[str, AnthropicTool[typing.Any, typing.Any]]
    _anthropic_client: AnthropicLLMClient
    _memory: list[MessageParam]

    @abc.abstractmethod
    def __init__(self, directive: str, anthropic_client: AnthropicLLMClient): ...

    @abc.abstractmethod
    def _act(self) -> None: ...

    def directive(self) -> str:
        return self._directive

    def tool_descriptions(self) -> list[ToolParam]:
        return [t.tool_description() for t in self._tools.values()]

    def register_tool(self, tool: AnthropicTool[typing.Any, typing.Any]) -> None:
        self._tools[tool.get_name()] = tool

    def unregister_tool(self, name: str) -> None:
        del self._tools[name]

    def act(self, request: str) -> None:
        try:
            self._act_eventloop(request)
        except RequestComplete as e:
            print(e.reason)
        except Exception as e:
            _logger.error(f"Error trying to execute request: {e}")
            _logger.error(f"Memory dump: {str(self._memory)}")
            raise RuntimeError(e)

    def _act_eventloop(self, request: str) -> None:
        self._memory.append({"role": "user", "content": request})
        while True:
            self._act()
