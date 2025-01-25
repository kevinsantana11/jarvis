import abc
import logging

from anthropic.types import MessageParam

from jarvis.exceptions import RequestComplete

_logger = logging.getLogger(__name__)


class Agent(abc.ABC):
    _directive: str
    _memory: list[MessageParam]
    _max_tokens: int

    def __init__(self, directive: str, max_tokens: int):
        self._directive = directive
        self._max_tokens = max_tokens
        self._memory = list[MessageParam]()

    @abc.abstractmethod
    def _act(self) -> None: ...

    def clear_mem(self):
        self._memory = list[MessageParam]()

    def directive(self) -> str:
        return self._directive

    def act(self, request: str) -> None:
        try:
            self._act_eventloop(request)
        except RequestComplete as e:
            print(e.reason)
        except Exception as e:
            _logger.error(f"Error trying to execute request: {e}")
            _logger.error(f"Memory dump: {str(self._memory)}")
            raise e

    def _act_eventloop(self, request: str) -> None:
        self._memory.append({"role": "user", "content": request})
        while True:
            self._act()
