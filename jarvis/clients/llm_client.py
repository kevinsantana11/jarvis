import abc

import anthropic
from anthropic.types import MessageParam, ToolParam


class _LLMClient(abc.ABC):
    @abc.abstractmethod
    def __init__(self, api_key: str, max_token: int, model: str): ...

    @abc.abstractmethod
    def _invoke(
        self, requests: str, memory: list[MessageParam], tools: list[ToolParam]
    ) -> list[MessageParam]: ...


class LLMClient(_LLMClient):
    _client: anthropic.Client
    _max_tokens: int
    _model: str

    def __init__(self, api_key: str, max_tokens: int, model: str):
        self._max_tokens = max_tokens
        self._model = model
        self._client = anthropic.Client(api_key=api_key)

    def _invoke(
        self, request: str, memory: list[MessageParam], tools: list[ToolParam]
    ) -> list[MessageParam]:
        new_memory = memory + list([MessageParam(content=request, role="user")])
        response = self._client.messages.create(
            messages=new_memory,
            max_tokens=self._max_tokens,
            model=self._model,
            tools=tools,
        )
        return new_memory + list([MessageParam(content=response.content, role="user")])


class StatefulLLMClient(LLMClient):
    _memory: list[MessageParam]

    def __init__(self, api_key: str, max_token: int, model: str):
        self._memory = list[MessageParam]()
        super().__init__(api_key, max_token, model)

    def invoke(self, request: str, tools: list[ToolParam]) -> None:
        self._memory = super()._invoke(request, self._memory, tools)

    def reset(self) -> None:
        self._memory = list()

    def back(self) -> None:
        self._memory = self._memory[:-2]

    def forward(self) -> None:
        self._memory = self._memory[2:]
