import abc
import typing

import anthropic
from anthropic.types import MessageParam, ToolParam


class _LLMClient(abc.ABC):
    @abc.abstractmethod
    def __init__(self, api_key: str, max_token: int, model: str): ...

    @abc.abstractmethod
    def invoke(
        self, requests: str, tools: list[ToolParam], memory: list[MessageParam]
    ) -> list[MessageParam]: ...


class LLMClient(_LLMClient):
    _client: anthropic.Client
    _max_tokens: int
    _model: str

    def __init__(self, api_key: str, max_tokens: int, model: str):
        self._max_tokens = max_tokens
        self._model = model
        self._client = anthropic.Client(api_key=api_key)

    @typing.override
    def invoke(
        self, request: str, tools: list[ToolParam], memory: list[MessageParam]
    ) -> list[MessageParam]:
        new_memory = memory + list([MessageParam(content=request, role="user")])
        response = self._client.messages.create(
            messages=new_memory,
            max_tokens=self._max_tokens,
            model=self._model,
            tools=tools,
        )
        return new_memory + list([MessageParam(content=response.content, role="user")])

    def invoke(
        self, request: str, memory: list[MessageParam], tools: list[ToolParam]
    ) -> list[MessageParam]:
        self._invoke(request, memory, tools)



class StatefulLLMClient(LLMClient):
    _memory: list[MessageParam]

    def __init__(self, api_key: str, max_token: int, model: str):
        self._memory = list[MessageParam]()
        super().__init__(api_key, max_token, model)

    @typing.override
    def invoke(self, request: str, tools: list[ToolParam], memory: list[MessageParam] | None = None) -> list[MessageParam]:
        self._memory = super().invoke(request, tools, memory or self._memory)
        return self._memory

    def reset(self) -> None:
        self._memory = list()

    def back(self) -> None:
        self._memory = self._memory[:-2]

    def forward(self) -> None:
        self._memory = self._memory[2:]
