import anthropic
from anthropic.types import Message, MessageParam, ToolParam


class AnthropicLLMClient:
    _client: anthropic.Client
    _max_tokens: int
    _model: str

    def __init__(self, api_key: str, max_tokens: int, model: str):
        self._max_tokens = max_tokens
        self._model = model
        self._client = anthropic.Client(api_key=api_key)

    def invoke(
        self, directive: str, messages: list[MessageParam], tools: list[ToolParam]
    ) -> Message:
        return self._client.messages.create(
            max_tokens=self._max_tokens,
            model=self._model,
            messages=messages,
            tools=tools,
            system=directive,
        )
