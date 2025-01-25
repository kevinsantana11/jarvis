import json
import logging
import typing

import anthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
)

from jarvis.agents._agents import Agent
from jarvis.tools._tool import AnthropicTool

_logger = logging.getLogger(__name__)


class AnthropicAgent(Agent):
    _tools: dict[str, AnthropicTool[typing.Any, typing.Any]]
    _anthropic_client: anthropic.Client

    def __init__(
        self, directive: str, max_tokens: int, anthropic_client: anthropic.Client
    ):
        self._tools = dict[str, AnthropicTool[typing.Any, typing.Any]]()
        self._anthropic_client = anthropic_client
        super().__init__(directive, max_tokens)

    def sonnet(self):
        return self._anthropic_client.messages.create(
            max_tokens=self._max_tokens,
            model="claude-3-5-sonnet-latest",
            messages=self._memory,
            tools=self.tool_descriptions(),
            system=self.directive(),
        )

    def haiku(self):
        return self._anthropic_client.messages.create(
            max_tokens=self._max_tokens,
            model="claude-3-haiku-20240307",
            messages=self._memory,
            tools=self.tool_descriptions(),
            system=self.directive(),
        )

    def tool_descriptions(self) -> list[ToolParam]:
        return [t.tool_description() for t in self._tools.values()]

    def register_tool(self, tool: AnthropicTool[typing.Any, typing.Any]) -> None:
        self._tools[tool.get_name()] = tool

    def unregister_tool(self, name: str) -> None:
        del self._tools[name]

    @typing.override
    def _act(self) -> None:
        response = self.sonnet()

        user_content = list[TextBlockParam | ToolResultBlockParam]()
        for cb in response.content:
            _logger.info(cb)
            if cb.type == "tool_use":
                tool_name = cb.name
                tool = self._tools.get(tool_name)

                if tool is not None:
                    output = tool.use(cb.input)
                    user_content.append(
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=cb.id,
                            is_error=False,
                            content=output.model_dump_json(),
                        )
                    )
                else:
                    user_content.append(
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=cb.id,
                            is_error=True,
                            content=json.dumps(
                                {
                                    "error": f"Couldn't find tool with tool name: `{tool_name}`"
                                }
                            ),
                        )
                    )

        self._memory.append(MessageParam(role="assistant", content=response.content))
        if len(user_content) == 0:
            user_content.append(
                TextBlockParam(type="text", text="system-message: <no output detected>")
            )
        self._memory.append(MessageParam(role="user", content=user_content))
