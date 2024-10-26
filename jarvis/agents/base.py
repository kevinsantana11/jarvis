import json
import logging
import typing

from anthropic.types import ContentBlock, MessageParam, TextBlockParam, ToolUseBlockParam, ToolResultBlockParam

from jarvis.agents._agents import Agent, MetaAgent
from jarvis.clients.llm_client import AnthropicLLMClient
from jarvis.exceptions import RequestComplete
from jarvis.tools._tool import AnthropicTool

_logger = logging.getLogger(__name__)


class BaseAgent(Agent):
    def __init__(self, directive: str, anthropic_client: AnthropicLLMClient):
        self._directive = directive
        self._anthropic_client = anthropic_client
        self._tools = dict[str, AnthropicTool[typing.Any, typing.Any]]()
        self._memory = list[MessageParam]()

    @typing.override
    def act(self, request: str) -> None:
        try:
            self._act(request)
        except RequestComplete as e:
            print(e.reason)
        except Exception as e:
            _logger.error(f"Error trying to execute request: {e}")
            _logger.error(f"Memory dump: {str(self._memory)}")
            raise RuntimeError(e)

    def _act(self, request: str) -> None:
        self._memory.append({"role": "user", "content": request})
        tools = [t.tool_description() for t in self._tools.values()]

        while True:
            response =  self._anthropic_client.invoke(self.directive(), self._memory, tools)
            user_content = list[TextBlockParam | ToolResultBlockParam]()
            for cb in response.content:
                _logger.info(cb)
                if cb.type == "tool_use":
                    tool_name = cb.name
                    tool = self._tools.get(tool_name)

                    if tool is not None:
                        output = tool.use(cb.input)
                        user_content.append(ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=cb.id,
                            is_error=False,
                            content=output.model_dump_json(),
                        ))
                    else:
                        user_content.append(ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=cb.id,
                            is_error=True,
                            content=json.dumps({"error": f"Couldn't find tool with tool name: `{tool_name}`"}),
                        ))
                        

            self._memory.append(
                MessageParam(role="assistant", content=response.content)
            )
            if len(user_content) == 0:
                user_content.append(TextBlockParam(type="text", text="system-message: <no output detected>"))
            self._memory.append(MessageParam(role="user", content=user_content))
