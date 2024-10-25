import logging
import typing

from anthropic.types import ContentBlock, MessageParam

from jarvis.agents._agents import Agent, MetaAgent
from jarvis.clients.llm_client import LLMClient
from jarvis.exceptions import RequestComplete
from jarvis.tools._tool import AnthropicTool

_logger = logging.getLogger(__name__)


class BaseAgent(Agent):
    def __init__(self, directive: str, llm_client: LLMClient):
        self._directive = directive
        self._llm_client = llm_client
        self._tools = dict[str, AnthropicTool[typing.Any, typing.Any]]()
        self._memory = list[MessageParam]()

    @typing.override
    def act(self, request: str) -> None:
        try:
            self._act(request)
        except RequestComplete as e:
            print(e.reason)
        except Exception as e:
            raise RuntimeError(e)

    def _act(self, request: str) -> None:
        while True:
            self._memory.append(MessageParam(role="user", content=request))
            response: list[ContentBlock] = self._llm_client.invoke(
                self._memory, [t.tool_description() for t in self._tools.values()]
            )

            assistant_message = ""
            output_message = ""
            for cb in response:
                if cb.type == "text":
                    assistant_message += f"${cb.text}\n"
                    _logger.info(f"assistant-message: ${cb.text}\n")
                elif cb.type == "tool_use":
                    tool_name = cb.name
                    tool = self._tools.get(tool_name)
                    message_part = f"Use `${tool_name}` tool with input: `${cb.input}`"
                    assistant_message += message_part
                    _logger.info(f"assistant-tool-use: ${message_part}")

                    if tool is not None:
                        output = tool.use(cb.input)
                        output_message += (
                            f"Output of using `${tool_name}` tool: ${output}"
                        )
                    else:
                        output_message += (
                            f"Couldn't find the tool named `${tool_name}`."
                        )

            self._memory.append(
                MessageParam(role="assistant", content=assistant_message)
            )
            self._memory.append(MessageParam(role="user", content=output_message))


class BaseMetaAgent(MetaAgent, BaseAgent):
    _agents: dict[str, Agent]

    def __init__(self, directive: str, llm_client: LLMClient):
        self._directive = directive
        self._llm_client = llm_client
        self._tools = dict[str, AnthropicTool[typing.Any, typing.Any]]()
        self._memory = list[MessageParam]()
        self._agents = dict[str, Agent]()
