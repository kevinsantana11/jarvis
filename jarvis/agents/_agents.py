import abc
import typing

from anthropic.types import ToolParam

from jarvis.clients.llm_client import LLMClient


class Agent(abc.ABC):
    _directive: str
    _tools: dict[str, ToolParam]
    _llm_client: LLMClient

    @abc.abstractmethod
    def __init__(self, directive: str, llm_client: LLMClient): ...

    @abc.abstractmethod
    def act(self, request: str) -> None: ...

    @abc.abstractmethod
    def initialize(self) -> None: ...

    def directive(self) -> str:
        return self._directive

    def register_tool(self, name: str, tool: ToolParam) -> None:
        self._tools[name] = tool

    def unregister_tool(self, name: str) -> None:
        del self._tools[name]


class MetaAgent(abc.ABC):
    _agents: dict[str, Agent]

    @abc.abstractmethod
    def __init__(self, directive: str, llm_client: LLMClient): ...

    @abc.abstractmethod
    def orchestrate(self, request: str) -> None: ...

    def register_agent(self, name: str, agent: Agent) -> None:
        self._agents[name] = agent

    def unregister_agent(self, name: str) -> None:
        del self._agents[name]


class BaseAgent(Agent):
    def __init__(self, directive: str, llm_client: LLMClient):
        self._directive = directive
        self._llm_client = llm_client
        self._tools = dict[str, ToolParam]()

    @typing.override
    def act(self, request: str) -> None:
        raise NotImplementedError()


class BaseMetaAgent(MetaAgent):
    _agents: dict[str, Agent]

    def __init__(self, directive: str, llm_client: LLMClient):
        self._directive = directive
        self._llm_client = llm_client
        self._agent = dict[str, Agent]()

    @abc.abstractmethod
    def orchestrate(self, request: str) -> None:
        raise NotImplementedError()
