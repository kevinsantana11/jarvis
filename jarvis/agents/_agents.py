import abc

from anthropic.types import ToolParam

from jarvis.clients.llm_client import LLMClient, StatefulLLMClient


class Agent(abc.ABC):
    _directive: str
    _tools: dict[str, ToolParam]
    _llm_client: StatefulLLMClient

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


class BaseAgent(Agent):
    def __init__(self, directive: str, llm_client: StatefulLLMClient):
        self._directive = directive
        self._llm_client = llm_client
        self._tools = dict[str, ToolParam]()

    def act(self, request: str) -> None:
        self._llm_client.invoke(request, list(self._tools.values()))


class MetaAgent(BaseAgent, abc.ABC):
    _agents: dict[str, Agent]

    @abc.abstractmethod
    def __init__(self, directive: str, llm_client: StatefulLLMClient): ...

    @abc.abstractmethod
    def orchestrate(self, request: str) -> None: ...

    def register_agent(self, name: str, agent: Agent) -> None:
        self._agents[name] = agent

    def unregister_agent(self, name: str) -> None:
        del self._agents[name]


class BaseMetaAgent(BaseAgent, abc.ABC):
    _agents: dict[str, Agent]

    def __init__(self, directive: str, llm_client: StatefulLLMClient):
        self._agent = dict[str, Agent]()
        super().__init__(directive, llm_client)

    @abc.abstractmethod
    def orchestrate(self, request: str) -> None: ...
