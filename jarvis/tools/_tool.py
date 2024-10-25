import abc

from anthropic.types import ToolParam
from pydantic import BaseModel


class Tool[_Controls: BaseModel, _Output: BaseModel](abc.ABC):
    controls: _Controls

    @abc.abstractmethod
    def use(self, control_request: _Controls) -> _Output: ...

    @abc.abstractmethod
    @classmethod
    def get_name(self) -> str: ...

    @abc.abstractmethod
    @classmethod
    def get_description(self) -> str: ...


class AnthropicTool[_Controls: BaseModel, _Output: BaseModel](
    Tool[_Controls, _Output], abc.ABC
):
    def tool_description(self) -> ToolParam:
        return ToolParam(
            input_schema=self.controls.model_json_schema(),
            name=self.get_name(),
            description=self.get_description(),
        )
