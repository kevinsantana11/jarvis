import abc

from anthropic.types import ToolParam
from pydantic import BaseModel


class Tool[_Controls: BaseModel, _Output: BaseModel](abc.ABC):
    controls: BaseModel

    def use(self, input: dict) -> BaseModel:
        return self._use(self.transform(input))

    @classmethod
    def transform(cls, input: dict) -> _Output:
        return cls.controls.model_validate(input)

    @abc.abstractmethod
    def _use(cls, control_request: _Controls) -> _Output: ...

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str: ...

    @classmethod
    @abc.abstractmethod
    def get_description(cls) -> str: ...


class AnthropicTool[_Controls: BaseModel, _Output: BaseModel](
    Tool[_Controls, _Output], abc.ABC
):
    @classmethod
    def tool_description(cls) -> ToolParam:
        return ToolParam(
            input_schema=cls.controls.model_json_schema(),
            name=cls.get_name(),
            description=cls.get_description(),
        )
