import abc
import json
import logging
import typing

from anthropic.types import ToolParam
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class Tool[_Controls: BaseModel, _Output: BaseModel](abc.ABC):
    controls: BaseModel

    def use(self, input: dict) -> _Output:
        return self._use(self.transform(input))

    @classmethod
    def transform(cls, input: dict) -> _Output:
        try:
            return cls.controls.model_validate(input)
        except Exception:
            _logger.warning("Cannot deserialize directly")

        # If the model can generate an approximate input we still try to utilize it
        input_obj_str = input.get("input", "{}")
        input_obj: dict[str, typing.Any] = json.loads(input_obj_str)
        return cls.controls.model_validate({"input": input_obj})

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
