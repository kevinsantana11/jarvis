import abc
import json
import logging
import typing

from anthropic.types import ToolParam
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class Tool[_Controls: BaseModel, _Output: BaseModel](abc.ABC):
    controls: BaseModel

    def use(self, input_: dict) -> _Output:
        try:
            return self._use(self.transform(input_))
        except Exception as e:
            _logger.error(f"Error trying to utilize tool", e)
            raise e

    @classmethod
    def transform(cls, input_: dict) -> _Output:
        # If the model can generate an approximate input we still try to utilize it
        try:
            return cls.controls.model_validate(input_)
        except Exception as e:
            _logger.warning(f"Cannot deserialize as directly, e: {e}, input: {input_}")

        # Sometimes the model puts the input as the stringified version
        try:
            input_obj_str = input_.get("input", "{}")
            input_obj: dict[str, typing.Any] = json.loads(input_obj_str)
            return cls.controls.model_validate({"input": input_obj})
        except Exception as e:
            _logger.warning(
                f"Cannot deserialize as stringified input, e: {e}, input: {input_}"
            )

        # Sometimes the model doesn't put the action in an input obj
        try:
            return cls.controls.model_validate({"input": input_})
        except Exception as e:
            _logger.error(f"Unable to deserialize input obj, e: {e}, input: {input_}")
            raise e

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
