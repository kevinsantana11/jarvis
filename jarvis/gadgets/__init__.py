from abc import ABC
from typing import Literal

from langchain.pydantic_v1 import BaseModel


class GadgetMetadata(BaseModel):
    name: str
    documentation: str
    usage_instructions: str


class Gadget(ABC):
    def get_documentation(self) -> str:
        raise RuntimeError("Not implemented yet")

    def get_usage_instructions(self) -> str:
        raise RuntimeError("Not implemented yet")

    def get_name(self) -> str:
        raise RuntimeError("Not implemented yet")

    def get_metadata(self) -> GadgetMetadata:
        return GadgetMetadata(
            name=self.get_name(),
            documentation=self.get_documentation(),
            usage_instructions=self.get_usage_instructions(),
        )


class StopGadget(Gadget):
    def get_name(self) -> str:
        return "stop"

    def get_documentation(self) -> str:
        return "See usage instructions"

    def get_usage_instructions(self) -> str:
        return "Use this gadget when you want to stop execution and return a message"


class StopAction(BaseModel):
    name: Literal["stop"]
    reason: str
