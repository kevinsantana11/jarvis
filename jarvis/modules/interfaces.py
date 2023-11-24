from abc import ABC, abstractmethod


class InputModule(ABC):
    """
    Module for retrieving agent inputs from the user (via text, audio/stt, etc...).
    """

    @abstractmethod
    def __call__(self) -> str:
        raise RuntimeError("Needs to be implemented")


class ProcessingModule(ABC):
    """
    Module that consumes agent input, processes it and finally returns an output.
    """

    @abstractmethod
    def __call__(self, input_: str) -> str:
        raise RuntimeError("Needs to be implemented")


class OutputModule(ABC):
    """
    Module that consumes some value and outputs it to the user (via text, audio/tts, etc...)
    """

    @abstractmethod
    def __call__(self, input_: str):
        raise RuntimeError("needs to be implemented")
