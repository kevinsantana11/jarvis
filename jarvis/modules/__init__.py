from .audio_in_module import AudioInputModule
from .interfaces import InputModule, OutputModule
from .reasoning_module import ReasoningModule


class TextInputModule(InputModule):
    def __call__(self) -> str:
        return input("\n\nINPUT:: ")


class PrintOutModule(OutputModule):
    def __call__(self, input_: str):
        print(f"OUTPUT:: {input_}")


__all__ = [
    "AudioInputModule",
    "ReasoningModule",
    "TextInputModule",
    "PrintOutModule",
]
