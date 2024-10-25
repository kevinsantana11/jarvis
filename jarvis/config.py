import os
import typing


class Config:
    anthropic_api_key: str
    loglevel: str

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        loglevel = os.environ.get("LOGLEVEL", "INFO")

        if anthropic_api_key is None:
            raise ValueError("Please provide a value for the `ANTHROPIC_API_KEY` environment variable")

        self.anthropic_api_key = anthropic_api_key
        self.loglevel = loglevel

