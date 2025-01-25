import os
import typing


class Config:
    anthropic_api_key: str
    openai_api_key: str
    google_api_key: str
    google_search_engine_id: str
    loglevel: str

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.google_search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        self.loglevel = os.environ.get("LOGLEVEL", "INFO")
