from __future__ import annotations

from typing import Any, Literal, override
import typing
import os
import googleapiclient.discovery
from pydantic import BaseModel, Field
from jarvis.tools._tool import AnthropicTool
if typing.TYPE_CHECKING:
    from googleapiclient._apis.customsearch.v1 import CustomSearchAPIResource


class RunQueryInput(BaseModel):
    control_type: Literal["run_query"] = "run_query"
    query: str


class SearchResult(BaseModel):
    url: str
    title: str
    description: str


class RunQueryOutput(BaseModel):
    control_type: Literal["run_query"] = "run_query"
    results: list[SearchResult]


class GoogleSearchControls(BaseModel):
    input: RunQueryInput = Field(..., discriminator="control_type")


class GoogleSearchOutput(BaseModel):
    output: RunQueryOutput | None = Field(..., discriminator="control_type")
    error: bool = False
    reason: str | None = None


class GoogleSearch(AnthropicTool[GoogleSearchControls, GoogleSearchOutput]):
    controls = GoogleSearchControls
    google_client: CustomSearchAPIResource

    def __init__(self, google_client: CustomSearchAPIResource) -> None:
        self.google_client = google_client

    @typing.override
    @classmethod
    def get_name(cls) -> str:
        return "GoogleSearch"

    @typing.override
    @classmethod
    def get_description(cls) -> str:
        return """This tool can be used to query the google search engine for 
        information.
        """

    @typing.override
    def _use(self, control_request: GoogleSearchControls) -> GoogleSearchOutput:
        try:
            if isinstance(control_request.input, RunQueryInput):
                return self._run_query(control_request.input.query)
        except Exception as e:
            return GoogleSearchOutput(output=None, error=True, reason=str(e))
        ...

    def _run_query(self, query: str) -> RunQueryOutput:
        res = self.google_client.cse().list(query=query).execute()
        return RunQueryOutput(
            control_type="run_query",
            results=[
                SearchResult(
                    url=item.link, title=item.title, description=item.htmlSnippet
                )
                for item in res.items
            ]
        )
