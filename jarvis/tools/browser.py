from __future__ import annotations

from typing import Any, Literal, override
import typing
import os
from pydantic import BaseModel, Field
from jarvis.tools._tool import AnthropicTool


class VisitInput(BaseModel):
    control_type: Literal["visit"] = "visit"
    url: str


class VisitOutput(BaseModel):
    control_type: Literal["visit"] = "visit"
    text: str


class BrowserControls(BaseModel):
    input: VisitInput = Field(..., discriminator="control_type")


class BrowserOutput(BaseModel):
    output: VisitOutput | None = Field(..., discriminator="control_type")
    error: bool = False
    reason: str | None = None


class Browser(AnthropicTool[BrowserControls, BrowserOutput]):
    controls = BrowserControls
    sep = "</sep/>"

    @typing.override
    @classmethod
    def get_name(cls) -> str:
        return "Browser"

    @typing.override
    @classmethod
    def get_description(cls) -> str:
        return f"""This tool can be used to browse the web, visit sites and get their text content.
        The text content is each leaf element that has text concatenated by `{cls.sep}`.
        Make sure to always include the control_type for all inputs, this is what defines
        which exact control to use for the tool.
        """

    @typing.override
    def _use(self, control_request: BrowserControls) -> BrowserOutput:
        try:
            if isinstance(control_request.input, VisitInput):
                output = self._visit(control_request.input.url)
            return BrowserOutput(output=output)
        except Exception as e:
            return BrowserOutput(output=None, error=True, reason=str(e))

    def _visit(self, url: str) -> VisitOutput:
        from bs4 import BeautifulSoup
        import requests

        resp = requests.get(url)
        resp.content
        soup = BeautifulSoup(resp.content, features="html.parser")
        text = soup.get_text(self.sep)
        return VisitOutput(control_type="visit", text=text)
