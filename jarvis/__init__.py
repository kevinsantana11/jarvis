import logging

from jarvis.config import Configuration
from jarvis.gadgets.clients import HAAPIClient
from jarvis.modules import PrintOutModule, TextInputModule, ReasoningModule

logging.basicConfig(level=logging.INFO)
cfg = Configuration()
ha_api_client = HAAPIClient("http", "localhost:8123", cfg.ha_token)

print_out = PrintOutModule()
reasoning = ReasoningModule(ha_api_client, cfg.openai_api_key)
text_input = TextInputModule()


def run():
    print_out(reasoning(text_input()))
