import os

import dotenv

dotenv.load_dotenv()


class Configuration:
    ha_token: str
    openai_api_key: str

    def __init__(self):
        self.ha_token = os.getenv("TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
