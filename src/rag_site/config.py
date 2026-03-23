import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    firecrawl_api_key: str
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_chat_model: str
    azure_openai_embedding_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            firecrawl_api_key=get_env("FIRECRAWL_API_KEY"),
            azure_search_endpoint=get_env("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=get_env("AZURE_SEARCH_KEY"),
            azure_search_index=get_env("AZURE_SEARCH_INDEX"),
            azure_openai_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=get_env("AZURE_OPENAI_API_KEY"),
            azure_openai_chat_model=get_env("AZURE_OPENAI_CHAT_MODEL"),
            azure_openai_embedding_model=get_env("AZURE_OPENAI_EMBEDDING_MODEL"),
        )
