from rag_site.azure_openai import AzureOpenAIEmbedder
from rag_site.azure_search import AzureSearchStore
from rag_site.config import Settings


def main() -> None:
    settings = Settings.from_env()

    embedder = AzureOpenAIEmbedder(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        embedding_model=settings.azure_openai_embedding_model,
    )

    search_store = AzureSearchStore(
        endpoint=settings.azure_search_endpoint,
        key=settings.azure_search_key,
        index_name=settings.azure_search_index,
    )

    vector_dimensions = embedder.get_vector_dimensions()
    search_store.create_index_if_needed(vector_dimensions=vector_dimensions)

    print(f"Index ready: {settings.azure_search_index}")


if __name__ == "__main__":
    main()
