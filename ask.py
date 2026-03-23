from rag_site.azure_openai import AzureOpenAIEmbedder
from rag_site.azure_search import AzureSearchStore
from rag_site.config import Settings


def print_results(results: list[dict]) -> None:
    for i, item in enumerate(results, start=1):
        print("=" * 100)
        print(f"[{i}] {item.get('title', '')}")
        print(f"URL: {item.get('url', '')}")
        print(f"Chunk: {item.get('chunk_index', '')}")
        content = (item.get("content") or "").replace("\n", " ")
        print(content[:500])


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

    question = "How much gold is mined each year?"
    qvec = embedder.embed_text(question)
    results = search_store.hybrid_search(question=question, query_vector=qvec, top_k=5)

    print_results(results)


if __name__ == "__main__":
    main()
