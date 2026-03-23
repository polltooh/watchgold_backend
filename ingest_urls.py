from rag_site.azure_openai import AzureOpenAIEmbedder
from rag_site.azure_search import AzureSearchStore
from rag_site.config import Settings
from rag_site.firecrawl_client import FirecrawlCrawler
from rag_site.ingest import build_documents_from_pages


def main() -> None:
    settings = Settings.from_env()

    crawler = FirecrawlCrawler(api_key=settings.firecrawl_api_key)
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

    urls_to_index = [
        # "https://www.gold.org/",
        # "https://silverinstitute.org/",
        # "https://gold.tether.to/",
        "https://www.metalsfocus.com/",
        "https://www.perthmint.com/",
        # "https://www.lbma.org.uk/"
    ]

    all_docs: list[dict] = []

    for url in urls_to_index:
        print(f"Crawling: {url}")
        pages = crawler.crawl_site(url, limit=100)
        print(f"  pages found: {len(pages)}")

        docs = build_documents_from_pages(pages=pages, embedder=embedder)
        print(f"  chunks built: {len(docs)}")
        all_docs.extend(docs)

    if not all_docs:
        print("No documents to upload.")
        return

    uploaded = search_store.upload_documents(all_docs)
    print(f"Uploaded chunks: {uploaded}")


if __name__ == "__main__":
    main()
