from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery


class AzureSearchStore:
    def __init__(self, endpoint: str, key: str, index_name: str) -> None:
        credential = AzureKeyCredential(key)
        self.index_name = index_name
        self.index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential,
        )
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
        )

    def create_index_if_needed(self, vector_dimensions: int) -> None:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="url", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="vp",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="vp", algorithm_configuration_name="hnsw")],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self.index_client.create_or_update_index(index)

    def upload_documents(self, docs: list[dict], batch_size: int = 100) -> int:
        uploaded = 0

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            results = self.search_client.upload_documents(batch)
            uploaded += len(results)

        return uploaded

    def hybrid_search(
        self,
        question: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        vector_query = VectorizedQuery(
            vector=query_vector,
            fields="content_vector",
            k_nearest_neighbors=top_k,
        )

        results = self.search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            top=top_k,
            select=["title", "url", "content", "chunk_index", "domain"],
        )

        return list(results)
