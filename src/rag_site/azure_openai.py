from openai import AzureOpenAI


class AzureOpenAIEmbedder:
    def __init__(self, endpoint: str, api_key: str, embedding_model: str) -> None:
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-10-21",
        )
        self.embedding_model = embedding_model

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def get_vector_dimensions(self) -> int:
        return len(self.embed_text("hello"))


class AzureOpenAIChat:
    def __init__(self, endpoint: str, api_key: str, chat_model: str) -> None:
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-10-21",
        )
        self.chat_model = chat_model

    def generate_answer(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in gold and silver markets, news, and history."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
