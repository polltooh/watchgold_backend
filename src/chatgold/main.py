from chatgold.feeds.rss import get_daily_summary
from fastapi.middleware.cors import CORSMiddleware
from rag_site.azure_search import AzureSearchStore
from rag_site.azure_openai import AzureOpenAIEmbedder, AzureOpenAIChat
from rag_site.config import Settings
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
import json
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]

sys.path.append(str(Path(__file__).parent.parent))


app = FastAPI(title="Gold & Silver RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize RAG components at startup ---

embedder = None
search_store = None
chat_model = None


@app.on_event("startup")
async def startup_event():
    global embedder, search_store, chat_model
    try:
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
        chat_model = AzureOpenAIChat(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            chat_model=settings.azure_openai_chat_model,
        )
        print("Successfully initialized Azure RAG components.")
    except Exception as e:
        print(f"Failed to initialize Azure components: {e}")

# --- API Endpoints ---


class ChatMessageInput(BaseModel):
    role: str
    content: str


class QuestionRequest(BaseModel):
    question: str
    history: list[ChatMessageInput] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    if not search_store or not chat_model or not embedder:
        raise HTTPException(
            status_code=503,
            detail="RAG components not initialized. Ensure Azure credentials are set."
        )

    try:
        # 1. Embed question
        qvec = embedder.embed_text(request.question)

        # 2. Hybrid search docs
        results = search_store.hybrid_search(
            question=request.question, query_vector=qvec, top_k=10
        )

        # Extract context and sources
        context_texts = []
        sources = []
        for item in results:
            content = item.get("content", "")
            url = item.get("url", "Unknown")
            if url not in sources:
                sources.append(url)

            source_idx = sources.index(url) + 1
            context_texts.append(f"Source [{source_idx}]:\n{content}")

        context_str = "\n\n".join(context_texts)

        # 3. Construct prompt
        prompt = (
            f"You have been provided with retrieved context and the conversation history. "
            f"If the retrieved context is relevant, use it to answer the question and provide inline citations (e.g., <ref=1> or <ref=1,2>). "
            f"If the retrieved context is irrelevant or unhelpful, ignore it and rely entirely on the conversation history and your own extensive knowledge to answer the question. Do not state that the context lacks the answer. "
            f"Answer the question directly without starting with phrases like 'Based on the provided context,'. "
            f"Do not write out any URLs in your response.\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Question:\n{request.question}"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in gold and silver markets, news, and history."}]
        for msg in request.history:
            messages.append({"role": "user" if msg.role ==
                            "user" else "assistant", "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        # 4. Generate answer
        answer = chat_model.generate_chat_answer(messages)

        # Suffix the answer with the URLs as requested
        formatted_answer = answer
        if sources:
            formatted_answer += "\n\nSources:\n" + \
                "\n".join(f"- {s}" for s in sources)

        return ChatResponse(answer=formatted_answer, sources=sources)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feeds")
async def get_feeds():
    try:
        feeds = get_daily_summary()
        return {"feeds": feeds}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
