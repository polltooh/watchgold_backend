import json
from dotenv import load_dotenv

load_dotenv()
import os
if "OPENAI_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import sys

sys.path.append(str(Path(__file__).parent.parent))
from rag_site.config import Settings
from rag_site.azure_openai import AzureOpenAIEmbedder, AzureOpenAIChat
from rag_site.azure_search import AzureSearchStore

from fastapi.middleware.cors import CORSMiddleware

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

class QuestionRequest(BaseModel):
    question: str

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
            f"Use the following pieces of retrieved context to answer the question. "
            f"If you don't know the answer, say that you don't know based on the provided context. "
            f"Answer the question directly without starting with phrases like 'Based on the provided context,'. "
            f"Provide a detailed answer with an inline citation for each sentence. "
            f"To create the citations, you MUST use the exact format <ref=X> where X is the source number (e.g. <ref=1> or <ref=1,2>). Do not write out any URLs in your response.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{request.question}"
        )

        # 4. Generate answer
        answer = chat_model.generate_answer(prompt)
        
        # Suffix the answer with the URLs as requested
        formatted_answer = answer
        if sources:
            formatted_answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
            
        return ChatResponse(answer=formatted_answer, sources=sources)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

from chatgold.feeds.rss import get_daily_summary

@app.get("/feeds")
async def get_feeds():
    try:
        feeds = get_daily_summary()
        return {"feeds": feeds}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
