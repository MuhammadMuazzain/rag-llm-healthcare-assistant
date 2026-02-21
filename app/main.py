"""
FastAPI application for the Healthcare RAG + Voice Agent system.
Exposes endpoints for:
  - Vapi.ai webhook handling (voice calls)
  - Direct RAG query API (testing/integration)
  - Clinical content management
  - System health monitoring
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag.retriever import ClinicalRAGRetriever
from app.voice.vapi_client import VapiVoiceClient
from app.llm_chain import ClinicalLLMChain
from config import get_settings

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

retriever: ClinicalRAGRetriever | None = None
vapi_client: VapiVoiceClient | None = None
llm_chain: ClinicalLLMChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, vapi_client, llm_chain

    logger.info("starting_application")

    retriever = ClinicalRAGRetriever()
    chunk_count = await retriever.load_clinical_content()
    logger.info("clinical_content_indexed", chunks=chunk_count)

    llm_chain = ClinicalLLMChain(retriever)

    vapi_client = VapiVoiceClient()
    vapi_client.set_response_callback(llm_chain.generate_response)

    logger.info("application_ready")
    yield

    if vapi_client:
        await vapi_client.close()
    logger.info("application_shutdown")


app = FastAPI(
    title="Healthcare RAG + Voice Agent",
    description=(
        "HIPAA-compliant healthcare application using RAG for clinical content "
        "delivery and Vapi.ai for AI voice agent integration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Models ---

class QueryRequest(BaseModel):
    query: str
    category_hint: str | None = None
    tag_hints: list[str] | None = None


class QueryResponse(BaseModel):
    response: str
    sources: list[dict]
    intent_category: str | None


class CallRequest(BaseModel):
    phone_number: str | None = None
    customer_name: str | None = None
    template_id: str | None = None
    metadata: dict | None = None


class HealthResponse(BaseModel):
    status: str
    vector_store: dict
    voice_client: dict


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not retriever or not vapi_client:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = await retriever.vector_store.get_collection_stats()

    return HealthResponse(
        status="healthy",
        vector_store=stats,
        voice_client=vapi_client.get_conversation_state(),
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_clinical_content(request: QueryRequest):
    """Query the RAG system for clinical content."""
    if not retriever or not llm_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    results = await retriever.retrieve(request.query)
    response = await llm_chain.generate_response(request.query)
    category, _ = retriever.classify_intent(request.query)

    sources = [
        {
            "title": r.metadata.get("title", ""),
            "section": r.metadata.get("section", ""),
            "score": round(r.score, 3),
        }
        for r in results
    ]

    return QueryResponse(
        response=response,
        sources=sources,
        intent_category=category,
    )


@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request):
    """Handle Vapi.ai webhook events."""
    if not vapi_client:
        raise HTTPException(status_code=503, detail="Voice client not initialized")

    body = await request.json()
    logger.info("webhook_received", event_type=body.get("message", {}).get("type"))

    result = await vapi_client.handle_webhook_event(body)

    if result:
        return result
    return {"status": "ok"}


@app.post("/api/calls")
async def create_call(request: CallRequest, background_tasks: BackgroundTasks):
    """Initiate an outbound voice call via Vapi."""
    if not vapi_client:
        raise HTTPException(status_code=503, detail="Voice client not initialized")

    if not llm_chain:
        raise HTTPException(status_code=503, detail="LLM chain not initialized")

    llm_chain.clear_history()

    call_data = await vapi_client.create_call(
        phone_number=request.phone_number,
        customer_name=request.customer_name,
        metadata=request.metadata,
    )

    return {
        "call_id": call_data.get("id"),
        "status": call_data.get("status"),
        "message": "Call initiated successfully",
    }


@app.post("/api/index/reload")
async def reload_clinical_content():
    """Re-index clinical content from source files."""
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    await retriever.vector_store.clear()
    count = await retriever.load_clinical_content()

    return {"status": "reindexed", "chunks_indexed": count}


@app.get("/api/conversation/state")
async def get_conversation_state():
    """Get current voice conversation state for debugging."""
    if not vapi_client:
        raise HTTPException(status_code=503, detail="Voice client not initialized")

    return vapi_client.get_conversation_state()
