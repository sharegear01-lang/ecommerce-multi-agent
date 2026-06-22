"""
FastAPI Application — entry point for the E-commerce Multi-Agent System.

Provides:
- POST /chat         — send message, get agent response + state trace
- POST /chat/stream  — SSE streaming with per-agent progress
- GET  /session/{id} — retrieve session state
- DELETE /session/{id} — clear session
- GET  /health       — health check
- Static files at /  — chat UI

Startup: initializes RAG vector store, builds StateGraph.
"""

import json
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from config import settings, Settings
from src.rag.store import VectorStore
from src.rag.retriever import Retriever
from src.agents.router_agent import RouterAgent
from src.agents.shopping_guide import ShoppingGuideAgent
from src.agents.order_agent import OrderAgent
from src.agents.aftersales_agent import AftersalesAgent
from src.conflict.resolver import ConflictResolver
from src.state.graph import build_graph
from src.state.schema import get_initial_state

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mas")

# ---------------------------------------------------------------------------
# Global singletons (initialized at startup)
# ---------------------------------------------------------------------------
graph_app = None
router_agent: Optional[RouterAgent] = None
shopping_guide: Optional[ShoppingGuideAgent] = None
order_agent: Optional[OrderAgent] = None
aftersales_agent: Optional[AftersalesAgent] = None
retriever: Optional[Retriever] = None

# Store session thread configs: session_id → {"configurable": {"thread_id": str}}
sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversation")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    current_state: str
    intent: str
    agent_trace: list[str] = Field(default_factory=list)


class StateResponse(BaseModel):
    session_id: str
    current_state: str
    intent: str
    context: dict
    task_chain: list
    message_count: int


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize RAG store & build StateGraph. Shutdown: cleanup."""
    global graph_app, router_agent, shopping_guide, order_agent, aftersales_agent, retriever

    logger.info("=" * 60)
    logger.info("Starting E-commerce Multi-Agent System...")
    Settings.print_config()

    # Validate config
    errors = Settings.validate()
    if errors:
        for e in errors:
            logger.error(e)
        logger.error("Cannot start — missing required configuration.")

    # --- Enable LangSmith tracing (must happen BEFORE any LangChain imports) ---
    Settings.setup_langsmith()

    # Initialize LLM (DeepSeek via OpenAI-compatible API)
    logger.info("Initializing LLM (DeepSeek v4 Flash)...")
    llm = ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1024,
    )

    # Initialize RAG
    logger.info("Initializing RAG vector store...")
    vector_store_obj = VectorStore()
    chroma_collection = vector_store_obj.collection
    retriever = Retriever(chroma_collection)

    # Initialize conflict resolver
    conflict_resolver = ConflictResolver(intent_priority_weight=0.6)

    # Initialize agents
    logger.info("Initializing agents...")
    router_agent = RouterAgent(llm, retriever, conflict_resolver=conflict_resolver)
    shopping_guide = ShoppingGuideAgent(llm, retriever)
    order_agent = OrderAgent(llm, retriever)
    aftersales_agent = AftersalesAgent(llm, retriever)

    # Build and compile the StateGraph
    logger.info("Building LangGraph StateGraph...")
    graph_app = build_graph(
        router=router_agent,
        shopping_guide=shopping_guide,
        order_agent=order_agent,
        aftersales_agent=aftersales_agent,
    )

    logger.info("✅ System ready! Visit http://%s:%s", settings.HOST, settings.PORT)
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="E-commerce Multi-Agent System",
    description="LangChain + LangGraph based multi-agent collaboration system for e-commerce",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "rag_ready": retriever is not None,
        "graph_ready": graph_app is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send a message to the multi-agent system.

    Returns the agent's response along with state transition information.
    """
    global graph_app

    if graph_app is None:
        raise HTTPException(status_code=503, detail="System is not initialized yet")

    # Get or create session
    session_id = req.session_id or str(uuid.uuid4())[:8]
    thread_config = _get_or_create_thread(session_id)

    # Invoke the graph
    initial_state = get_initial_state(user_id="web_user", session_id=session_id)
    initial_state["messages"] = [HumanMessage(content=req.message)]

    try:
        # Stream to capture all agent intermediate steps
        agent_trace = []
        final_state = None

        for event in graph_app.stream(
            initial_state,
            config={
                **thread_config,
                "metadata": {"langsmith_project": settings.LANGCHAIN_PROJECT},
                "run_name": f"chat-{session_id}",
            },
            stream_mode="values",
        ):
            final_state = event
            # Track state changes for trace
            node_state = event.get("current_state", "?")
            node_intent = event.get("intent", "")
            if node_state not in agent_trace:
                agent_trace.append(node_state)

        if final_state is None:
            raise RuntimeError("Graph did not produce output")

        # Extract the last AI message as response
        messages = final_state.get("messages", [])
        response_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                response_text = msg.content
                break

        if not response_text:
            response_text = "系统已处理您的请求。"

        return ChatResponse(
            session_id=session_id,
            response=response_text,
            current_state=final_state.get("current_state", "IDLE"),
            intent=final_state.get("intent", ""),
            agent_trace=agent_trace,
        )

    except Exception as e:
        logger.error("Chat error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint — sends events as each agent processes the message.
    """
    global graph_app

    if graph_app is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    session_id = req.session_id or str(uuid.uuid4())[:8]
    thread_config = _get_or_create_thread(session_id)

    initial_state = get_initial_state(user_id="web_user", session_id=session_id)
    initial_state["messages"] = [HumanMessage(content=req.message)]

    async def event_generator():
        try:
            agent_trace = []
            final_state = None

            for event in graph_app.stream(
                initial_state,
                config={
                    **thread_config,
                    "metadata": {"langsmith_project": settings.LANGCHAIN_PROJECT},
                    "run_name": f"chat-stream-{session_id}",
                },
                stream_mode="values",
            ):
                final_state = event
                current = event.get("current_state", "?")
                intent = event.get("intent", "")

                if current not in agent_trace:
                    agent_trace.append(current)
                    yield f"data: {json.dumps({'type': 'state', 'state': current, 'intent': intent, 'trace': agent_trace}, ensure_ascii=False)}\n\n"

            # Extract final response
            if final_state:
                messages = final_state.get("messages", [])
                response_text = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        response_text = msg.content
                        break

                yield f"data: {json.dumps({'type': 'response', 'text': response_text or '系统已处理您的请求。', 'session_id': session_id, 'current_state': final_state.get('current_state', 'IDLE'), 'intent': final_state.get('intent', ''), 'trace': agent_trace}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error("Stream error: %s", str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/session/{session_id}", response_model=StateResponse)
async def get_session(session_id: str):
    """Retrieve the current state of a session."""
    thread_config = sessions.get(session_id)
    if not thread_config:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get current state from graph
    if graph_app:
        current_state = graph_app.get_state(thread_config)
        if current_state and current_state.values:
            s = current_state.values
            return StateResponse(
                session_id=session_id,
                current_state=s.get("current_state", "IDLE"),
                intent=s.get("intent", ""),
                context=s.get("context", {}),
                task_chain=s.get("task_chain", []),
                message_count=len(s.get("messages", [])),
            )

    return StateResponse(
        session_id=session_id,
        current_state="IDLE",
        intent="",
        context={},
        task_chain=[],
        message_count=0,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear/reset a session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/products")
async def list_products():
    """List all products in the knowledge base (for debugging)."""
    import json as _json
    data_file = settings.DATA_DIR / "products.json"
    if data_file.exists():
        with open(data_file, "r", encoding="utf-8") as f:
            return _json.load(f)
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_or_create_thread(session_id: str) -> dict:
    """Get or create a LangGraph thread config for a session."""
    if session_id not in sessions:
        sessions[session_id] = {"configurable": {"thread_id": session_id}}
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
