"""
FAQ Agent: Answers frequently asked questions about Dexa Medica.
Implements document processing, vector store, semantic search, and LangGraph flow
with query refinement loop and memory.
"""

import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Document Processing (Phase 1) ---
FAQ_PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "FAQ Dexa Medica.pdf")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_faq")


def load_and_chunk_faq() -> list[Document]:
    """Extract PDF content and split into chunks."""
    if not os.path.exists(FAQ_PDF_PATH):
        raise FileNotFoundError(
            f"FAQ PDF not found at {FAQ_PDF_PATH}. Run: python scripts/download_faq_pdf.py"
        )
    loader = PyPDFLoader(FAQ_PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(docs)


# --- Vector Store (Phase 2) ---
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

try:
    from langchain_chroma import Chroma

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    vector_store = Chroma(
        collection_name="faq_dexa_medica",
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    # Populate if empty
    try:
        existing_count = vector_store._collection.count()
    except Exception:
        existing_count = 0
    if existing_count == 0:
        chunks = load_and_chunk_faq()
        vector_store.add_documents(chunks)
        vector_store.persist()
except ImportError:
    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embedding_model)
    chunks = load_and_chunk_faq()
    vector_store.add_documents(chunks)


# --- Semantic Search (Phase 3) ---
def semantic_search(query: str, k: int = 5) -> list[Document]:
    """Search for relevant chunks by semantic similarity."""
    return vector_store.similarity_search(query, k=k)


def format_docs_for_llm(docs: list[Document]) -> str:
    """Format retrieved documents for LLM consumption."""
    return "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
    )


# --- LLM and State ---
llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
MAX_RETRIES = 3


class AnswerCheck(BaseModel):
    contains_answer: bool = Field(description="Whether the retrieved context contains the answer")


class FAQFullState(MessagesState):
    query: str
    retrieved_docs: list
    retry_count: int
    _contains_answer: bool


# --- Graph Nodes ---
def entry_node(state: dict) -> dict:
    msgs = state.get("messages", [])
    query = msgs[-1].content if msgs else ""
    return {"query": query, "retry_count": 0}


def search_node_fn(state: dict) -> dict:
    query = state.get("query") or (state["messages"][-1].content if state.get("messages") else "")
    docs = semantic_search(query, k=5)
    return {"retrieved_docs": docs, "query": query}


def check_answer_node_fn(state: dict) -> dict:
    msgs = state.get("messages", [])
    user_question = msgs[-1].content if msgs else state.get("query", "")
    docs_content = format_docs_for_llm(state.get("retrieved_docs", []))
    model_with_structure = llm.with_structured_output(AnswerCheck)
    response = model_with_structure.invoke(
        [
            SystemMessage(
                content="""Determine if the following retrieved context contains enough information to answer the user's question.
                Answer with contains_answer=true only if the context clearly and directly addresses the question.
                Answer with contains_answer=false if the context is irrelevant, insufficient, or does not address the question."""
            ),
            HumanMessage(content=f"User question: {user_question}\n\nRetrieved context:\n{docs_content}"),
        ]
    )
    return {"_contains_answer": response.contains_answer}


def modify_query_node_fn(state: dict) -> dict:
    msgs = state.get("messages", [])
    user_question = msgs[-1].content if msgs else state.get("query", "")
    prev_query = state.get("query", user_question)
    retry_count = state.get("retry_count", 0)
    docs_content = format_docs_for_llm(state.get("retrieved_docs", []))

    response = llm.invoke(
        [
            SystemMessage(
                content="""You are a query rewriter. The user asked a question but the retrieved documents did not contain the answer.
                Rewrite or expand the search query to better find relevant information. Use synonyms, related terms, or rephrase.
                Return ONLY the new search query, nothing else. Keep it concise (1-2 sentences)."""
            ),
            HumanMessage(
                content=f"Original question: {user_question}\nPrevious search query: {prev_query}\n"
                f"What was retrieved (not helpful):\n{docs_content[:500]}..."
            ),
        ]
    )
    new_query = response.content.strip().strip('"')
    return {"query": new_query, "retry_count": retry_count + 1}


def generate_node_fn(state: dict) -> dict:
    msgs = state.get("messages", [])
    user_question = msgs[-1].content if msgs else state.get("query", "")
    docs_content = format_docs_for_llm(state.get("retrieved_docs", []))

    system_content = (
        "You are a helpful customer service assistant for Dexa Medica. "
        "Use the following retrieved FAQ context to answer the user's question. "
        "If the context does not contain the answer, say you don't have that information. "
        "Keep answers concise and customer-friendly."
        "\n\nRetrieved context:\n"
        f"{docs_content}"
    )
    response = llm.invoke(
        [SystemMessage(content=system_content), HumanMessage(content=user_question)]
    )
    return {"messages": [response]}


def route_after_check_fn(state: dict) -> Literal["generate", "modify_query"]:
    if state.get("_contains_answer", False):
        return "generate"
    return "modify_query"


def route_after_modify_fn(state: dict) -> Literal["search", "generate"]:
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "search"
    return "generate"


# --- Build Graph ---
memory = MemorySaver()

graph = (
    StateGraph(FAQFullState)
    .add_node("entry", entry_node)
    .add_node("search", search_node_fn)
    .add_node("check_answer", check_answer_node_fn)
    .add_node("modify_query", modify_query_node_fn)
    .add_node("generate", generate_node_fn)
    .add_edge(START, "entry")
    .add_edge("entry", "search")
    .add_edge("search", "check_answer")
    .add_conditional_edges(
        "check_answer",
        route_after_check_fn,
        {"generate": "generate", "modify_query": "modify_query"},
    )
    .add_conditional_edges(
        "modify_query",
        route_after_modify_fn,
        {"search": "search", "generate": "generate"},
    )
    .add_edge("generate", END)
    .compile(name="FAQ", checkpointer=memory)
)
