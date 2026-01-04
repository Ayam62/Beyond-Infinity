# ------------------------------
# Imports
# ------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pinecone_text.sparse import BM25Encoder
from retriever.retriever import retriever 
from typing import List, Dict, TypedDict, Tuple, Annotated, Optional
from pydantic import BaseModel, Field
import operator

# ------------------------------
# Configuration Constants
# ------------------------------
DEFAULT_N_RETRIEVAL = 5  # Number of chunks to retrieve for evaluation
DEFAULT_N_GENERATION = 3  # Number of chunks to use for generation
DEFAULT_ALPHA = 0.5  # Hybrid search balance

# ------------------------------
# Structured Output Schema
# ------------------------------

class StructuredResponse(BaseModel):
    """Schema for structured LLM response with answer and sources."""
    answer: str = Field(description="The formatted answer to the user's question in Markdown")
    source_indices: List[int] = Field(description="List of context indices (1-based) used in the answer")
    target_office: Optional[str] = Field(description="The specific office name identified (e.g., 'DAO', 'Ward Office', 'Municipality')")

# ------------------------------
# Global Type Definitions for LangGraph State
# ------------------------------

class ChatState(TypedDict):
    """
    Represents the state of the RAG conversation in LangGraph.
    """
    query: str
    context: List[Dict]
    chat_history: Annotated[List[BaseMessage], operator.add]
    response_language: str
    answer: str
    sources: List[Dict]
    error: str
    is_location_query: bool 
    target_office: Optional[str]
    retrieved_chunk_ids: List[str]
    reranked_chunk_ids: List[str]
    all_retrieved_chunks: List[Tuple]
    reranked_chunks: List[Tuple]

# ------------------------------
# Global Memory & LangGraph Instance
# ------------------------------

_chat_message_history: List[BaseMessage] = []
_rag_app = None

def _get_current_history() -> List[BaseMessage]:
    """Returns the current chat history (last 6 exchanges/12 messages)."""
    return _chat_message_history[-12:] 

def _update_global_history(new_messages: List[BaseMessage]):
    """Updates the global chat history with new messages."""
    global _chat_message_history
    _chat_message_history.extend(new_messages)

# ------------------------------
# Helper Functions
# ------------------------------

def _format_input(query: str, context_list: List[Dict], chat_history: List[BaseMessage], response_language: str) -> str:
    """Format the prompt for LLM with context and chat history."""
    
    history_str = ""
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    
    history_section = f"\n**Previous Conversation:**\n{history_str.strip()}\n---\n" if history_str.strip() else ""
    
    language_instruction = (
        "**CRITICAL: You MUST respond in English.**\n- Translate context from Nepali to clear, professional English."
        if response_language == "english"
        else "**CRITICAL: You MUST respond in Nepali.**\n- Answer in clear, natural Nepali language."
    )
    
    contexts_formatted = ""
    for idx, ctx_obj in enumerate(context_list, 1):
        contexts_formatted += f"\n**Context {idx}:**\n{ctx_obj['context']}\n"

    return f"""You are a helpful AI assistant for Nepali government document processes (citizenship, passport etc.).

{history_section}

{language_instruction}

**Answer Rules:**
1. Answer ONLY using the provided context below
2. Use chat history only if directly relevant to the current question
3. Make logical inferences only when clearly supported by the context
4. If information is not available in the context:
   - Nepali: "यस बारेमा मलाई जानकारी छैन।"
   - English: "Sorry, I don't have information about that."
5. Never mention context IDs, numbers, or metadata in your answer
6. Keep the answer complete and relevant to the question only.
7. Keep the answer short, clear, and concise. Avoid unnecessary details.

**CRITICAL MARKDOWN FORMATTING RULES:**

**MOST IMPORTANT: BLANK LINES ARE MANDATORY**

React-markdown REQUIRES blank lines to render properly. You MUST:
1. Add a blank line BEFORE every heading
2. Add a blank line AFTER every heading
3. Add a blank line BEFORE every list
4. Add a blank line AFTER every list
5. Add blank lines between paragraphs

**Markdown Rules:**
- `##` for main headings (MUST have space after #)
- `###` for subheadings (MUST have space after #)
- `-` for bullet points (NEVER use *)
- `**text**` for bold emphasis
- Each list item on its own line
- ALWAYS put blank lines around headings and lists

**DO NOT start your response with a large title heading (##). Start directly with the answer content.**

**Context from Documents (in Nepali):**
{contexts_formatted}

**Current Question:** {query}

**Your Response Must Include:**
- `answer`: Properly formatted Markdown in {'English' if response_language == 'english' else 'Nepali'} (DO NOT include a title heading at the start)
- `source_indices`: Array of context numbers used (e.g., [1, 2])
"""
# ------------------------------
# LangGraph Nodes
# ------------------------------

def _retrieve_node(state: ChatState, index, dense_embeddings, bm25_encoder: BM25Encoder, 
                   alpha: float, n_retrieval: int, n_generation: int) -> Dict:
    """
    LangGraph node: Finds context based on the query and chat history.
    Now captures location-based intent.
    """
    try:
        query = state["query"]
        chat_history = state["chat_history"]
        
        history_str = ""
        for msg in chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_str += f"{role}: {msg.content}\n"

        # Retrieve relevant context
        # result now contains processed_info with our new location flags
        result = retriever(
            index, query, dense_embeddings, bm25_encoder,
            chat_history=history_str, 
            n_retrieval=n_retrieval,  
            n_generation=n_generation,  
            alpha=alpha,
            use_reranker=True
        )

        # Handle different return types
        if isinstance(result, str) and result.startswith("Error"):
            return {
                "error": result, 
                "context": [], 
                "is_location_query": False,
                "target_office": None,
                "retrieved_chunk_ids": [],
                "reranked_chunk_ids": [],
                "all_retrieved_chunks": [],
                "reranked_chunks": []
            }
        
        if isinstance(result, tuple) and len(result) == 4:
            context_list, processed_info, all_original_chunks, reranked_chunks = result
        else:
            return {
                "error": "Unexpected return format from retriever",
                "context": [],
                "is_location_query": False,
                "target_office": None,
                "retrieved_chunk_ids": [],
                "reranked_chunk_ids": [],
                "all_retrieved_chunks": [],
                "reranked_chunks": []
            }
        
        # Determine language and location intent from processed_info
        response_language = processed_info.get("language", "nepali") if processed_info else "nepali"
        
        # --- NEW LOCATION LOGIC ---
        is_location_query = processed_info.get("is_location_query", False)
        target_office = processed_info.get("target_office", None)
        
        # Extract chunk IDs for evaluation
        all_chunk_ids = [chunk[3] for chunk in all_original_chunks if len(chunk) > 3]
        reranked_chunk_ids = [chunk[3] for chunk in reranked_chunks if len(chunk) > 3]
            
        return {
            "context": context_list,  
            "response_language": response_language, 
            "is_location_query": is_location_query,  # Stored in state for 'generate' node
            "target_office": target_office,          # Stored in state for 'generate' node
            "error": "",
            "retrieved_chunk_ids": all_chunk_ids,
            "reranked_chunk_ids": reranked_chunk_ids,
            "all_retrieved_chunks": all_original_chunks,
            "reranked_chunks": reranked_chunks
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f" Error during retrieval: {e}", 
            "context": [], 
            "is_location_query": False,
            "target_office": None,
            "retrieved_chunk_ids": [],
            "reranked_chunk_ids": [],
            "all_retrieved_chunks": []
        }

def _generate_node(state: ChatState) -> Dict:
    """
    LangGraph node: Generates the final response.
    Checks for location intent, handles tokens, and strips them for the UI.
    """
    try:
        query = state["query"]
        context_list = state["context"]
        chat_history = state["chat_history"]
        response_language = state["response_language"]
        
        # 1. Check the location flag from the Retriever
        is_location = state.get("is_location_query", False)

        if state.get("error"):
            return {"answer": state["error"], "sources": [], "chat_history": []}

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.5,
        ).with_structured_output(StructuredResponse)

        # 2. Modify Prompt based on location flag
        full_prompt = _format_input(query, context_list, chat_history, response_language)
        
        if is_location:
            location_instruction = """
            \n**CRITICAL INSTRUCTION FOR LOCATION QUERY:**
            - Identify the office name (e.g., 'District Administration Office', 'Ward Office') and put it ONLY in the `target_office` field.
            - **DO NOT** include the office name at the beginning of the `answer` text.
            - The `answer` should start directly with the descriptive information/procedure.
            """
            full_prompt += location_instruction

        # 3. Invoke LLM
        structured_response: StructuredResponse = llm.invoke(full_prompt)
        
        raw_answer = structured_response.answer.strip()
        office_name = structured_response.target_office
        source_indices = structured_response.source_indices

        if office_name:
            is_location = True
            # FIX: Initialize final_answer here so it's always associated with a value
            final_answer = raw_answer 
            
            if raw_answer.startswith(office_name):
                final_answer = raw_answer[len(office_name):].lstrip(" ,:-")
            
            # Now final_answer is guaranteed to exist
            final_answer = final_answer.replace("[[START_LOCATION]]", "").strip()

        else:
            final_answer = raw_answer


        # 5. Build sources list
        sources = []
        for idx in source_indices:
            if 1 <= idx <= len(context_list):
                ctx_obj = context_list[idx - 1]
                sources.append({
                    "source_link": ctx_obj.get("source_link", ""),
                    "source_type": ctx_obj.get("source_type", "")
                })

        # Remove duplicate sources
        unique_sources = []
        seen = set()
        for src in sources:
            src_tuple = (src.get("source_link", ""), src.get("source_type", ""))
            if src_tuple not in seen:
                seen.add(src_tuple)
                unique_sources.append(src)
                
        new_history_messages = [HumanMessage(content=query), AIMessage(content=final_answer)]
        
        return {
            "answer": final_answer,
            "sources": unique_sources,
            "error": "",
            "is_location_query": is_location, # Pass it back to state
            "target_office": office_name,
            "chat_history": new_history_messages 
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f" Error generating response: {e}", 
            "sources": [], 
            "error": f"Generation Error: {e}", 
            "chat_history": []
        }
# ------------------------------
# LangGraph Builder & Entry Point
# ------------------------------

def _build_rag_graph(index, dense_embeddings, bm25_encoder: BM25Encoder, 
                     alpha: float, n_retrieval: int, n_generation: int):
    """Initializes and returns the compiled LangGraph."""
    workflow = StateGraph(ChatState)

    workflow.add_node(
        "retrieve", 
        lambda state: _retrieve_node(
            state, index, dense_embeddings, bm25_encoder, 
            alpha, n_retrieval, n_generation
        )
    )
    workflow.add_node("generate", _generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

def rag_chat(index, query: str, dense_embeddings, bm25_encoder: BM25Encoder,
             alpha: float = DEFAULT_ALPHA,
             n_retrieval: int = DEFAULT_N_RETRIEVAL,
             n_generation: int = DEFAULT_N_GENERATION) -> Dict[str, any]:
    """
    Main entry point for RAG chat.
    
    Args:
        index: Pinecone index
        query: User query
        dense_embeddings: Dense embedding model
        bm25_encoder: BM25 encoder
        alpha: Hybrid search balance (0=sparse, 1=dense)
        n_retrieval: Number of chunks to retrieve (for evaluation metrics)
        n_generation: Number of chunks to use for answer generation
        
    Returns:
        Dict with:
        - 'answer': Generated answer
        - 'sources': Source documents used
        - 'retrieved_chunk_ids': Original retrieval order (for evaluation)
        - 'reranked_chunk_ids': Reranked order used for generation (for evaluation)
        - 'all_retrieved_chunks': Original chunks from retrieval
        - 'reranked_chunks': Reranked chunks used for generation
    """
    global _rag_app
    
    # Rebuild graph if parameters changed or first time
    if _rag_app is None:
        try:
            _rag_app = _build_rag_graph(
                index, dense_embeddings, bm25_encoder, 
                alpha, n_retrieval, n_generation
            )
        except Exception as e:
            return {
                "answer": f" Error initializing RAG app: {e}", 
                "sources": [], 
                "retrieved_chunk_ids": [],
                "reranked_chunk_ids": [],
                "all_retrieved_chunks": [],
                "reranked_chunks": []
            }

    initial_state = ChatState(
        query=query,
        context=[],
        chat_history=_get_current_history(),
        response_language="nepali",
        answer="",
        sources=[],
        error="",
        retrieved_chunk_ids=[],
        reranked_chunk_ids=[],
        all_retrieved_chunks=[],
        reranked_chunks=[]
    )

    try:
        final_state = _rag_app.invoke(initial_state)

        answer = final_state.get("answer", "An unknown error occurred.")
        sources = final_state.get("sources", [])
        retrieved_chunk_ids = final_state.get("retrieved_chunk_ids", [])
        reranked_chunk_ids = final_state.get("reranked_chunk_ids", [])
        all_chunks = final_state.get("all_retrieved_chunks", [])
        reranked_chunks = final_state.get("reranked_chunks", [])
        is_location_query = final_state.get("is_location_query", False)
        target_office = final_state.get("target_office", None)
        
        new_messages_from_state = final_state.get("chat_history", [])
        
        if len(new_messages_from_state) == len(initial_state["chat_history"]) + 2:
             _update_global_history(new_messages_from_state[-2:]) 

        return {
            "answer": answer, 
            "sources": sources,
            "is_location_query": is_location_query, # Now returned to FastAPI
            "target_office": target_office,
            "retrieved_chunk_ids": retrieved_chunk_ids,  # Original retrieval order
            "reranked_chunk_ids": reranked_chunk_ids,  # Reranked order (used for generation)
            "all_retrieved_chunks": all_chunks,  # Original chunks
            "reranked_chunks": reranked_chunks,  # Reranked chunks
            
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f" Error running RAG graph: {e}", 
            "sources": [], 
            "retrieved_chunk_ids": [],
            "reranked_chunk_ids": [],
            "all_retrieved_chunks": [],
            "reranked_chunks": []
        }

# ------------------------------
# Clear chat memory
# ------------------------------
def clear_chat_history():
    """Clears the global chat message history."""
    global _chat_message_history, _rag_app
    _chat_message_history = []
    _rag_app = None  # Reset graph to pick up new parameters
    print(" Chat history cleared!")