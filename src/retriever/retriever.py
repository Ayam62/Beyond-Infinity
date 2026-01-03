from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser  # Modern replacement
from pydantic import BaseModel, Field                       # Standard for data validation
from typing import List, Dict, Tuple, Optional
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import json
import re
import os
from embeddings.utils import preprocess_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize reranker model globally
reranker_model = None
reranker_tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def initialize_reranker():
    global reranker_model, reranker_tokenizer
    if reranker_model is None:
        print("Loading reranker model...")
        reranker_tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-reranker-v2-base-multilingual')
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            dtype="auto", # Changed from torch_dtype to dtype
            trust_remote_code=True,
        )
        reranker_model.to(device)
        reranker_model.eval()
        print(f"Reranker model loaded successfully")

def rerank_chunks(query: str, chunks: List[Tuple], top_k: int = None) -> Tuple[List[Tuple], List[str]]:
    """
    Rerank chunks using BGE reranker model.
    
    Returns:
        Tuple of (reranked_chunks, reranked_chunk_ids)
    """
    global reranker_model, reranker_tokenizer
    
    if not chunks:
        return chunks, []
    
    if reranker_model is None:
        initialize_reranker()
    
    print(f"\nReranking {len(chunks)} chunks...")
    print(f"Original order (chunk IDs): {[chunk[3][:15] for chunk in chunks[:5]]}...")
    
    pairs = [[query, chunk[0]] for chunk in chunks]

    with torch.no_grad():
        inputs = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1,).float()
    
    # Create list of (chunk, score, original_index) tuples
    reranked_chunks = []
    for i, chunk in enumerate(chunks):
        text, original_score, metadata, chunk_id = chunk
        reranker_score = scores[i].item()
        reranked_chunks.append((text, reranker_score, metadata, chunk_id, i))
    
    # Sort by reranker score (descending)
    reranked_chunks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Reranking complete!")
    print(f"Top 3 scores: {[f'{chunk[1]:.4f}' for chunk in reranked_chunks[:3]]}")
    
    # Remove the original index from final output
    final_chunks = [(chunk[0], chunk[1], chunk[2], chunk[3]) for chunk in reranked_chunks]
    
    # Apply top_k if specified
    if top_k:
        final_chunks = final_chunks[:top_k]
    
    # Extract reranked chunk IDs in the new order
    reranked_ids = [chunk[3] for chunk in final_chunks]
    
    print(f"Reranked order (chunk IDs): {[id[:15] for id in reranked_ids[:5]]}...")
    
    return final_chunks, reranked_ids

def LLM(prompt: PromptTemplate, temperature: float):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    chain = prompt | llm
    return chain

def to_nepali_numeral(number: str) -> str:
    eng_to_nep = str.maketrans("0123456789", "०१२३४५६७८९")
    return str(number).translate(eng_to_nep)


class QueryAnalysis(BaseModel):
    original_query: str = Field(description="The original user query")
    language: str = Field(description="The language of the query (english/nepali)")
    translated_query: str = Field(description="Nepali translation if English, else empty")
    rewritten_query: str = Field(description="Standalone, context-aware Nepali query")
    document_type: str = Field(description="Either 'passport' or 'citizenship'")
    tag: str = Field(description="The specific category tag")



def process_query_unified(query: str, chat_history: str = "") -> Dict[str, str]:
    """
    Unified LLM call that handles:
    1. Language detection and translation (English -> Nepali)
    2. Query rewriting using chat history
    3. Query categorization (document_type and tag)
    """
    
    # 2. Initialize the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
    
    # This generates the same instructions the prompt expects
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    
    history_section = ""
    if chat_history and chat_history.strip():
        history_section = f"""
**Previous Conversation History:**
{chat_history}

---
"""
    
    # --- PROMPT TEXT REMAINS UNCHANGED ---
    prompt_text = f"""
You are an intelligent assistant for a Nepali government document Q&A system. You must perform THREE tasks in a single response:

{history_section}

**TASK 1: LANGUAGE DETECTION & TRANSLATION**
- Detect if the user query is in English or Nepali
- Set language field to either 'english' or 'nepali' based on the original query
- If English: Translate it to proper Nepali question suitable for query (natural, accurate translation)
- If Nepali: Keep it as is (translated_query will be empty string)
- If question is vague or unclear, try query expansion by expanding the question yourself (but striclty when necessary).

**TASK 2: QUERY REWRITING**
- Use the translated query (if English) or original query (if Nepali)
- If there is conversation history and the query refers to previous context (using pronouns), replace them with specific entities from the history
- Make the query standalone and self-contained for semantic search
- Keep it concise and in Nepali
- If query is already standalone, return it as-is
- DO NOT answer the question - only rewrite it

**TASK 3: CATEGORIZATION**
Classify the rewritten query into:
1. document_type: Either 'passport' or 'citizenship'
2. tag: EXACTLY ONE category tag (in English) that best represents the query's intent

--- Citizenship Tags ---
- eligibility & requirements: Age, relationship, residence, or document requirements
- procedure: Steps for obtaining citizenship, application process, or office procedures
- recommendation: Recommendations, certificates, or identification from ward or authority
- legal: Legal provisions, rules, penalties, or implementation clauses
- special case: Exceptional or unusual citizenship circumstances
- correction & modification: Name, surname, or birth-date corrections and related modifications

--- Passport Tags ---
- eligibility & requirements: Who can apply, age, documents needed, National ID verification
- procedure: Step-by-step application, form filling, where to apply, fees, collection, wait times
- special case: Immediate need, lost/stolen passports, Temporary Passport, special provisions
- correction & modification: Fixing errors, changing personal details
- legal: Legal implications, rules, penalties, document cancellation

**Note:** Fee amount related queries should be strictly categorized under 'procedure'.

**CRITICAL RULES:**
- Output ONLY valid JSON matching the specified format
- Do NOT include explanations, reasoning, or extra text
- Ensure all fields are filled
- rewritten_query must be in Nepali
- language field must be either 'english' or 'nepali' (lowercase)

{format_instructions}

**USER QUERY:** {{query}}

**OUTPUT (JSON only):**
"""
    
    prompt = PromptTemplate(input_variables=["query"], template=prompt_text)
    chain = LLM(prompt, temperature=0.0)
    response = chain.invoke({"query": query})
    
    try:
        # 3. Parse the output (parser.parse takes the string from response.content)
        parsed_obj = parser.parse(response.content)
        
        # Convert Pydantic object back to the dictionary your code expects
        parsed = parsed_obj.dict()
        
        print(f"\nOriginal Query: {parsed['original_query']}")
        print(f"Language: {parsed['language']}")
        if parsed.get('language') == 'english' and parsed.get('translated_query'):
            print(f"Translated to Nepali: {parsed['translated_query']}")
        print(f"Rewritten Query: {parsed['rewritten_query']}")
        print(f"Document Type: {parsed['document_type']}")
        print(f"Category Tag: {parsed['tag']}\n")
        
        return parsed
        
    except Exception as e:
        print(f"Error parsing unified LLM output: {e}")
        print(f"Raw response: {response.content}")
        return None


def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    
    return hdense, hsparse


def hybrid_search(index, query: str, dense_embeddings, bm25_encoder: BM25Encoder,
                  doc_type: str, tag: str, alpha: float = 0.5, top_k: int = 5) -> List[Tuple]:
    """
    Perform hybrid search and return top_k results.
    
    Args:
        index: Pinecone index
        query: Search query
        dense_embeddings: Dense embedding model
        bm25_encoder: BM25 encoder
        doc_type: Document type filter
        tag: Tag filter
        alpha: Balance between dense (1.0) and sparse (0.0)
        top_k: Number of results to retrieve
        
    Returns:
        List of tuples: (text, score, metadata, chunk_id)
    """
    try:
        print(f"Generating dense vector...")
        dense_vector = dense_embeddings.embed_query(query)
        
        filter_dict = {"document_type": {"$eq": doc_type}}
        # filter_dict = {
        #     "$and": [
        #         {"document_type": {"$eq": doc_type}},
        #         {"$or": [{"tag_1": {"$eq": tag}}, {"tag_2": {"$eq": tag}}]}
        #     ]
        # }
        
        if alpha == 1.0:
            print(f"Performing DENSE-ONLY search for top-{top_k} chunks...")
            results = index.query(
                vector=dense_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
        else:
            print(f"Generating sparse vector (BM25)...")
            processed_query = preprocess_text(query)
            sparse_vector = bm25_encoder.encode_queries([processed_query])[0]
            
            scaled_dense, scaled_sparse = hybrid_scale(dense_vector, sparse_vector, alpha)
            
            print(f"Performing HYBRID search (alpha={alpha}) for top-{top_k} chunks...")
            results = index.query(
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
        
        chunks = [
            (
                m['metadata'].get('text', ''), 
                float(m['score']), 
                m["metadata"],
                m.get('id', '')
            )
            for m in results['matches']
        ]
        
        print(f"Retrieved {len(chunks)} chunks from hybrid search")
        return chunks

    except Exception as e:
        print(f"Error in search: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_subsections_with_llm(chunk_content: str, section: str, subsections: List[str]) -> List[str]:
    """Extract multiple subsections from a chunk in a single LLM call."""
    subs_text = ", ".join(subsections)
    prompt_text = f"""
    तपाईं नेपाली कानूनी कागजातमा विशेषज्ञ हुनुहुन्छ।

    तल दिइएको दफा {section} को उपदफा धेरै उपदफा समावेश छ:

    {chunk_content}

    कृपया **सिर्फ उपदफा ({subs_text}) को सामग्री मात्र** निकाल्नुहोस्।
    प्रत्येक उपदफा छुट्टाछुट्टै निकाल्नुहोस्। अरू उपदफा समावेश नगर्नुहोस्। केवल नेपाली Unicode टेक्स्ट फर्काउनुहोस्।
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_text)
    llm_chain = LLM(prompt, temperature=0.0)
    response = llm_chain.invoke({"text": chunk_content})

    extracted = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    return extracted


def fetch_reference_chunks(index, filename: str, section: str, doc_type: str) -> List[Tuple]:
    """
    Fetch reference chunks from Pinecone based on metadata filters.
    """
    try:
        filter_dict = {
            "$and": [
                {"file": {"$eq": filename}},
                {"section": {"$eq": section}},
                {"document_type": {"$eq": doc_type}}
            ]
        }
        
        results = index.query(
            vector=[0.0] * 1024,
            top_k=10,
            filter=filter_dict,
            include_metadata=True
        )
        
        return [(match['metadata'].get('text', ''), match['metadata']) 
                for match in results['matches']]
        
    except Exception as e:
        print(f"Error fetching reference chunks: {e}")
        return []


def retriever(index, query: str, dense_embeddings, bm25_encoder: BM25Encoder,
              chat_history: str = "", 
              n_retrieval: int = 5,
              n_generation: int = 3,
              alpha: float = 0.5,
              use_reranker: bool = True):
    """
    Retrieve relevant chunks and prepare context for RAG.
    
    Args:
        index: Pinecone index
        query: User query
        dense_embeddings: Dense embedding model
        bm25_encoder: BM25 encoder
        chat_history: Previous conversation context
        n_retrieval: Number of chunks to retrieve (for evaluation metrics)
        n_generation: Number of top chunks to use for answer generation
        alpha: Hybrid search balance parameter
        use_reranker: Whether to apply reranking (default: True)
        
    Returns:
        Tuple of (context_for_generation, processed_query_info, original_retrieved_chunks, reranked_chunks_for_generation)
        - context_for_generation: Processed chunks ready for LLM (with references)
        - processed_query_info: Query processing metadata
        - original_retrieved_chunks: All chunks from initial retrieval (for evaluation)
        - reranked_chunks_for_generation: Reranked chunks used for generation (for evaluation)
    """
    try:
        # Step 1: Process query
        processed = process_query_unified(query, chat_history)
        if not processed:
            return ("Error: Could not process query.", None, [], [])

        rewritten_query = processed["rewritten_query"]
        doc_type = processed["document_type"]
        tag = processed["tag"]

        # Step 2: Search - retrieve top n_retrieval chunks
        search_results = hybrid_search(
            index, rewritten_query, dense_embeddings, bm25_encoder,
            doc_type, tag, alpha=alpha, top_k=n_retrieval
        )

        if not search_results:
            return (f"No relevant chunks found for '{doc_type}' with tag '{tag}'.", processed, [], [])

        print(f"\nRetrieved {len(search_results)} chunks for evaluation")
        
        # Store original retrieval order for evaluation
        original_chunk_ids = [chunk[3] for chunk in search_results]
        print(f"Original retrieval order: {[id[:15] for id in original_chunk_ids]}")
        
        # Store original chunks for evaluation
        all_original_chunks = search_results.copy()
        
        # Step 2.5: Apply reranking if enabled
        reranked_chunks_for_generation = []
        
        if use_reranker:
            # Rerank all retrieved chunks
            reranked_results, reranked_chunk_ids = rerank_chunks(rewritten_query, search_results)
            print(f" Reranking applied!")
            print(f"Reranked order: {[id[:15] for id in reranked_chunk_ids]}")
            
            # Select top n_generation from reranked results
            generation_results = reranked_results[:n_generation]
            
            # Store reranked chunks for evaluation (full reranked list)
            reranked_chunks_for_generation = reranked_results
        else:
            # Use original search results for generation
            generation_results = search_results[:n_generation]
            
            # If no reranking, reranked list is same as original
            reranked_chunks_for_generation = search_results
        
        # Step 3: Log which chunks are being used for generation
        print(f"\nUsing top {len(generation_results)} chunks for answer generation")
        print(f"Generation chunk IDs: {[chunk[3][:15] for chunk in generation_results]}")

        # Step 4: Extract base chunks and references for generation
        final_context_with_sources = []

        for content, score, metadata, chunk_id in generation_results:
            chunk_refs = []
            if metadata is None or not isinstance(metadata, dict):
                metadata = {}

            refs = [v for k, v in metadata.items() if k.startswith("references_") and v]
            filename = metadata.get("file")
            
            for ref in refs:
                match = re.match(r"(\d+)(?:\((\d+)\))?", ref)
                if not match:
                    continue
                ref_section, ref_sub = match.groups()
                try:
                    ref_chunks = fetch_reference_chunks(index, filename, ref_section, doc_type)
                    for ref_content, ref_meta in ref_chunks:
                        subsections = [v for k, v in ref_meta.items() if k.startswith("subsection_") and v]

                        if ref_sub is None:
                            nep_section = to_nepali_numeral(ref_section)
                            chunk_refs.append(f"{nep_section}: {ref_content}")
                        elif ref_sub in subsections:
                            extracted_text = extract_subsections_with_llm(ref_content, ref_section, [ref_sub])
                            nep_section = to_nepali_numeral(ref_section)
                            nep_sub = to_nepali_numeral(ref_sub)
                            chunk_refs.append(f"{nep_section}({nep_sub}): {extracted_text[0]}")

                except Exception as e:
                    print(f"Error fetching reference {ref_section} from {filename}: {e}")

            # Step 5: Merge base chunk and references
            context_text = ""
            if chunk_refs:
                context_text += "Reference Chunks:\n" + "\n".join(chunk_refs) + "\n"

            context_text += "Base Chunk:\n" + content
            context_text += f"\n\nSource Link: {metadata.get('source_link','')}\nSource Type: {metadata.get('source_type','')}"

            final_context_with_sources.append({
                "context": context_text.strip(),
                "source_link": metadata.get("source_link", ""),
                "source_type": metadata.get("source_type", ""),
                "chunk_id": chunk_id,
                "score": score
            })

        print("\nFinal context chunks prepared for RAG prompt:")
        for i, ctx_obj in enumerate(final_context_with_sources):
            print(f"--- Chunk {i+1} ---")
            print(f"Chunk ID: {ctx_obj['chunk_id'][:15]}...")
            print(f"Score: {ctx_obj['score']:.4f}")
            print(f"Source: {ctx_obj['source_type']}")
            print(f"Preview: {ctx_obj['context'][:150]}...\n")

        # Return structure:
        # 1. final_context_with_sources: For LLM generation
        # 2. processed: Query processing info
        # 3. all_original_chunks: Original retrieval results (for evaluation)
        # 4. reranked_chunks_for_generation: Reranked results (for evaluation)
        return (final_context_with_sources, processed, all_original_chunks, reranked_chunks_for_generation)
        
    except Exception as e:
        print(f" Error in retriever: {e}")
        import traceback
        traceback.print_exc()
        return ([], None, [], []) # Return an empty list, not an error string