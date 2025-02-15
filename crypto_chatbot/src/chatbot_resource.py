import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from mistralai import Mistral
import uvicorn

MISTRAL_API_KEY = dotenv.get_key(".env", "MISTRAL_API_KEY")
RAG_MODEL_NAME = "BAAI/bge-m3"
MISTRAL_MODEL_NAME = "mistral-large-latest"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cpu"

embedding_model = HuggingFaceEmbeddings(
    model_name=RAG_MODEL_NAME,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

VECTOR_STORE_PATH = "vector_store"
try:
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    logger.info(f"Loaded vector store from {VECTOR_STORE_PATH}")
except Exception as e:
    raise Exception(f"Failed to load vector store from {VECTOR_STORE_PATH}: {e}")

llm = Mistral(api_key=MISTRAL_API_KEY)
app = FastAPI(title="Crypto Chatbot RAG", description="A chatbot powered by a RAG pipeline using Mistral and FAISS.")

class ChatRequest(BaseModel):
    user_question: str
    context: str = "This is the first message from the user. Please infer the context solely from the provided documents."

class ChatResponse(BaseModel):
    context: str
    chatbot_response: str

def generate_prompt(user_question: str, retrieved_docs, is_first_message: bool = True) -> str:
    docs_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    if is_first_message:
        context_instruction = (
            "This is the first message from the user. Please infer the context solely from the provided documents."
        )
    else:
        context_instruction = (
            "Please use the conversation context provided below to refine your answer."
        )
    
    prompt = f"""
You are a knowledgeable assistant specialized in cryptocurrencies, blockchain, and crypto scams.
{context_instruction}

Relevant Documents:
{docs_text}

User Question:
{user_question}

Return your answer in the following JSON format:
{{
  "context": "context",
  "chatbot_response": "chatbot response to the user message"
}}
"""
    return prompt.strip()

def rag_chat(user_question: str, context: str) -> dict:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(user_question)
    logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {user_question}")
    is_first_message = (context.strip() == "" or "first message" in context.lower())
    prompt = generate_prompt(user_question, retrieved_docs, is_first_message)
    logger.info(f"Generated prompt (first 200 chars): {prompt[:200]}...")
    
    chat_response = llm.chat.complete(
        model=MISTRAL_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_message = chat_response.choices[0].message.content
    logger.info(f"Raw Mistral message content: {raw_message}")
    
    response_json = None
    try:
        response_json = json.loads(raw_message)
    except Exception as e:
        logger.warning(f"Direct JSON parsing failed: {e}")
        try:
            extracted = raw_message.split("```json")[-1].split("```")[0].strip()
            response_json = json.loads(extracted)
        except Exception as e2:
            logger.error(f"Error extracting JSON from code block: {e2}")
            response_json = {
                "context": "",
                "chatbot_response": raw_message
            }
    if "chatbot_response" in response_json:
        inner = response_json["chatbot_response"].strip()
        if inner.startswith("{"):
            try:
                inner_json = json.loads(inner)
                if "context" in inner_json and "chatbot_response" in inner_json:
                    response_json = inner_json
            except Exception as e:
                logger.warning(f"Could not parse inner chatbot_response as JSON: {e}")
    return response_json

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.user_question:
        raise HTTPException(status_code=400, detail="User question is required.")
    result = rag_chat(request.user_question, request.context)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5010)
