from fastapi import FastAPI
from rag_chatbotv2 import generate_answer
from rag_chatbotv2 import generate_answer, OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, Render is working!"}

@app.get("/ask")
def ask(query: str):
    answer = generate_answer(query)
    return {"query": query, "answer": answer}

@app.get("/check_keys")
def check_keys():
    return {
        "OPENAI_API_KEY_set": OPENAI_API_KEY is not None,
        "PINECONE_API_KEY_set": PINECONE_API_KEY is not None,
        "PINECONE_INDEX_NAME": INDEX_NAME
    }