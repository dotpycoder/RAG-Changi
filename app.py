from fastapi import FastAPI
from rag_chatbotv2 import generate_answer

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, Render is working!"}

@app.get("/ask")
def ask(query: str):
    answer = generate_answer(query)
    return {"query": query, "answer": answer}
