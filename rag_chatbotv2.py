import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize embeddings and Pinecone vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Chat history
chat_history = []  # Stores {"role": "user"/"assistant", "content": "text"}

def retrieve_context(query, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

def generate_answer(query):
    global chat_history

    context = retrieve_context(query)
    messages = [
        {"role": "system", "content": "You are an AI assistant for Changi Airport Group and Jewel Changi Airport. Use the provided context and past conversation to answer accurately."}
    ] + chat_history + [
        {"role": "user", "content": f"Context:\n{context}\n\nUser question: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_tokens=200
    )

    answer = response.choices[0].message["content"]

    # Update chat history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

