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

def retrieve_context(query, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

def generate_answer(query):
    context = retrieve_context(query)
    prompt = f"Context:\n{context}\n\nUser question: {query}\nAnswer as helpfully as possible."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message["content"]
