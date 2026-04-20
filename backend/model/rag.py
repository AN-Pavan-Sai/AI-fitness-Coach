import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Shared globals
embeddings = None
vectorstore = None
retriever = None
llm = None

FITNESS_PROMPT = """You are an expert AI Fitness Coach.
Your primary role is to answer questions strictly related to health, fitness, workouts, nutrition, and wellness.

Context from our fitness database:
{context}

Question: {question}

IMPORTANT RULES:
1. If the user's question is NOT about health, fitness, nutrition, or wellness, you MUST politely refuse to answer. Example: "I am an AI Fitness Coach and can only advise on health and fitness topics."
2. Base your coaching strictly on the provided Context if it's relevant, otherwise use your expertise.
3. Keep answers concise, actionable, and encouraging.

Coach's Answer:"""

def init_rag():
    global embeddings, vectorstore, retriever, llm
    
    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize LLM only if key exists
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)
    else:
        print("WARNING: GROQ_API_KEY is not set. Chat will not work.")
    
    # DB path relative to the backend execution context
    persist_directory = "./model/chroma_db"
    
    # Check if we need to ingest data
    if not os.path.exists(persist_directory):
        print("VectorDB not found. Starting ingestion of merged dataset...")
        try:
            # Assuming backend is executed from /backend folder, dataset is at root
            csv_path = "../merged_rag_dataset.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                docs = df['content'].tolist()
                vectorstore = Chroma.from_texts(
                    texts=docs,
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                print("Ingestion complete.")
            else:
                print(f"Dataset not found at {csv_path}")
        except Exception as e:
            print("Error loading dataset:", e)
            vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    else:
        print("VectorDB found. Loading existing database...")
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def get_fitness_response(user_query: str) -> str:
    if not llm or not retriever:
        return "ERROR: RAG model not configured natively. Please ensure GROQ_API_KEY is defined in your environment."
        
    prompt = ChatPromptTemplate.from_template(FITNESS_PROMPT)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(user_query)
