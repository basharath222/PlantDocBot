import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def create_database():
    pdf_path = "knowledge_base.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error {pdf_path} not found!")
        return
    
    print("__Step 1: Loading PDF__")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    print("__Step 2: Chunking Text__")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunk = text_splitter.split_documents(data)
    print(f"Created {len(chunk)} text chunks.")

    print("__Step 3: Generate Embedding and save to disk__")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        task_type="RETRIEVAL_QUERY"
    )
    vector_db = Chroma.from_documents(
        documents=chunk,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Database successfully created!")

if __name__ == "__main__":
    create_database()