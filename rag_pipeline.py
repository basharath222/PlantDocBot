import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    task_type="RETRIEVAL_QUERY"
)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function = embeddings)

def get_expert_advice(disease_name, language="English"):
    docs = vector_db.similarity_search(disease_name, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    client = genai.Client(api_key = os.getenv("GOOGLE_API_KEY"))
    lang_instruction = "Give the response in Tamil." if language == "Tamil" else "Give the response in English."
    prompt = f"""
        You are an expert plant pathologist. 
        Using the following excerpts from a specialized tomato research manual:
        {context_text}

        Provide a detailed diagnosis and treatment plan for {disease_name}. 
        {lang_instruction}
        Include:
        1. Visual identification markers (based on the manual).
        2. Biological/Cultural control (Prevention).
        3. Chemical control (if mentioned).

        """
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt
    )
    return response.text

# print(get_expert_advice("Early Blight"))