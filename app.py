import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import streamlit as st
from rag_pipeline import get_expert_advice

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
st.set_page_config(page_title="PlantdocBot",page_icon="🌿")

st.markdown(
    """
    <style>
    .main{
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border: none;
    }
    .report-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1d2129;
        border-left: 5px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)
    




with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/128/7963/7963836.png", width = 100)
    st.title("Settings")
    language = st.selectbox("Choose Language / மொழியைத் தேர்ந்தெடுக்கவும்",["English","Tamil"])

# Dictionary to handle translations
strings = {
    "English": {
        "title": "🌿 PlantDocBot",
        "sub": "AI Plant Leaf Diagnosis",
        "upload_label": "Upload a photo of an infected leaf",
        "btn": "Get Diagnosis",
        "spinner1": "Identifying disease...",
        "spinner2": "Fetching expert advice...",
        "detected": "Detected:",
        "info": "Please upload a leaf image to start.",
        "complete":"Analysis Complete!"
    },
    "Tamil": {
        "title": "🌿 பிளாண்ட்-டாக் பாட்",
        "sub": "AI தாவர இலை நோய் கண்டறிதல்",
        "upload_label": "பாதிக்கப்பட்ட இலையின் புகைப்படத்தைப் பதிவேற்றவும்",
        "btn": "நோய் கண்டறிதல்",
        "spinner1": "நோயைக் கண்டறிகிறது...",
        "spinner2": "நிபுணர் ஆலோசனையைப் பெறுகிறது...",
        "detected": "கண்டறியப்பட்டது:",
        "info": "தொடங்க இலை படத்தை பதிவேற்றவும்.",
        "complete":"ஆய்வீடு முடிந்தது! "
    }
}
s = strings[language]

st.title(s["title"])
st.subheader(s["sub"])

upload = st.file_uploader(s["upload_label"], type = ["jpg","jpeg","webp","png"])

if upload is not None:
    img = Image.open(upload)
    st.image(img, caption = "Uploaded leaf", width='stretch')
    if st.button(s["btn"]):
        disease = None
        with st.spinner(s["spinner1"]):
            try:
                response = client.models.generate_content(
                    model = "gemini-2.5-flash",
                    contents=["What is the name of the disease in this leaf? Give only the name.", img]                    
                  )
                disease = response.text.strip()
                st.write(f"### {s['detected']} **{disease}**")
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        if disease:
            with st.spinner(s["spinner2"]):
                try:
                    from rag_pipeline import get_expert_advice
                    final_advice = get_expert_advice(disease,language)
                    
                    st.success(f"✅ {s['complete']}")
                    st.markdown(f'<div class="report-box">{final_advice}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    
else:
    st.info(s["info"])

