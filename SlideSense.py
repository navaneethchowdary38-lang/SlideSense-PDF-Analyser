import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from PIL import Image
import io

# ------------------------------------
# CONFIG
# ------------------------------------
st.set_page_config(
    page_title="SlideSense PDF Analyser",
    page_icon="📘",
    layout="wide"
)

load_dotenv()

# ------------------------------------
# SIMPLE SAFE CSS (BOOT SAFE)
# ------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
.response-section { padding: 1.5rem; border-radius: 12px; background: #020617; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# TITLE
# ------------------------------------
st.title("📘 SlideSense PDF Analyser")

# ------------------------------------
# PDF UPLOAD
# ------------------------------------
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed successfully")

    query = st.text_input("Ask a question about the PDF")

    if query:
        docs = vector_db.similarity_search(query)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        prompt = ChatPromptTemplate.from_template(
            "Answer using the context below:\n\n{context}\n\nQuestion: {question}"
        )

        chain = create_stuff_documents_chain(llm, prompt)
        response = chain.invoke({
            "context": docs,
            "question": query
        })

        st.markdown(f"""
        <div class="response-section">
            <b>PDF Response</b><br><br>
            {response}
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------
# IMAGE QUESTION ANSWERING (SAFE)
# ------------------------------------
st.divider()
st.subheader("🖼️ Visual Question Answering")

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
image_question = st.text_input("Ask a question about the image")

if image_file and image_question:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes = img_bytes.getvalue()

    vision_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro"
    )

    response = vision_llm.invoke([
        {
            "role": "user",
            "parts": [
                {"text": image_question},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_bytes
                    }
                }
            ]
        }
    ])

    st.markdown(f"""
    <div class="response-section">
        <b>Image Response</b><br><br>
        {response.content}
    </div>
    """, unsafe_allow_html=True)
