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
import google.generativeai as genai
import base64
import io
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="SlideSense PDF Analyser",
    page_icon="📘",
    layout="wide"
)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --------------------------------------------------
# BASIC SAFE STYLING (Cloud-Safe)
# --------------------------------------------------
st.markdown("""
<style>
.stApp { background-color:#0f172a; color:white; }
.section { padding:1.5rem; border-radius:12px; background:#020617; margin-top:1.5rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📘 SlideSense PDF Analyser")

# --------------------------------------------------
# PDF UPLOAD & PROCESSING
# --------------------------------------------------
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    with st.spinner("Processing PDF..."):
        reader = PdfReader(pdf)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_texts(chunks, embeddings)

    st.success("✅ PDF processed successfully")

    # --------------------------------------------------
    # PDF QUESTION ANSWERING
    # --------------------------------------------------
    pdf_query = st.text_input("Ask a question about the PDF")

    if pdf_query:
        with st.spinner("🤖 Generating PDF answer..."):
            docs = vector_db.similarity_search(pdf_query)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash"
            )

            prompt = ChatPromptTemplate.from_template(
                "Answer the question using the context below.\n\n{context}\n\nQuestion: {question}"
            )

            chain = create_stuff_documents_chain(llm, prompt)
            pdf_response = chain.invoke({
                "context": docs,
                "question": pdf_query
            })

        st.markdown(f"""
        <div class="section">
            <b>📘 PDF Response</b><br><br>
            {pdf_response}
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# IMAGE VISUAL QUESTION ANSWERING (SAFE VERSION)
# --------------------------------------------------
st.divider()
st.subheader("🖼️ Visual Question Answering")

image_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

image_question = st.text_input(
    "Ask a question about the image"
)

if image_file and image_question:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    with st.spinner("🧠 Analyzing image..."):
        model = genai.GenerativeModel("gemini-1.5-pro")

        response = model.generate_content([
            image_question,
            {
                "mime_type": "image/jpeg",
                "data": img_base64
            }
        ])

    st.markdown(f"""
    <div class="section">
        <b>🖼️ Image Response</b><br><br>
        {response.text}
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------
if not pdf and not image_file:
    st.info("Upload a PDF or an image to begin analysis.")
