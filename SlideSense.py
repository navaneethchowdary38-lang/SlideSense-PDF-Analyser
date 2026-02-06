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
import asyncio

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="SlideSense PDF Analyser",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# ----------------------------------------------------
# STYLES (UNCHANGED)
# ----------------------------------------------------
st.markdown("""<style>/* YOUR FULL CSS IS UNCHANGED – OMITTED HERE FOR BREVITY */
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------
# HERO SECTION
# ----------------------------------------------------
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">SlideSense PDF Analyser</h1>
    <p class="hero-subtitle">Advanced Document Analysis with AI Technology</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# PDF UPLOAD
# ----------------------------------------------------
pdf = st.file_uploader(
    "Document Upload",
    type="pdf",
    help="Choose a PDF document for analysis"
)

if pdf is not None:
    with st.spinner("Processing your document..."):
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

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_texts(chunks, embeddings)

    st.markdown("""
    <div class="success-notification">
        <p class="success-text">✅ Document processed successfully</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # PDF QUESTION ANSWERING
    # ----------------------------------------------------
    st.markdown("""
    <div class="query-section">
        <div class="section-title">PDF Question Answering</div>
        <div class="section-subtitle">Ask questions about the uploaded PDF</div>
    </div>
    """, unsafe_allow_html=True)

    pdf_query = st.text_input(
        "Ask a question about the PDF",
        placeholder="Enter your question...",
        label_visibility="collapsed"
    )

    if pdf_query:
        with st.spinner("🤖 Generating answer..."):
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
        <div class="response-section">
            <div class="response-header">
                <span class="response-icon">📘</span>
                <h3 class="response-title">PDF Response</h3>
            </div>
            <div class="response-content">{pdf_response}</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # IMAGE QUESTION ANSWERING (NEW)
    # ----------------------------------------------------
    st.markdown("""
    <div class="query-section">
        <div class="section-title">Visual Question Answering</div>
        <div class="section-subtitle">Upload an image and ask questions about it</div>
    </div>
    """, unsafe_allow_html=True)

    image_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    image_question = st.text_input(
        "Ask a question about the image",
        placeholder="What is shown in this image?",
        label_visibility="collapsed"
    )

    if image_file is not None and image_question:
        image = Image.open(image_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing image..."):
            vision_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro"
            )

            vision_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert visual reasoning assistant."),
                ("human", [
                    {"type": "text", "text": image_question},
                    {"type": "image", "image": image}
                ])
            ])

            vision_chain = vision_prompt | vision_llm
            vision_response = vision_chain.invoke({})

        st.markdown(f"""
        <div class="response-section">
            <div class="response-header">
                <span class="response-icon">🖼️</span>
                <h3 class="response-title">Image Response</h3>
            </div>
            <div class="response-content">{vision_response.content}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="instruction-container">
        <div class="instruction-title">Get Started</div>
        <div class="instruction-text">
            Upload a PDF or an image to unlock AI-powered document and visual understanding.
        </div>
    </div>
    """, unsafe_allow_html=True)
