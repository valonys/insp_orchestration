"""
Unified Streamlit App: Notification Summary + LLM Chat for PDFs

This app combines:
1. notifs.py - NI/NC notifications analysis and FPSO layout
2. app.py - LLM-powered assistant with PDF context

Author: ValonyLabs
"""


# === IMPORTS ===
import streamlit as st
import pandas as pd
import os
import json
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from functools import wraps
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from cerebras.cloud.sdk import Cerebras

# === GLOBAL CONFIG ===
load_dotenv()
st.set_page_config(page_title='DigiTwin RAG Dashboard', layout='wide')
st.markdown('''
    <style>
    @import url("https://fonts.cdnfonts.com/css/tw-cen-mt");
    * {
        font-family: "Tw Cen MT", sans-serif !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "▶";
        font-size: 1.3rem;
        margin-right: 0.4rem;
    }
    .logo-container {
        position: fixed;
        top: 5rem;
        right: 12rem;
        z-index: 9999;
    }
    </style>
''', unsafe_allow_html=True)
st.markdown('''<div class="logo-container"><img src="https://github.com/valonys/DigiTwin/blob/29dd50da95bec35a5abdca4bdda1967f0e5efff6/ValonyLabs_Logo.png?raw=true" width="70"></div>''', unsafe_allow_html=True)
st.title('📊 DigiTwin - The Insp Nerdzx')


# === DECORATORS ===
def safe_run(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"🚨 Error in {func.__name__}: {e}")
            return None
    return wrapper

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        st.sidebar.write(f"⏱️ {func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return wrapper

# === SIDEBAR ===
with st.sidebar:
    st.header("📄 Upload Files")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    uploaded_excel = st.file_uploader("Upload Excel", type=["xlsx"])

    st.markdown("---")
    selected_prompt = st.selectbox("Choose Prompt Type", ["Inspection Summary", "Failure Cause", "Planning"])
    selected_fpso = st.selectbox("Select FPSO for Layout", ["GIR", "DAL", "PAZ", "CLV"])

# === NOTIF KEYWORDS & MAPPING ===
NI_keywords = ['WRAP', 'WELD', 'TBR', 'PACH', 'PATCH', 'OTHE', 'CLMP', 'REPL', 'BOND', 'BOLT', 'SUPP', 'OT', 'GASK']
NC_keywords = ['COA', 'ICOA', 'CUSP', 'WELD', 'REPL', 'CUSP1', 'CUSP2']
module_keywords = ['M110', 'M111', 'M112', 'M113', 'M114', 'M115', 'M116', 'H151', 'M120', 'M121', 'M122', 'M123', 'M124', 'M125', 'M126', 'M151']
rack_keywords = ['141', '142', '143', '144', '145', '146']
living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4']
flare_keywords = ['131']
hexagons_keywords = ['HELIDECK']

@safe_run
@time_it
def parse_excel_notifications(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        ni_df = df[df["TAG"].astype(str).str.contains('|'.join(NI_keywords), case=False, na=False)]
        nc_df = df[df["TAG"].astype(str).str.contains('|'.join(NC_keywords), case=False, na=False)]
        return ni_df, nc_df
    return None, None

@safe_run
def show_fpso_layout(fpso):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(patches.Rectangle((2, 2), 6, 4, edgecolor='black', facecolor='lightblue'))
    ax.text(5, 4, f"{fpso} Layout", ha='center', va='center', fontsize=14, weight='bold')
    st.pyplot(fig)

@safe_run
@time_it
def extract_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

@safe_run
def build_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [LCDocument(page_content=chunk) for chunk in text_chunks]
    db = FAISS.from_documents(documents, embeddings)
    return db

@safe_run
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_text(text)

tab1, tab2, tab3 = st.tabs(["📈 Notifications Dashboard", "📍 FPSO Layout", "💬 LLM Assistant"])

with tab1:
    st.subheader("📌 NI / NC Summary")
    ni_df, nc_df = parse_excel_notifications(uploaded_excel)
    if ni_df is not None:
        st.write("**NI Summary**", ni_df.head(10))
    if nc_df is not None:
        st.write("**NC Summary**", nc_df.head(10))

with tab2:
    st.subheader("📍 FPSO Layout View")
    show_fpso_layout(selected_fpso)

with tab3:
    st.subheader("💬 Chat with the LLM using PDF Knowledge")
    if uploaded_pdf:
        text = extract_pdf_text(uploaded_pdf)
        chunks = split_text(text)
        db = build_vectorstore(chunks)
        query = st.text_input("Ask a question based on the uploaded PDF")
        if query and db:
            docs = db.similarity_search(query)
            st.markdown("**Context Found:**")
            for i, doc in enumerate(docs[:3]):
                st.info(doc.page_content[:500])
            st.markdown("**Model Response:**")
            st.write(f"🧠 _Simulated answer for_: {query}")
    else:
        st.info("Upload a PDF to start chatting.")
