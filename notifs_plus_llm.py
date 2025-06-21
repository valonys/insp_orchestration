import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import math
import os
import time
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from cerebras.cloud.sdk import Cerebras
from functools import wraps
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# --- UI CONFIG & STYLE ---
st.set_page_config(page_title="DigiTwin Analytics", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * {
        font-family: 'Tw Cen MT', sans-serif !important;
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Display logo
st.markdown(
    """
    <div class="logo-container">
        <img src="https://github.com/valonys/DigiTwin/blob/29dd50da95bec35a5abdca4bdda1967f0e5efff6/ValonyLabs_Logo.png?raw=true" width="70">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 DigiTwin Analytics - Powered by AI")

# --- AVATARS ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- SYSTEM PROMPTS ---
PROMPTS = {
    "Daily Report Summarization": """You are DigiTwin, an expert inspector and maintenance engineer...""",
    "5-Day Progress Report": """You are DigiTwin, an expert inspector with deep knowledge in KPIs, GM, CR...""",
    "Backlog Extraction": """You are DigiTwin, an expert inspector trained to extract and classify backlogs...""",
    "Inspector Expert": """You are DigiTwin, an expert inspector for advanced diagnosis and recommendation...""",
    "Complex Reasoning": """You are DigiTwin, trained to analyze multi-day reports using GS-OT-MIT-511 rules..."""
}

# --- STATE ---
for key in ["vectorstore", "chat_history", "model_intro_done", "current_model", "current_prompt"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None if key == "vectorstore" else False

# --- NOTIFICATION KEYWORDS AND MAPPINGS ---
NI_keywords = ['WRAP', 'WELD', 'TBR', 'PACH', 'PATCH', 'OTHE', 'CLMP', 'REPL', 
               'BOND', 'BOLT', 'SUPP', 'OT', 'GASK']
NC_keywords = ['COA', 'ICOA', 'CUSP', 'WELD', 'REPL', 'CUSP1', 'CUSP2']
module_keywords = ['M110', 'M111', 'M112', 'M113', 'M114', 'M115', 'M116', 'H151',
                  'M120', 'M121', 'M122', 'M123', 'M124', 'M125', 'M126', 'M151']
rack_keywords = ['141', '142', '143', '144', '145', '146']
living_quarters_keywords = ['LQ', 'LQ1', 'LQ2', 'LQ3', 'LQ4', 'LQL0', 'LQPS', 'LQSB', 'LQROOF', 'LQL4', 'LQL2', 'LQ-5', 'LQPD', 'LQ PS', 'LQAFT', 'LQ-T', 'LQL1S']
flare_keywords = ['131']
fwd_keywords = ['FWD']
hexagons_keywords = ['HELIDECK']

NI_keyword_map = {'TBR1': 'TBR', 'TBR2': 'TBR', 'TBR3': 'TBR', 'TBR4': 'TBR'}
NC_keyword_map = {'COA1': 'COA', 'COA2': 'COA', 'COA3': 'COA', 'COA4': 'COA'}

# CLV location dictionaries
clv_modules = {
    'M120': (0.75, 2), 'M121': (0.5, 3), 'M122': (0.5, 4), 'M123': (0.5, 5),
    'M124': (0.5, 6), 'M125': (0.5, 7), 'M126': (0.5, 8), 'M151': (0.5, 9), 'M110': (1.75, 2),
    'M111': (2, 3), 'M112': (2, 4), 'M113': (2, 5), 'M114': (2, 6),
    'M115': (2, 7), 'M116': (2, 8), 'H151': (2, 9)
}
clv_racks = {'141': (1.5, 3), '142': (1.5, 4), '143': (1.5, 5),
             '144': (1.5, 6), '145': (1.5, 7), '146': (1.5, 8)}
clv_flare = {'131': (1.5, 9)}
clv_living_quarters = {'LQ': (0.5, 1)}
clv_hexagons = {'HELIDECK': (2.75, 1)}
clv_fwd = {'FWD': (0.5, 10)}

# --- DECORATORS ---
def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

# --- SIDEBAR ---
with st.sidebar:
    st.title("DigiTwin Control Panel")
    model_alias = st.selectbox("Choose AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    uploaded_pdfs = st.file_uploader("📄 Upload PDF Reports", type=["pdf"], accept_multiple_files=True)
    uploaded_xlsx = st.file_uploader("📊 Upload Excel Notifications", type=["xlsx"])
    prompt_type = st.selectbox("Select Task Type", list(PROMPTS.keys()))
    selected_fpso = st.selectbox("Select FPSO for Layout", ['GIR', 'DAL', 'PAZ', 'CLV'])

# --- DATA PROCESSING FUNCTIONS ---
@log_execution
def parse_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

@st.cache_resource
def build_faiss_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for i, doc in enumerate(_docs):
        for chunk in splitter.split_text(doc.page_content):
            chunks.append(LCDocument(page_content=chunk, metadata={"source": f"doc_{i}"}))
    return FAISS.from_documents(chunks, embeddings)

@log_execution
def preprocess_keywords(description):
    description = str(description).upper()
    for lq_variant in living_quarters_keywords:
        if lq_variant != 'LQ':
            description = description.replace(lq_variant, 'LQ')
    for module in module_keywords:
        number = module[1:]
        if number in description:
            description = description.replace(number, module)
    for original, grouped in {**NI_keyword_map, **NC_keyword_map}.items():
        description = description.replace(original, grouped)
    return description

@log_execution
def extract_ni_nc_keywords(row, notif_type_col, desc_col):
    description = preprocess_keywords(row[desc_col])
    notif_type = row[notif_type_col]
    keywords = [kw for kw in (NI_keywords if notif_type == 'NI' else NC_keywords) if kw in description]
    return ', '.join(keywords) if keywords else 'None'

@log_execution
def extract_location_keywords(row, desc_col, keyword_list):
    description = preprocess_keywords(row[desc_col])
    if keyword_list == living_quarters_keywords:
        return 'LQ' if any(kw in description for kw in living_quarters_keywords) else 'None'
    locations = [kw for kw in keyword_list if kw in description]
    return ', '.join(locations) if locations else 'None'

@log_execution
def create_pivot_table(df, index, columns, aggfunc='size', fill_value=0):
    df_exploded = df.assign(Keywords=df[columns].str.split(', ')).explode('Keywords')
    df_exploded = df_exploded[df_exploded['Keywords'] != 'None']
    pivot = pd.pivot_table(df_exploded, index=index, columns='Keywords', aggfunc=aggfunc, fill_value=fill_value)
    return pivot

@log_execution
def apply_fpso_colors(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    color_map = {'GIR': '#FFA07A', 'DAL': '#ADD8E6', 'PAZ': '#D8BFD8', 'CLV': '#90EE90'}
    for fpso, color in color_map.items():
        if fpso in df.index:
            styles.loc[fpso] = f'background-color: {color}'
    return styles

# --- FPSO LAYOUT FUNCTIONS ---
@log_execution
def add_rectangle(ax, xy, width, height, **kwargs):
    rectangle = patches.Rectangle(xy, width, height, **kwargs)
    ax.add_patch(rectangle)

@log_execution
def add_chamfered_rectangle(ax, xy, width, height, chamfer, **kwargs):
    x, y = xy
    coords = [
        (x + chamfer, y), (x + width - chamfer, y), (x + width, y + chamfer),
        (x + width, y + height - chamfer), (x + width - chamfer, y + height),
        (x + chamfer, y + height), (x, y + height - chamfer), (x, y + chamfer)
    ]
    polygon = patches.Polygon(coords, closed=True, **kwargs)
    ax.add_patch(polygon)

@log_execution
def add_hexagon(ax, xy, radius, **kwargs):
    x, y = xy
    vertices = [(x + radius * math.cos(2 * math.pi * n / 6), y + radius * math.sin(2 * math.pi * n / 6)) for n in range(6)]
    hexagon = patches.Polygon(vertices, closed=True, **kwargs)
    ax.add_patch(hexagon)

@log_execution
def add_fwd(ax, xy, width, height, **kwargs):
    x, y = xy
    top_width = width * 0.80
    coords = [
        (0, 0), (width, 0), (width - (width - top_width) / 2, height),
        ((width - top_width) / 2, height)
    ]
    trapezoid = patches.Polygon(coords, closed=True, **kwargs)
    t = transforms.Affine2D().rotate_deg(90).translate(x, y)
    trapezoid.set_transform(t + ax.transData)
    ax.add_patch(trapezoid)
    text_t = transforms.Affine2D().rotate_deg(90).translate(x + height / 2, y + width / 2)
    ax.text(0, -1, "FWD", ha='center', va='center', fontsize=7, weight='bold', transform=text_t + ax.transData)

@log_execution
def draw_clv(ax, df_selected, notification_type, location_counts):
    for module, (row, col) in clv_modules.items():
        height, y_position, text_y = (1.25, row, row + 0.5) if module == 'M110' else (1.25, row - 0.25, row + 0.25) if module == 'M120' else (1, row, row + 0.5)
        add_chamfered_rectangle(ax, (col, y_position), 1, height, 0.1, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, text_y, module, ha='center', va='center', fontsize=7, weight='bold')
        if module in module_keywords and int(location_counts['Modules'].loc[module, 'Count']) > 0:
            ax.text(col + 0.8, row + 0.8, f"{int(location_counts['Modules'].loc[module, 'Count'])}", 
                    ha='center', va='center', fontsize=6, weight='bold', color='red')

    for rack, (row, col) in clv_racks.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, rack, ha='center', va='center', fontsize=7, weight='bold')
        if rack in rack_keywords and int(location_counts['Racks'].loc[rack, 'Count']) > 0:
            ax.text(col + 0.7, row + 0.4, f"{int(location_counts['Racks'].loc[rack, 'Count'])}", 
                    ha='center', va='center', fontsize=6, weight='bold', color='red')

    for flare_loc, (row, col) in clv_flare.items():
        add_chamfered_rectangle(ax, (col, row), 1, 0.5, 0.05, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 0.25, flare_loc, ha='center', va='center', fontsize=7, weight='bold')
        if flare_loc in flare_keywords and int(location_counts['Flare'].loc[flare_loc, 'Count']) > 0:
            ax.text(col + 0.7, row + 0.4, f"{int(location_counts['Flare'].loc[flare_loc, 'Count'])}", 
                    ha='center', va='center', fontsize=6, weight='bold', color='red')

    for lq, (row, col) in clv_living_quarters.items():
        add_rectangle(ax, (col, row), 1, 2.5, edgecolor='black', facecolor='white')
        ax.text(col + 0.5, row + 1.25, lq, ha='center', va='center', fontsize=7, rotation=90, weight='bold')
        total_lq_count = sum(df_selected['Extracted_LivingQuarters'].str.contains(keyword, na=False).sum() for keyword in living_quarters_keywords)
        if total_lq_count > 0:
            ax.text(col + 0.7, row + 1.4, f"{total_lq_count}", 
                    ha='center', va='center', fontsize=6, weight='bold', color='red')

    for hexagon, (row, col) in clv_hexagons.items():
        add_hexagon(ax, (col, row), 0.60, edgecolor='black', facecolor='white')
        ax.text(col, row, hexagon, ha='center', va='center', fontsize=7, weight='bold')
        if hexagon in hexagons_keywords and int(location_counts['HeliDeck'].loc[hexagon, 'Count']) > 0:
            ax.text(col + 0.2, row + 0.2, f"{int(location_counts['HeliDeck'].loc[hexagon, 'Count'])}", 
                    ha='center', va='center', fontsize=6, weight='bold', color='red')

    for fwd_loc, (row, col) in clv_fwd.items():
        add_fwd(ax, (col, row), 2.5, -1, edgecolor='black', facecolor='white')
        if fwd_loc in fwd_keywords and int(location_counts['FWD'].loc[fwd_loc, 'Count']) > 0:
            ax.text(col + 0.75, row + 1.4, f"{int(location_counts['FWD'].loc[fwd_loc, 'Count'])}", 
                    ha='center', va='center', fontsize=6, weight='bold', color='red')

    total_ni = df_selected[df_selected['Notifictn type'] == 'NI'].shape[0]
    total_nc = df_selected[df_selected['Notifictn type'] == 'NC'].shape[0]
    ax.text(6, 0.25, f"NI: {total_ni}\nNC: {total_nc}", ha='center', va='center', fontsize=8, weight='bold', color='red')

@log_execution
def draw_fpso_layout(selected_unit, df_selected, notification_type, location_counts):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_facecolor('#E6F3FF')
    if selected_unit == 'CLV':
        draw_clv(ax, df_selected, notification_type, location_counts)
    else:
        ax.text(6, 1.75, f"{selected_unit} Layout\n(Implementation work in progress...)", 
                ha='center', va='center', fontsize=16, weight='bold')
    plt.title(f"FPSO Visualization - {selected_unit}", fontsize=16, fontfamily='Tw Cen MT')
    return fig

# --- LLM RESPONSE LOGIC ---
@log_execution
def generate_response(prompt, df=None):
    messages = [{"role": "system", "content": PROMPTS[prompt_type]}]
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages.append({"role": "system", "content": f"Context from PDF reports:\n{context}"})
    if df is not None:
        summary = df.describe().to_string()
        messages.append({"role": "system", "content": f"Excel data summary:\n{summary}"})
    messages.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        if model_alias == "EE Smartest Agent":
            client = openai.OpenAI(api_key=os.getenv("API_KEY"), base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(model="grok-3", messages=messages, stream=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"

        elif model_alias == "JI Divine Agent":
            client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.sambanova.ai/v1")
            response = client.chat.completions.create(model="DeepSeek-R1-Distill-Llama-70B", messages=messages, stream=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"

        elif model_alias == "EdJa-Valonys":
            client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
            response = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=messages)
            content = response.choices[0].message.content if hasattr(response.choices[0], "message") else str(response.choices[0])
            for word in content.split():
                full_response += word + " "
                yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                time.sleep(0.01)

        elif model_alias == "XAI Inspector":
            model_id = "amiguel/GM_Qwen1.8B_Finetune"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", token=os.getenv("HF_TOKEN"))
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
            output = model.generate(input_ids, max_new_tokens=512, do_sample=True, top_p=0.9)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

        elif model_alias == "Valonys Llama":
            model_id = "amiguel/Llama3_8B_Instruct_FP16"
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=os.getenv("HF_TOKEN"))
            input_ids = tokenizer(PROMPTS[prompt_type] + "\n\n" + prompt, return_tensors="pt").to(model.device)
            output = model.generate(**input_ids, max_new_tokens=512)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

    except Exception as e:
        yield f"<span style='color:red'>⚠️ Error: {str(e)}</span>"

# --- DATA LOADING ---
if uploaded_pdfs:
    parsed_docs = [LCDocument(page_content=parse_pdf(f), metadata={"name": f.name}) for f in uploaded_pdfs]
    st.session_state.vectorstore = build_faiss_vectorstore(parsed_docs)
    st.sidebar.success(f"{len(parsed_docs)} PDF reports indexed.")

df = None
if uploaded_xlsx:
    try:
        df = pd.read_excel(uploaded_xlsx, sheet_name='Global Notifications')
        df.columns = df.columns.str.strip()
        expected_columns = {
            'Notifictn type': 'Notifictn type',
            'Created on': 'Created on',
            'Description': 'Description',
            'FPSO': 'FPSO'
        }
        missing_columns = [col for col in expected_columns.values() if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
            st.stop()
        df = df[list(expected_columns.values())]
        df.columns = list(expected_columns.keys())
        df = df[df['FPSO'].isin(['GIR', 'DAL', 'PAZ', 'CLV'])]
        df['Extracted_Keywords'] = df.apply(extract_ni_nc_keywords, axis=1, args=('Notifictn type', 'Description'))
        for loc_type, keywords in [
            ('Modules', module_keywords), ('Racks', rack_keywords), ('LivingQuarters', living_quarters_keywords),
            ('Flare', flare_keywords), ('FWD', fwd_keywords), ('HeliDeck', hexagons_keywords)
        ]:
            df[f'Extracted_{loc_type}'] = df.apply(extract_location_keywords, axis=1, args=('Description', keywords))
        st.sidebar.success("Excel file processed successfully.")
    except Exception as e:
        st.error(f"Error processing Excel: {e}")

# --- AGENT INTRO ---
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias or st.session_state.current_prompt != prompt_type:
    agent_intros = {
        "EE Smartest Agent": "💡 EE Agent Activated — Pragmatic & Smart",
        "JI Divine Agent": "✨ JI Agent Activated — DeepSeek Reasoning",
        "EdJa-Valonys": "⚡ EdJa Agent Activated — Cerebras Speed",
        "XAI Inspector": "🔍 XAI Inspector — Qwen Custom Fine-tune",
        "Valonys Llama": "🦙 Valonys Llama — LLaMA3-Based Reasoning"
    }
    st.session_state.chat_history.append({"role": "assistant", "content": agent_intros.get(model_alias)})
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias
    st.session_state.current_prompt = prompt_type

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "NI Notifications", "NC Notifications", "Summary Stats", "FPSO Layout"])

# Chat Tab
with tab1:
    st.subheader("Interact with DigiTwin")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
            st.markdown(msg["content"], unsafe_allow_html=True)
    if prompt := st.chat_input("Ask about reports or notifications..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in generate_response(prompt, df):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# NI Notifications Tab
with tab2:
    st.subheader("NI Notifications Analysis")
    if df is not None and not df[df['Notifictn type'] == 'NI'].empty:
        ni_pivot = create_pivot_table(df[df['Notifictn type'] == 'NI'], index='FPSO', columns='Extracted_Keywords')
        st.write("Pivot Table (Count of Keywords by FPSO):")
        styled_ni_pivot = ni_pivot.style.apply(apply_fpso_colors, axis=None)
        st.dataframe(styled_ni_pivot)
        st.write(f"Total NI Notifications: {df[df['Notifictn type'] == 'NI'].shape[0]}")
    else:
        st.write("No NI notifications found or no Excel file uploaded.")

# NC Notifications Tab
with tab3:
    st.subheader("NC Notifications Analysis")
    if df is not None and not df[df['Notifictn type'] == 'NC'].empty:
        nc_pivot = create_pivot_table(df[df['Notifictn type'] == 'NC'], index='FPSO', columns='Extracted_Keywords')
        st.write("Pivot Table (Count of Keywords by FPSO):")
        styled_nc_pivot = nc_pivot.style.apply(apply_fpso_colors, axis=None)
        st.dataframe(styled_nc_pivot)
        st.write(f"Total NC Notifications: {df[df['Notifictn type'] == 'NC'].shape[0]}")
    else:
        st.write("No NC notifications found or no Excel file uploaded.")

# Summary Stats Tab
with tab4:
    st.subheader("2025 Notification Summary")
    if df is not None:
        df_2025 = df[pd.to_datetime(df['Created on']).dt.year == 2025].copy()
        if not df_2025.empty:
            df_2025['Month'] = pd.to_datetime(df_2025['Created on']).dt.strftime('%b')
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            df_2025['Month'] = pd.Categorical(df_2025['Month'], categories=months_order, ordered=True)
            summary = df_2025.groupby(['FPSO', 'Month', 'Notifictn type']).size().unstack(fill_value=0)
            ni_summary = summary['NI'].unstack(level='Month').reindex(columns=months_order, fill_value=0)
            nc_summary = summary['NC'].unstack(level='Month').reindex(columns=months_order, fill_value=0)
            st.write("NI Notifications by Month:")
            st.dataframe(ni_summary.style.set_properties(**{'text-align': 'center'}))
            st.write("NC Notifications by Month:")
            st.dataframe(nc_summary.style.set_properties(**{'text-align': 'center'}))
            st.write(f"Grand Total NI Notifications: {df_2025[df_2025['Notifictn type'] == 'NI'].shape[0]}")
            st.write(f"Grand Total NC Notifications: {df_2025[df_2025['Notifictn type'] == 'NC'].shape[0]}")
        else:
            st.write("No notifications found for 2025.")

# FPSO Layout Tab
with tab5:
    st.subheader("FPSO Layout Visualization")
    if df is not None:
        notification_type = st.radio("Select Notification Type", ['NI', 'NC'])
        df_selected = df[df['FPSO'] == selected_fpso].copy()
        df_selected = df_selected[df_selected['Notifictn type'] == notification_type]
        location_counts = {
            'Modules': pd.DataFrame(index=module_keywords, columns=['Count']).fillna(0),
            'Racks': pd.DataFrame(index=rack_keywords, columns=['Count']).fillna(0),
            'LivingQuarters': pd.DataFrame(index=living_quarters_keywords, columns=['Count']).fillna(0),
            'Flare': pd.DataFrame(index=flare_keywords, columns=['Count']).fillna(0),
            'FWD': pd.DataFrame(index=fwd_keywords, columns=['Count']).fillna(0),
            'HeliDeck': pd.DataFrame(index=hexagons_keywords, columns=['Count']).fillna(0)
        }
        for location_type, keywords in [
            ('Modules', module_keywords), ('Racks', rack_keywords), ('LivingQuarters', living_quarters_keywords),
            ('Flare', flare_keywords), ('FWD', fwd_keywords), ('HeliDeck', hexagons_keywords)
        ]:
            for keyword in keywords:
                count = df_selected[f'Extracted_{location_type}'].str.contains(keyword, na=False).sum()
                location_counts[location_type].loc[keyword, 'Count'] = count
        fig = draw_fpso_layout(selected_fpso, df_selected, notification_type, location_counts)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Please upload an Excel file to view the FPSO layout.")
