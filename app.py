import streamlit as st
import pickle
import numpy as np
from PIL import Image
import requests
from pathlib import Path
import base64

# --- ML & NLP Libraries ---
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sentence_transformers import SentenceTransformer

# --- UI/UX Libraries ---
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="StyleSphere | Your AI Stylist",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# MODEL & DATA LOADING (Cached for performance)
# ------------------------------
@st.cache_resource
def load_models():
    """Load and cache the ML models."""
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return resnet_model, text_model

@st.cache_data
def load_embeddings():
    """Load and cache the precomputed embeddings from pickle files."""
    try:
        with open('archive (4)/data/fashion_embeddings2.pkl', 'rb') as f:
            image_data = pickle.load(f)
        with open("archive (4)/data/fashion_embeddings_text.pkl", "rb") as f:
            text_data = pickle.load(f)
        return image_data, text_data
    except FileNotFoundError:
        st.error("Embedding files not found. Please ensure the 'archive (4)/data/' directory and its contents are correctly placed in your project folder.")
        return None, None

# Load the resources
resnet_model, text_model = load_models()
image_data, text_data = load_embeddings()

if image_data and text_data:
    image_embeddings = image_data['embeddings']
    image_paths = image_data['image_paths']
    image_categories = image_data['categories']

    text_embeddings = text_data["text_embeddings"]
    titles = text_data["texts"]
    text_image_paths = text_data["image_paths"]
    text_categories = text_data["categories"]

# ------------------------------
# STYLING & ASSETS
# ------------------------------
def load_lottieurl(url: str):
    """Fetches a Lottie JSON from a URL with error handling."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None

def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Playfair+Display:wght@700&display=swap');

    /* --- General Styles --- */
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #212529;
    }
    p, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        font-family: 'Lato', sans-serif;
        color: #495057;
    }

    /* --- Main Page Title --- */
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        text-align: left;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .main-subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 1.2rem;
        color: #D63384;
        text-align: left;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .feature-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #dee2e6;
        height: 100%;
    }
    .feature-card h3 {
        color: #D63384;
    }

    /* --- Recommendation Card Styling --- */
    .rec-card {
        background-color: #ffffff;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1.5rem;
        height: 100%;
    }
    .rec-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    .rec-card img {
        border-radius: 10px;
        width: 100%;
        height: 250px;
        object-fit: cover;
    }
    .rec-card .caption {
        font-family: 'Lato', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #212529;
        margin-top: 1rem;
        height: 40px; /* Fixed height for 2 lines */
        overflow: hidden;
    }
    .rec-card .similarity {
        font-size: 0.85rem;
        color: #D63384;
        font-weight: 700;
    }
    
    /* --- Custom Search Toggle --- */
    .search-toggle-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        background-color: #FFF;
        padding: 0.75rem;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        margin: 1rem auto 2rem auto;
        width: fit-content;
    }
    .search-toggle-button {
        background-color: transparent;
        color: #495057;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-family: 'Lato', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .search-toggle-button.active {
        background: linear-gradient(45deg, #D63384, #9E4784);
        color: white;
        box-shadow: 0 5px 15px rgba(214, 51, 132, 0.4);
    }

    /* --- File Uploader Styling --- */
    .stFileUploader {
        border: 2px dashed #D63384;
        background-color: #FADAEB;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
    }

    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# BACKEND HELPER FUNCTIONS
# ------------------------------
def get_image_embedding(img_input, model=resnet_model, target_size=(224, 224)):
    try:
        if isinstance(img_input, str): img = image.load_img(img_input, target_size=target_size)
        else: img = Image.open(img_input).convert("RGB").resize(target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x, verbose=0).flatten()
    except Exception: return None

def recommend_by_image(uploaded_img, embeddings, paths, categories, top_k=5):
    query_emb = get_image_embedding(uploaded_img)
    if query_emb is None: return []
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [(paths[i], categories[i], sims[i]) for i in top_idx]

def recommend_by_text(query, embeddings, texts, paths, categories, top_k=5):
    query_emb = text_model.encode([query])[0]
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [(paths[i], categories[i], texts[i], sims[i]) for i in top_idx]
def load_image_safe(img_path):
    img_path = Path(img_path)
    if not img_path.exists():
        img_path = Path.cwd() / img_path  # try relative to project root
    try:
        return Image.open(img_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="gray")


# ------------------------------
# UI RENDERING
# ------------------------------
local_css()

# --- Main Navigation ---
selected_page = option_menu(
    menu_title=None,
    options=["Home", "Discover"],
    icons=["gem", "sparkles"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fff", "border-radius": "12px", "box-shadow": "0 4px 12px rgba(0,0,0,0.05)"},
        "icon": {"color": "#D63384", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#f7f7f7", "font-family": "Lato"},
        "nav-link-selected": {"background-color": "#FADAEB", "font-weight": "700", "color": "#D63384"},
    }
)

# --- HOME PAGE ---
if selected_page == "Home":
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    with col1:
        st.markdown("<h1 class='main-title'>StyleSphere</h1>", unsafe_allow_html=True)
        st.markdown("<p class='main-subtitle'>YOUR PERSONAL AI STYLIST</p>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 1.1rem; margin-top: 1.5rem;">
        Welcome to the future of fashion discovery. StyleSphere uses advanced AI to understand your unique taste,
        curating a personalized wardrobe that's perfectly you. 
        <br><br>
        Whether you have a photo of a look you love or just an idea in mind, our system will find the perfect pieces to match your personal style sphere.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # <<< CHANGE HERE: Updated the Lottie animation URL to be more fashion-related
        lottie_json = load_lottieurl("https://lottie.host/4313f837-01a6-4171-9f3f-42a98f79f298/4Nbm2b5z1O.json")
        if lottie_json: st_lottie(lottie_json, height=450, speed=1, quality="high")

    st.markdown("<hr style='margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Unlock Your Personal Style in 3 Steps</h2>", unsafe_allow_html=True)
    st.markdown("<div><br></div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>🖼️ Find by Image</h3>
            <p>Have a photo of an outfit you love? Upload it, and our AI will instantly find similar items from our extensive collection, matching the color, pattern, and style.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>✍️ Find by Text</h3>
            <p>Describe what you're looking for, from a 'vintage leather jacket' to a 'flowy summer dress,' and let our intelligent stylist search and curate a selection just for you.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>✨ AI-Powered Curation</h3>
            <p>Our system goes beyond simple keywords. It understands the nuances of fashion to provide truly personalized recommendations that reflect your unique taste.</p>
        </div>
        """, unsafe_allow_html=True)


# --- RECOMMENDATIONS PAGE ---
if selected_page == "Discover" and image_data and text_data:
    st.markdown("<h1 style='text-align:center;'>Find Your Style Inspiration</h1>", unsafe_allow_html=True)
    
    # --- Custom Search Toggle ---
    if 'search_method' not in st.session_state:
        st.session_state.search_method = 'Text'

    # This is a hack to use st.columns as clickable containers for the custom buttons
    # The actual buttons are hidden by CSS.
    cols_button = st.columns([1, 0.25, 0.25, 1])
    with cols_button[1]:
        if st.button('Text', use_container_width=True): st.session_state.search_method = 'Text'
    with cols_button[2]:
        if st.button('Image', use_container_width=True): st.session_state.search_method = 'Image'

    text_class = "active" if st.session_state.search_method == 'Text' else ""
    image_class = "active" if st.session_state.search_method == 'Image' else ""
    
    st.markdown(f"""
        <div class="search-toggle-container">
            <button id="text-btn" class="search-toggle-button {text_class}">Search with Text</button>
            <button id="image-btn" class="search-toggle-button {image_class}">Search with Image</button>
        </div>
        <style>
            .st-emotion-cache-70k1qc {{ /* This targets the button container */
                display: none;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    # --- Conditional UI based on state ---
    if st.session_state.search_method == 'Text':
        with st.form(key='search_form'):
            search_query = st.text_input("Describe your desired look (e.g., 'black leather jacket' or 'floral summer dress')", key="search_query", label_visibility="collapsed", placeholder="Describe your desired look...")
            submit = st.form_submit_button("✨ Find My Style")
        if submit and search_query:
            with st.spinner('Styling your recommendations...'):
                results = recommend_by_text(search_query, text_embeddings, titles, text_image_paths, text_categories)
            if results:
                st.subheader(f"Inspired by '{search_query}'")
                # Using 5 columns for the 5 results
                cols = st.columns(len(results) or [1])
                for i, (path, cat, title, sim) in enumerate(results):
                    with cols[i]:
                        img = load_image_safe(path)
                        # Display image directly, card will be below
                        st.image(img, use_container_width=True)
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="caption">{title.replace("_", " ").title()}</div>
                            <div class="similarity">{cat.replace("_", " ").title()} | Match: {sim:.0%}</div>
                        </div>""", unsafe_allow_html=True)
            else: st.warning("Couldn't find matches. Try a different description!")

    elif st.session_state.search_method == 'Image':
        uploaded_file = st.file_uploader("Upload an image to find similar styles", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            st.image(uploaded_file, caption="Your Style Inspiration", width=250)
            with st.spinner('Analyzing your image...'):
                results = recommend_by_image(uploaded_file, image_embeddings, image_paths, image_categories)
            if results:
                st.subheader("Here are some similar styles we found:")
                # Using 5 columns for the 5 results
                cols = st.columns(len(results) or [1])
                for i, (path, cat, sim) in enumerate(results):
                    with cols[i]:
                        img = load_image_safe(path)
                        # Display image directly, card will be below
                        st.image(img, use_container_width=True)
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="caption">{cat.replace("_", " ").title()}</div>
                            <div class="similarity">Match: {sim:.0%}</div>
                        </div>""", unsafe_allow_html=True)
            else: st.warning("Couldn't find any recommendations for this image.")