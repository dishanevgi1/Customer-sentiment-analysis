import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
import spacy.displacy
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI-Powered Sentiment & Linguistic Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD RESOURCES (Cache them for speed) ---

@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# 2. Load ANN Resources (Model + Tokenizer + Encoder)
@st.cache_resource
def load_ann_resources():
    try:
        # Load Model
        model = tf.keras.models.load_model('sentiment_model.h5')
        
        # Load Tokenizer (Converts Text -> Numbers)
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Load Encoder (Converts Numbers -> "Positive/Negative" Labels)
        with open('encoder.pickle', 'rb') as handle:
            encoder = pickle.load(handle)
            
        return model, tokenizer, encoder
    except Exception as e:
        # If files are missing, return None so we can show a friendly error later
        return None, None, None

nlp = load_nlp_model()
ann_model, tokenizer, encoder = load_ann_resources()

# 3. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Customer_Sentiment.csv")
    return df

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error("Error: 'Customer_Sentiment.csv' not found. Please ensure the file is in the same directory.")
    data_loaded = False

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 30px; font-weight: bold; color: #4B4B4B;}
    .sub-header {font-size: 20px; font-weight: bold; color: #6C757D;}
    .card {background-color: #F8F9FA; padding: 20px; border-radius: 10px; border: 1px solid #E9ECEF; margin-bottom: 20px;}
    .highlight {color: #007BFF; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title(" Configuration")
mode = st.sidebar.radio("Select Input Mode:", ["Analyze Dataset Reviews", "Custom Text Input"])

# --- MAIN LAYOUT ---
st.markdown('<div class="main-header"> NLP & ANN Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown("Perform deep linguistic analysis (Syntax/Semantics) and AI Prediction (ANN).")
st.markdown("---")

selected_text = ""
meta_info = {}

if mode == "Analyze Dataset Reviews" and data_loaded:
    st.sidebar.subheader("Filter Data")
    platform = st.sidebar.selectbox("Select Platform", df['platform'].unique())
    
    # Filter by platform
    filtered_df = df[df['platform'] == platform]
    
    # Select a specific review
    review_index = st.sidebar.selectbox(
        "Select a Review ID:", 
        filtered_df.index,
        format_func=lambda x: f"ID {filtered_df.loc[x, 'customer_id']} ({filtered_df.loc[x, 'sentiment']})"
    )
    
    selected_row = filtered_df.loc[review_index]
    selected_text = selected_row['review_text']
    
    # Meta info for display
    meta_info = {
        "Platform": selected_row['platform'],
        "Rating": f"{selected_row['customer_rating']} / 5",
        "Region": selected_row['region'],
        "Original Sentiment": selected_row['sentiment']
    }

elif mode == "Custom Text Input":
    selected_text = st.text_area("Enter text to analyze:", "The product quality was amazing, but the delivery was late.", height=150)

# --- ANALYSIS SECTION ---
if selected_text:
    
    # 1. Display Selected Text & Meta Info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header"> Text Under Analysis</div>', unsafe_allow_html=True)
        st.info(f'"{selected_text}"')
        
    with col2:
        if meta_info:
            st.markdown('<div class="sub-header"> Meta Data</div>', unsafe_allow_html=True)
            with st.container():
                c1, c2 = st.columns(2)
                c1.metric("Rating", meta_info["Rating"])
                c2.metric("Region", meta_info["Region"])
                st.caption(f"Platform: {meta_info['Platform']}")

    st.markdown("---")

    # Run NLP Pipeline (For Tabs 1 & 2)
    doc = nlp(selected_text)
    blob = TextBlob(selected_text)

    # --- TABS: Syntax, Semantics, ANN ---
    tab1, tab2, tab3 = st.tabs(["Syntax Analysis", " Semantic Analysis", " ANN Prediction (Task 3)"])

    # --- TAB 1: SYNTAX ANALYSIS ---
    with tab1:
        st.markdown("### 1. Part-of-Speech (POS) Tagging")
        st.write("Breaking down the sentence into grammatical components.")
        
        pos_data = []
        for token in doc:
            pos_data.append({
                "Token": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_,
                "Tag": token.tag_,
                "Dependency": token.dep_
            })
        
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

        st.markdown("### 2. Dependency Parsing")
        try:
            html = spacy.displacy.render(doc, style="dep", options={"compact": True, "bg": "#ffffff", "distance": 100})
            st.components.v1.html(html, height=300, scrolling=True)
        except Exception as e:
            st.warning("Visualization could not be rendered.")

    # --- TAB 2: SEMANTIC ANALYSIS ---
    with tab2:
        col_sem1, col_sem2 = st.columns(2)
        
        with col_sem1:
            st.markdown("### 1. Sentiment Score")
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0:
                sent_label = "Positive "
                color = "green"
            elif polarity < 0:
                sent_label = "Negative "
                color = "red"
            else:
                sent_label = "Neutral "
                color = "gray"
                
            st.metric(label="Sentiment Polarity", value=f"{polarity:.2f}", delta=sent_label)
            st.progress((polarity + 1) / 2)
            st.metric(label="Subjectivity", value=f"{subjectivity:.2f}")

        with col_sem2:
            st.markdown("### 2. Named Entity Recognition")
            ents = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]
            
            if ents:
                st.dataframe(pd.DataFrame(ents, columns=["Entity", "Label", "Description"]), use_container_width=True)
            else:
                st.info("No named entities found.")
                
        st.markdown("### 3. Noun Chunks")
        chunks = [chunk.text for chunk in doc.noun_chunks]
        st.write(f"Key phrases: **{', '.join(chunks)}**")

    # --- TAB 3: ANN PREDICTION  ---
    with tab3:
        st.markdown("### 🤖 Artificial Neural Network Prediction")
        st.write("This tab uses your Deep Learning model to predict sentiment.")

        # Check if all files are loaded
        if ann_model is None:
            st.error(" Model file 'sentiment_model.h5' not found.")
        elif tokenizer is None:
            st.error(" Tokenizer file 'tokenizer.pickle' not found.")
        elif encoder is None:
            st.error(" Encoder file 'encoder.pickle' not found.")
        else:
            if st.button("Predict with ANN"):
                try:
                    # --- PREPROCESSING ---
                    # 1. Convert text to sequence of integers
                    sequences = tokenizer.texts_to_sequences([selected_text])
                    
                    # 2. Pad the sequence (MUST match Colab max_length=100)
                    padded_input = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
                    
                    # --- PREDICTION ---
                    prediction = ann_model.predict(padded_input)
                    
                    # --- INTERPRET RESULT (Softmax / Multi-class) ---
                    # Find the index with highest probability (e.g., 2)
                    predicted_class_index = np.argmax(prediction)
                    confidence_score = np.max(prediction)
                    
                    # Convert index back to text (e.g., 2 -> "Positive")
                    predicted_label = encoder.inverse_transform([predicted_class_index])[0]
                    
                    st.write(f"**Prediction:** {predicted_label}")
                    st.write(f"**Confidence Score:** {confidence_score:.4f}")
                    
                    # Display Result with Colors
                    label_str = str(predicted_label).lower()
                    if "positive" in label_str:
                        st.success(f"Result: {predicted_label} ")
                    elif "negative" in label_str:
                        st.error(f"Result: {predicted_label} ")
                    else:
                        st.info(f"Result: {predicted_label} ")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

else:
    st.info("Please select a review from the sidebar or enter custom text to begin analysis.")