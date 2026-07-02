import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Page Settings
st.set_page_config(
    page_title="Multi-Language Detector AI",
    page_icon="🌍",
    layout="centered"
)

# Custom Glassmorphism CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #FF4B4B, #4F46E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        text-align: center;
        color: #71717a;
        margin-bottom: 30px;
    }
    .status-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f4f4f5;
        border-left: 5px solid #4F46E5;
        margin-bottom: 20px;
    }
    .result-card {
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #e0e7ff, #e0f2fe);
        border: 1px solid #bae6fd;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌍 Global Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An advanced ML model that predicts languages accurately in real-time.</div>',
            unsafe_allow_html=True)


# 2. Cache Model Training to avoid training on every re-run
@st.cache_resource
def train_and_get_pipeline():
    try:
        # Load dataset
        df = pd.read_csv('language.csv')
        df.dropna(subset=['Text', 'language'], inplace=True)

        # Prevent PyArrow / Indexing structural mismatches by converting to native lists
        x = df['Text'].astype(str).tolist()
        y = df['language'].astype(str).tolist()
        y = np.array(y)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

        # Machine Learning Pipeline
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])

        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)

        return pipeline, accuracy
    except FileNotFoundError:
        return None, 0
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return None, 0


# Load Model Engine
with st.spinner("🧠 Booting up AI Engine & Training Dataset... Please wait."):
    pipeline, accuracy = train_and_get_pipeline()

if pipeline is None:
    st.error(
        "❌ **'language.csv'** file not found! Please check if the dataset is placed in the correct project directory.")
else:
    # Sidebar Info Panel
    st.sidebar.header("📊 Model Metrics")
    st.sidebar.markdown(f"""
    <div class='status-card'>
        <strong>Model Type:</strong><br>Multinomial Naive Bayes<br><br>
        <strong>Tested Accuracy:</strong><br><span style='color:#4F46E5; font-size:20px; font-weight:bold;'>{accuracy * 100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.info(
        "💡 Tip: You can paste a single sentence or long multilingual paragraphs to test the evaluation pipeline.")

    # Main User Interface
    user_input = st.text_area("✍️ Paste your multilingual text here:", height=180,
                              placeholder="E.g., Comment allez-vous? / Hola, ¿cómo estás? / How are you? ...")

    if st.button("Detect Language 🚀", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before processing!")
        else:
            # Prediction
            predicted_lang = pipeline.predict([user_input])[0]

            # Confidence Calculation
            probs = pipeline.predict_proba([user_input])
            confidence = np.max(probs) * 100

            # Show Clean Visual Results
            st.markdown(f"""
                <div class='result-card'>
                    <span style='font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #4338ca; font-weight: 600;'>Detected Language</span>
                    <h2 style='margin: 10px 0; color: #1e1b4b; font-size: 36px;'>🎉 {predicted_lang.upper()}</h2>
                    <span style='color: #0369a1; font-weight: 500;'>AI Confidence Score: {confidence:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)

            # Celebration effect if confidence is high
            if confidence > 75:
                st.balloons()