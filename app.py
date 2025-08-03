import streamlit as st
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------- SETUP ----------------
st.set_page_config(page_title="📰 NewsRadar", layout="wide")

st.markdown("<h1 style='text-align: center;'>📰 NewsRadar</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Measure the truth in every word.</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- MODEL CACHING ----------------

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentiment_pipeline = load_sentiment_model()
embedding_model = load_embedding_model()

# ---------------- UTILITY FUNCTIONS ----------------

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return ""

def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])
    except Exception as e:
        return f"Error extracting from URL: {e}"

def sentiment_label(label, score):
    if label == "POSITIVE":
        return f"😊 Positive ({score:.2f})"
    elif label == "NEGATIVE":
        return f"☹️ Negative ({score:.2f})"
    return f"😐 Neutral ({score:.2f})"

def jaccard_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    if not union:
        return 0.0, []
    return len(intersection) / len(union), list(intersection)

def compare_articles(text1, text2, heading1, heading2):
    # Cosine Similarity (Semantic)
    embeddings = embedding_model.encode([text1, text2], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item() * 100

    # Jaccard Index
    jaccard_sim, common_words = jaccard_similarity(text1, text2)
    jaccard_sim *= 100

    # Sentiment (truncate to 512 characters for speed)
    sent1 = sentiment_pipeline(text1[:512])[0]
    sent2 = sentiment_pipeline(text2[:512])[0]

    # --- Output ---
    st.markdown(f"## 📊 Comparison Results: `{heading1}` vs `{heading2}`")

    st.markdown("### 🔢 Similarity Metrics")
    st.success(f"**Cosine Similarity:** `{cosine_sim:.2f}%`")
    st.info(f"**Jaccard Index:** `{jaccard_sim:.2f}%`")

    st.markdown("### 🧩 Common Words")
    st.write(common_words if common_words else "No common words found.")

    st.markdown("### 📈 Similarity Graph")
    fig, ax = plt.subplots()
    ax.bar(["Cosine Similarity", "Jaccard Index"], [cosine_sim, jaccard_sim], color=["deepskyblue", "mediumseagreen"])
    ax.set_ylim([0, 100])
    ax.set_ylabel("Similarity (%)")
    st.pyplot(fig)

    st.markdown("### 💬 Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {heading1}")
        st.markdown(f"- Mood: {sentiment_label(sent1['label'], sent1['score'])}")
    with col2:
        st.markdown(f"#### {heading2}")
        st.markdown(f"- Mood: {sentiment_label(sent2['label'], sent2['score'])}")

    st.markdown("### 🧠 Sentiment Comparison Pie")
    fig2, ax2 = plt.subplots()
    labels = [heading1, heading2]
    values = [sent1['score'], sent2['score']]
    ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['salmon', 'skyblue'])
    ax2.axis('equal')
    st.pyplot(fig2)

# ---------------- INPUT SECTION ----------------

tab1, tab2, tab3 = st.tabs(["✍️ Paste Text", "📁 Upload File", "🌐 Compare from URL"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        h1 = st.text_input("Title for Article 1", "Article 1")
        a1 = st.text_area("Paste content of Article 1", height=250)
    with col2:
        h2 = st.text_input("Title for Article 2", "Article 2")
        a2 = st.text_area("Paste content of Article 2", height=250)

    if st.button("🔍 Compare Pasted"):
        if a1.strip() and a2.strip():
            compare_articles(a1, a2, h1, h2)
        else:
            st.warning("🚨 Please paste both articles.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        h1f = st.text_input("Title for File 1", "File 1")
        file1 = st.file_uploader("Upload File 1", type=["txt", "pdf"])
    with col2:
        h2f = st.text_input("Title for File 2", "File 2")
        file2 = st.file_uploader("Upload File 2", type=["txt", "pdf"])

    if st.button("📂 Compare Files"):
        if file1 and file2:
            t1 = extract_text(file1)
            t2 = extract_text(file2)
            compare_articles(t1, t2, h1f, h2f)
        else:
            st.warning("🚨 Upload both files.")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        h1url = st.text_input("Title for URL 1", "URL 1")
        url1 = st.text_input("Paste URL for Article 1")
    with col2:
        h2url = st.text_input("Title for URL 2", "URL 2")
        url2 = st.text_input("Paste URL for Article 2")

    if st.button("🌐 Compare URLs"):
        if url1 and url2:
            a1 = extract_text_from_url(url1)
            a2 = extract_text_from_url(url2)
            if "Error" not in a1 and "Error" not in a2:
                compare_articles(a1, a2, h1url, h2url)
            else:
                st.error("🚨 Failed to fetch content from one or both URLs.")
        else:
            st.warning("🚨 Enter both URLs.")
