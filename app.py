import streamlit as st
import matplotlib.pyplot as plt
import model  # Import your backend logic
import tempfile

# ---------------- SETUP ----------------
st.set_page_config(page_title="📰 NewsMatch", layout="wide")
st.markdown("<h1 style='text-align: center;'>📰 NewsMatch</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Read in between the lines.</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- FRONTEND FUNCTIONS ----------------

def sentiment_label(label, score):
    if label == "POSITIVE":
        return f"😊 Positive ({score:.2f})"
    elif label == "NEGATIVE":
        return f"☹ Negative ({score:.2f})"
    return f"😐 Neutral ({score:.2f})"

def compare_articles(text1, text2, heading1, heading2):
    cosine_sim = model.calculate_cosine_similarity(text1, text2) * 100
    jaccard_sim = model.calculate_jaccard_index(text1, text2) * 100
    common_words = model.find_common_words(text1, text2)
    polarity1, label1 = model.perform_sentiment_analysis(text1)
    polarity2, label2 = model.perform_sentiment_analysis(text2)

    # --- Output ---
    st.markdown(f"## 📊 Comparison Results: {heading1} vs {heading2}")

    st.markdown("### 🔢 Similarity Metrics")
    st.success(f"*Cosine Similarity:* {cosine_sim:.2f}%")
    st.info(f"*Jaccard Index:* {jaccard_sim:.2f}%")

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
        st.markdown(f"- Mood: {sentiment_label(label1, abs(polarity1))}")
    with col2:
        st.markdown(f"#### {heading2}")
        st.markdown(f"- Mood: {sentiment_label(label2, abs(polarity2))}")

    st.markdown("### 🧠 Sentiment Comparison Pie")
    fig2, ax2 = plt.subplots()
    labels = [heading1, heading2]
    values = [abs(polarity1), abs(polarity2)]
    ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['salmon', 'skyblue'])
    ax2.axis('equal')
    st.pyplot(fig2)

# ---------------- INPUT SECTION ----------------

tab1, tab2, tab3 = st.tabs(["✍ Paste Text", "📁 Upload File", "🌐 Compare from URL"])

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
            with tempfile.NamedTemporaryFile(delete=False) as temp1:
                temp1.write(file1.read())
            with tempfile.NamedTemporaryFile(delete=False) as temp2:
                temp2.write(file2.read())
            t1 = model.extract_text_from_pdf(temp1.name) if file1.name.endswith(".pdf") else file1.getvalue().decode("utf-8")
            t2 = model.extract_text_from_pdf(temp2.name) if file2.name.endswith(".pdf") else file2.getvalue().decode("utf-8")
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
            a1 = model.extract_text_from_url(url1)
            a2 = model.extract_text_from_url(url2)
            if not a1.lower().startswith("error") and not a2.lower().startswith("error"):
                compare_articles(a1, a2, h1url, h2url)
            else:
                st.error("🚨 Failed to fetch content from one or both URLs.")
        else:
            st.warning("🚨 Enter both URLs.")
