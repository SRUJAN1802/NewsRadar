import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load once: Hugging Face models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 1. Manual Text Join
def extract_text_manual(text_lines):
    return "\n".join(text_lines)

# 2. Extract from PDF
def extract_text_from_pdf(file_path):
    all_text = ""
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        for page in pdf.pages:
            page_content = page.extract_text()
            if page_content:
                all_text += page_content
    return all_text

# 3. Extract from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = ' '.join([p.get_text() for p in paragraphs])
        return full_text
    except Exception as error:
        return f"Error while fetching URL content: {error}"

# 4. Hugging Face-based Semantic Similarity (Cosine)
def calculate_cosine_similarity(text1, text2):
    embeddings = embedding_model.encode([text1, text2], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()  # Returns a float

# 5. Jaccard Index
def calculate_jaccard_index(text1, text2):
    words_in_text1 = set(text1.lower().split())
    words_in_text2 = set(text2.lower().split())
    common_words = words_in_text1.intersection(words_in_text2)
    all_words = words_in_text1.union(words_in_text2)
    return len(common_words) / len(all_words) if all_words else 0

# 6. Hugging Face-based Sentiment
def perform_sentiment_analysis(text):
    result = sentiment_pipeline(text[:512])  # Truncate to 512 tokens if needed
    label = result[0]['label']
    score = result[0]['score']
    # Map label to numeric polarity like TextBlob (-1 to 1)
    if label == "POSITIVE":
        polarity = score
    elif label == "NEGATIVE":
        polarity = -score
    else:
        polarity = 0
    return polarity, label

# 7. Common Words
def find_common_words(text1, text2):
    words_text1 = set(text1.lower().split())
    words_text2 = set(text2.lower().split())
    return list(words_text1.intersection(words_text2))
