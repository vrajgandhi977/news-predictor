# file: app.py

import os
import streamlit as st
import cohere
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔒 Setup API keys
COHERE_API_KEY = "DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp"
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"

# 🔧 Initialize Cohere client and sentence transformer
co = cohere.Client(COHERE_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📥 Fetch news articles from NewsAPI
@st.cache_data(ttl=600)
def fetch_news_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] + ". " + (a.get("description") or "") for a in articles if a.get("title")]

# 🔍 Perform RAG-style verification (context removed from display)
def rag_check(user_input, corpus):
    corpus_embeddings = model.encode(corpus).astype("float32")
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    query_embedding = model.encode([user_input]).astype("float32")
    _, I = index.search(query_embedding, k=3)
    context = "\n".join([corpus[i] for i in I[0]])

    prompt = f"""
You are a fact-checking assistant.

Real News Context:
{context}

News to Verify:
"{user_input}"

Is this news likely to be fake or real? Explain briefly.
"""

    response = co.generate(prompt=prompt, model="command", max_tokens=120)
    return response.generations[0].text.strip()

# 🚀 Streamlit UI setup
st.set_page_config(page_title="Fake News Verifier", layout="centered")
st.title("📰 Fake News Verifier using RAG + Cohere")

st.markdown("Enter a news headline or short story snippet to analyze its credibility.")
user_news = st.text_area("📝 News Input", height=150)

if st.button("Verify News"):
    if not user_news.strip():
        st.warning("Please enter a news snippet or headline.")
    else:
        with st.spinner("Analyzing using RAG and Cohere..."):
            corpus = fetch_news_articles()
            verdict = rag_check(user_news, corpus)

        st.subheader("🤖 Verdict")
        st.success(verdict)

st.markdown("---")
st.caption("Built with Streamlit, Cohere, Sentence Transformers, FAISS, and NewsAPI")
