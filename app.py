# file: rag_fake_news_app.py

import streamlit as st
import requests
import cohere
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Initialize components
co = cohere.Client("DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Fetch trusted news from NewsAPI
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"
def fetch_trusted_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] + ". " + (a.get("description") or "") for a in articles if a.get("title")]

# Build and return FAISS index with context
@st.cache_resource()
def build_rag_index():
    trusted_articles = fetch_trusted_articles()
    embeddings = embedder.encode(trusted_articles).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, trusted_articles

# RAG Query function
def rag_query(input_text, index, trusted_articles):
    query_embedding = embedder.encode([input_text]).astype('float32')
    D, I = index.search(query_embedding, k=3)
    context = "\n".join([trusted_articles[i] for i in I[0]])

    prompt = f"""
    Given the following real news reports:
    {context}

    Evaluate this input:
    "{input_text}"

    Is it likely to be fake or real? Provide a clear and reasoned answer.
    """

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=120,
        temperature=0.3
    )
    return response.generations[0].text.strip(), context

# Streamlit UI
st.set_page_config(page_title="RAG-Based Fake News Detector")
st.title("🧠 RAG-Powered Fake News Verifier (Live News Context)")

st.write("Enter a headline or news snippet to verify its credibility using real-time news context.")
user_input = st.text_area("News Headline or Snippet")

with st.spinner("Building semantic index from live news..."):
    index, corpus = build_rag_index()

if st.button("Verify"):
    if user_input.strip():
        verdict, context_used = rag_query(user_input, index, corpus)
        st.subheader("Verdict")
        st.write(verdict)

        with st.expander("Context Used from Live News Sources"):
            st.write(context_used)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.caption("Built with Cohere, FAISS, Sentence Transformers, and NewsAPI")
