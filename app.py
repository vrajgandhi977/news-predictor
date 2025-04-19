import streamlit as st
import requests
import cohere
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Setup API keys
co = cohere.Client("DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp")
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"

# Cache model loading
@st.cache_resource()
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Cache live news fetch (refresh every 10 minutes)
@st.cache_data(ttl=600)
def fetch_trusted_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [(a["title"] + ". " + (a.get("description") or ""), a.get("url", "")) for a in articles if a.get("title")]

# Cache FAISS index
@st.cache_resource()
def build_rag_index(corpus):
    embeddings = model.encode(corpus).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# RAG Query function
def rag_query(input_text, index, trusted_articles, urls):
    query_embedding = model.encode([input_text]).astype('float32')
    D, I = index.search(query_embedding, k=3)
    matched_articles = [trusted_articles[i] for i in I[0]]
    matched_urls = [urls[i] for i in I[0]]
    context = "\n".join(matched_articles)

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
    return response.generations[0].text.strip(), matched_articles, matched_urls

# Streamlit UI
st.set_page_config(page_title="RAG-Based Fake News Detector")
st.title("🧠 RAG-Powered Fake News Verifier (Live News Context)")

st.write("Enter a headline or news snippet to verify its credibility using real-time news context.")
user_input = st.text_area("News Headline or Snippet")

with st.spinner("Fetching news and preparing index..."):
    raw_articles = fetch_trusted_articles()
    trusted_articles = [t[0] for t in raw_articles]
    urls = [t[1] for t in raw_articles]
    index, _ = build_rag_index(trusted_articles)

if st.button("Verify"):
    if user_input.strip():
        verdict, context_used, links = rag_query(user_input, index, trusted_articles, urls)
        st.subheader("Verdict")
        st.write(verdict)

        with st.expander("Context Used from Live News Sources"):
            for i in range(len(context_used)):
                st.markdown(f"**Article {i+1}:** {context_used[i]}")
                st.markdown(f"🔗 [View Source]({links[i]})")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.caption("Built with Cohere, FAISS, Sentence Transformers, and NewsAPI")
