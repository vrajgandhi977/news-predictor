# file: app.py (Streamlit version of RAG news verifier)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for Streamlit compatibility

import streamlit as st
import cohere
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔑 API Keys
COHERE_API_KEY = "DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp"
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"

# 🤖 Initialize model + Cohere
co = cohere.Client(COHERE_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📥 Fetch live news
@st.cache_data(ttl=600)
def fetch_news_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] + ". " + (a.get("description") or "") for a in articles if a.get("title")]

# 🔍 Check user news with RAG
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
    return response.generations[0].text.strip(), context

# 🌐 Streamlit UI
st.set_page_config(page_title="Fake News Detector (RAG + Cohere)")
st.title("📰 Real-Time Fake News Verifier")
st.caption("Powered by Cohere, FAISS, and Sentence Transformers")

user_news = st.text_area("Paste a news headline or article snippet:")

if st.button("Check Authenticity"):
    with st.spinner("Fetching trusted news and analyzing..."):
        news_articles = fetch_news_articles()
        if not news_articles:
            st.error("❌ Failed to fetch news. Check your API key or try again later.")
        else:
            verdict, context = rag_check(user_news, news_articles)
            st.subheader("🤖 Verdict")
            st.success(verdict)

            st.markdown("---")
            st.subheader("🔎 Matched Trusted Context")
            st.text(context)

st.markdown("---")
st.caption("Tip: Refresh every 10 mins for live news updates")
