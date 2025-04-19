import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 💡 Hide GPU from torch completely (Streamlit-safe)

import cohere
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔑 API Keys
COHERE_API_KEY = "DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp"
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"

co = cohere.Client(COHERE_API_KEY)

# ✅ Load model normally (no .to('cpu'))
model = SentenceTransformer("all-MiniLM-L6-v2")  # Torch handles CPU fallback internally

# 📥 Fetch live news headlines
def fetch_news_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] + ". " + (a.get("description") or "") for a in articles if a.get("title")]

# 🧠 RAG check
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
\"{user_input}\"

Is this news likely to be fake or real? Explain briefly.
"""

    response = co.generate(prompt=prompt, model="command", max_tokens=120)
    return response.generations[0].text.strip()

# ▶️ Run
user_news = input("📝 Enter a news headline or short article to verify: ")

trusted_news = fetch_news_articles()
verdict = rag_check(user_news, trusted_news)

# 📤 Output
print("\n🤖 RAG Verdict:\n")
print(verdict)
