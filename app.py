import cohere
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔐 API Keys
COHERE_API_KEY = "DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp"
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"

# Initialize clients
co = cohere.Client(COHERE_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔎 Fetch news titles + descriptions + URLs
def fetch_news_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    # Return (text, url) tuples
    return [(a["title"] + ". " + (a.get("description") or ""), a.get("url")) for a in articles if a.get("title")]

# 🧠 Perform RAG-based analysis
def rag_check(user_input, corpus, urls):
    embeddings = model.encode([c for c in corpus]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = model.encode([user_input]).astype("float32")
    _, I = index.search(query_embedding, k=3)
    matched_texts = [corpus[i] for i in I[0]]
    matched_urls = [urls[i] for i in I[0]]

    context = "\n".join(matched_texts)
    prompt = f"""
You are a fact-checking assistant.

Real News Context:
{context}

News to Verify:
\"{user_input}\"

Is this news likely to be fake or real? Explain briefly.
"""
    response = co.generate(prompt=prompt, model="command", max_tokens=120)
    return response.generations[0].text.strip(), matched_texts, matched_urls

# 🚀 Run the system
user_news = input("📝 Enter a news headline or short article to verify: ")

trusted_articles = fetch_news_articles()
corpus = [t[0] for t in trusted_articles]
urls = [t[1] for t in trusted_articles]

verdict, articles_used, article_links = rag_check(user_news, corpus, urls)

# 📤 Output verdict + article sources
print("\n🤖 RAG Verdict:\n")
print(verdict)
print("\n🔗 Related Trusted News Articles:")
for i, (text, link) in enumerate(zip(articles_used, article_links)):
    print(f"\nArticle {i+1}: {text}\nSource: {link}")
