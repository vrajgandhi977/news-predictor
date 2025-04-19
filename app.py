import cohere
import requests

# ✅ Setup your keys
COHERE_API_KEY = "DvaKcnKcvC6LpDmfey3ZmeK3DQ8KXV2TUYaNz7tp"
NEWS_API_KEY = "e240a31a6fa94a77a5e52be5da2dd0a0"

co = cohere.Client(COHERE_API_KEY)

# ✅ Fetch latest news from trusted sources
def fetch_news_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] + ". " + (a.get("description") or "") for a in articles if a.get("title")]

# ✅ Direct LLM-based fake news check
def check_news_with_llm(user_input):
    prompt = f"""
You are a fact-checking assistant.

News to Verify:
\"{user_input}\"

Based on your knowledge and reasoning, is this news likely to be fake or real? Explain briefly.
"""
    response = co.generate(prompt=prompt, model="command", max_tokens=120)
    return response.generations[0].text.strip()

# ✅ Input from user
user_news = input("📝 Enter a news headline or short article to verify: ")

# ✅ Direct analysis (no matching)
verdict = check_news_with_llm(user_news)

# ✅ Output
print("\n🤖 LLM Verdict:\n")
print(verdict)
