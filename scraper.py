from dotenv import load_dotenv
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import praw  # ✅ New Reddit library

# Load .env
load_dotenv(dotenv_path=".env")
print("Bearer Token:", os.getenv("BEARER_TOKEN"))

# --- BBC News Scraper ---
def scrape_bbc():
    print("Scraping BBC News...")
    url = "https://www.bbc.com/news"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [h.get_text(strip=True) for h in soup.select("h2, a") if h.get_text(strip=True)]
    headlines = list(dict.fromkeys(headlines))[:30]
    print(f"✅ Collected {len(headlines)} BBC headlines")
    return pd.DataFrame({"text": headlines, "label": "real"})


# --- Reddit Scraper using PRAW ---
def scrape_reddit():
    print("Scraping Reddit posts (via Reddit API)...")

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

    posts = []
    for submission in reddit.subreddit("news+worldnews+politics").search("fake news", limit=30):
        posts.append(submission.title)

    print(f"✅ Collected {len(posts)} Reddit posts")
    return pd.DataFrame({"text": posts, "label": "fake"})


# --- Main ---
if __name__ == "__main__":
    try:
        news_df = scrape_bbc()
    except Exception as e:
        print(f"⚠️ Error scraping BBC: {e}")
        news_df = pd.DataFrame(columns=["text", "label"])

    try:
        reddit_df = scrape_reddit()
    except Exception as e:
        print(f"⚠️ Error scraping Reddit: {e}")
        reddit_df = pd.DataFrame(columns=["text", "label"])

    df = pd.concat([news_df, reddit_df], ignore_index=True)
    df.to_csv("scraped_data.csv", index=False)
    print("✅ Data saved to scraped_data.csv")
