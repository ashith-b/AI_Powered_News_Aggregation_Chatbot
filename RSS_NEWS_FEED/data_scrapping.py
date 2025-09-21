import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

# List of RSS feed URLs
RSS_FEEDS = [
    "https://www.abc.net.au/news/feed/2942460/rss.xml",  # ABC Sport
    "https://www.abc.net.au/news/feed/45910/rss.xml",    # ABC Politics
    "https://www.abc.net.au/news/feed/52278/rss.xml",     # ABC lifestyle
    "https://www.cbsnews.com/latest/rss/politics",
    "https://www.smh.com.au/rss/lifestyle.xml",
    "https://www.smh.com.au/rss/sport.xml",
    "https://www.smh.com.au/rss/politics/federal.xml",
    "https://www.smh.com.au/rss/business.xml",
    "https://www.9news.com/feeds/syndication/rss/sports",
    "https://www.9news.com/feeds/syndication/rss/news/politics",
    "https://www.news.com.au/content-feeds/latest-news-lifestyle/",
    ""
]

def scrape_rss_feed(url):
    """Scrape one RSS feed and return a list of articles"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    root = ET.fromstring(response.content)
    channel = root.find("channel")
    publication_title = channel.find("title").text if channel.find("title") is not None else "Unknown"

    records = []
    for item in channel.findall("item"):
        title = item.find("title").text if item.find("title") is not None else None
        link = item.find("link").text if item.find("link") is not None else None
        description = item.find("description").text.strip() if item.find("description") is not None else None
        author = item.find("{http://purl.org/dc/elements/1.1/}creator").text if item.find("{http://purl.org/dc/elements/1.1/}creator") is not None else None
        pub_date = item.find("pubDate").text if item.find("pubDate") is not None else None

        # Extract image if available
        media_content = item.find(".//{http://search.yahoo.com/mrss/}content")
        image_url = media_content.attrib.get("url") if media_content is not None else None

        records.append({
            "Date Scraped": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Title": title,
            "news_summary": description,
            "news_card_image": image_url,
            "Date Published": pub_date,
            "Link": link,
            "Publication": publication_title,
            "Author": author
        })

    return records


def scrape_multiple_feeds(feed_urls, output_file="Config.NEWS_CSV_PATH"):
    all_records = []
    for url in feed_urls:
        print(f"🔎 Scraping {url} ...")
        all_records.extend(scrape_rss_feed(url))

    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} articles from {len(feed_urls)} feeds into {output_file}")


if __name__ == "__main__":
    scrape_multiple_feeds(RSS_FEEDS)
