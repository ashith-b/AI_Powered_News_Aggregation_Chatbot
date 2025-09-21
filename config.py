import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"


class Config:
    """Application configuration"""
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Data paths
    NEWS_CSV_PATH = 'datasets/Aggregated News Dataset - Sheet1.csv'
    CLASSIFIED_ARTICLES_CSV_PATH = 'datasets/classified_articles.csv'
    HIGHLIGHTS_CSV_PATH = 'datasets/daily_highlights.csv'
    
    
    # Categories
    NEWS_CATEGORIES = {
        "sports": "Sports news about matches, athletes, teams and sporting events",
        "finance": "Financial markets, stocks, economy and business news",
        "politics": "Political news, elections, government policies",
        "lifestyle": "Lifestyle, health, travel, food and culture",
        "music": "Music artists, albums, concerts and industry news"
    }
    
    # Priority keywords by category
    PRIORITY_KEYWORDS = {
        "sports": ["breaking", "championship", "cup final", "olympics", "transfer",
                "injury update", "record broken", "doping scandal", "last minute",
                "trophy", "victory", "defeat", "comeback"],
        "finance": ["alert", "breaking", "market crash", "rate hike", "earnings report",
                    "recession", "inflation", "fed decision", "stock plunge", "merger",
                    "bankruptcy", "crypto crash", "dividend", "short squeeze", "ipo"],
        "politics": ["election", "scandal", "resignation", "law passed", "protest",
                    "breaking", "impeachment", "summit", "sanctions", "diplomatic crisis",
                    "war", "treaty", "vote result", "corruption", "speech"],
        "lifestyle": ["viral", "trending", "recipe", "hack", "celebrity",
                    "wedding", "divorce", "pregnancy", "royal family", "makeover",
                    "controversial", "banned", "recall", "study finds"],
        "music": ["tour announced", "new album", "grammy", "controversy", "feud",
                "concert disaster", "chart-topping", "streaming record", "comeback",
                "breakup", "collaboration", "lyrics decoded", "canceled", "viral hit"]
    }
    
    # UMAP & HDBSCAN parameters
    UMAP_N_COMPONENTS = 5
    UMAP_RANDOM_STATE = 42
    HDBSCAN_MIN_CLUSTER_SIZE = 15
    
    # Highlights settings
    HIGHLIGHTS_PER_CATEGORY = 5
    
