import pandas as pd
import pytest
from highlight_extractor import HighlightExtractor

class DummyConfig:
    PRIORITY_KEYWORDS = {
        "politics": ["election", "president"],
        "sports": ["football", "cricket"]
    }
    HIGHLIGHTS_PER_CATEGORY = 2

def test_priority_detection():
    extractor = HighlightExtractor(priority_keywords=DummyConfig.PRIORITY_KEYWORDS)
    row = {"predicted_category": "politics", "title_lc": "new election announced"}
    assert extractor._is_priority(row) is True

def test_non_priority_detection():
    extractor = HighlightExtractor(priority_keywords=DummyConfig.PRIORITY_KEYWORDS)
    row = {"predicted_category": "politics", "title_lc": "new tax reforms introduced"}
    assert extractor._is_priority(row) is False

def test_extract_highlights_selects_priority_first():
    data = {
        "Title": ["Football world cup starts", "Tax reforms update", "Election results announced"],
        "predicted_category": ["sports", "politics", "politics"],
        "cluster_size": [50, 100, 10]
    }
    df = pd.DataFrame(data)
    extractor = HighlightExtractor(priority_keywords=DummyConfig.PRIORITY_KEYWORDS)
    highlights = extractor.extract_highlights(df, highlights_per_category=1)
    
    # Election should be picked even though it has lower cluster size
    assert "Election results announced".lower() in highlights['title_lc'].values
