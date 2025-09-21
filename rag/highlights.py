import re
import pandas as pd
from loguru import logger
from config import Config

class HighlightExtractor:
    """Extract important news highlights based on priority keywords and clustering"""
    
    def __init__(self, priority_keywords=Config.PRIORITY_KEYWORDS):
        self.priority_keywords = priority_keywords
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for each category's priority keywords"""
        self.keyword_patterns = {}
        for cat, keywords in self.priority_keywords.items():
            pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
            self.keyword_patterns[cat] = re.compile(pattern)
    
    def _is_priority(self, row):
        """Check if an article is a priority based on keywords in title
        
        Args:
            row: DataFrame row with 'title_lc' and 'predicted_category'
            
        Returns:
            bool: True if article contains priority keywords
        """
        category = row['predicted_category']
        title = row['title_lc']
        
        pattern = self.keyword_patterns.get(category)
        if pattern:
            return bool(pattern.search(title))
        return False
    
    def extract_highlights(self, df, highlights_per_category=Config.HIGHLIGHTS_PER_CATEGORY):
        """Extract top highlights for each category
        
        Args:
            df: DataFrame with classified and clustered articles
            highlights_per_category: Number of highlights to extract per category
            
        Returns:
            DataFrame: DataFrame with top highlights
        """
        df = df.copy()
        
        # Lower-case titles for keyword matching
        df['title_lc'] = df['Title'].str.lower()
        
        # Flag priority articles
        df['is_priority'] = df.apply(self._is_priority, axis=1)
        
        # Compute highlight score: keyword hits get highest priority, then cluster size
        df['highlight_score'] = df['is_priority'].astype(int) * 1000 + df['cluster_size']
        
        # Extract top highlights per category
        highlights = (
            df.sort_values(['predicted_category', 'highlight_score'], ascending=[True, False])
            .groupby('predicted_category')
            .head(highlights_per_category)
            .reset_index(drop=True)
        )
        
        logger.info(f"Extracted {len(highlights)} highlights across {df['predicted_category'].nunique()} categories")
        return highlights 