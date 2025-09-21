import pandas as pd
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from .faiss_vectore_store import init_faiss_with_categories, get_langchain_embeddings

class NewsClassifier:
    """Classify news articles into predefined categories using embeddings"""
    
    def __init__(self):
        self.embeddings = get_langchain_embeddings()
        self.categories_collection = init_faiss_with_categories()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def classify_text(self, text):
        """Classify a single text using the categories collection
        
        Args:
            text: The article text to classify
            
        Returns:
            tuple: (category, similarity_score)
        """
        emb = self.embeddings.embed_query(text)
        results = self.categories_collection.query(
            query_embeddings=[emb], 
            n_results=1
        )
        return results['ids'][0][0], results['distances'][0][0]
    
    def classify_dataframe(self, df, text_column='text'):
        """Classify all articles in a dataframe
        
        Args:
            df: DataFrame with articles
            text_column: Column containing the text to classify
            
        Returns:
            DataFrame: Original dataframe with added predicted_category and similarity columns
        """
        logger.info(f"Classifying {len(df)} articles")
        
        # Apply classification to each row
        results = []
        for text in df[text_column].tolist():
            category, similarity = self.classify_text(text)
            results.append((category, similarity))
        
        # Add results to dataframe
        df = df.copy()
        df['predicted_category'], df['similarity'] = zip(*results)
        
        logger.info(f"Classified {len(df)} articles into categories")
        return df
    
    def batch_embed_texts(self, texts):
        """Create embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            list: Embeddings for each text
        """
        logger.info(f"Creating embeddings for {len(texts)} texts")
        return self.embeddings.embed_documents(texts)

def prepare_article_text(df):
    """Prepare article text by combining title and content
    
    Args:
        df: DataFrame with articles (must contain Title and news_summary/content columns)
    
    Returns:
        DataFrame: DataFrame with added text column
    """
    df = df.copy()
    
    # Combine title + summary/content
    if 'news_summary' in df.columns:
        # Replace 'nil' with empty string in news_summary
        news_summary = df['news_summary'].fillna('')
        news_summary = news_summary.replace('nil', '')
        df['text'] = (df['Title'].fillna('') + ' \n\n ' + news_summary)
    else:
        # If no content column, just use title
        df['text'] = df['Title'].fillna('')
    
    # Ensure ID column exists
    if 'id' not in df.columns:
        df['id'] = df.index
    
    return df 