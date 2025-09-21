import os
import pandas as pd
from pathlib import Path
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import Config
from .categorizer import NewsClassifier, prepare_article_text
from .clustering import NewsClustering
from .highlights import HighlightExtractor
from .faiss_vectore_store import (
    get_langchain_embeddings,
    init_faiss_with_categories,
    upsert_articles_faiss,
    save_faiss_index,
    load_faiss_index,
    query_faiss
)


# ───────────────────────────────
# Pipeline
# ───────────────────────────────
def process_news_pipeline(news_csv_path=None, highlights_csv_path=None, save_results=True):
    """Run the complete news processing pipeline using FAISS instead of Chroma"""
    
    if news_csv_path is None:
        news_csv_path = Config.NEWS_CSV_PATH
    
    if highlights_csv_path is None:
        highlights_csv_path = Config.HIGHLIGHTS_CSV_PATH
    
    # 1. Load and prepare data
    logger.info(f"Loading news data from {news_csv_path}")
    df = pd.read_csv(news_csv_path)
    df = prepare_article_text(df)
    
    # 2. Initialize components
    classifier = NewsClassifier()
    clustering = NewsClustering()
    highlighter = HighlightExtractor()
    
    # 3. Classify articles
    logger.info("Classifying articles")
    df = classifier.classify_dataframe(df)
    
    # 4. Create embeddings for clustering
    logger.info("Creating embeddings for clustering")
    embeddings = classifier.batch_embed_texts(df['text'].tolist())
    
    # 5. Cluster articles
    logger.info("Clustering articles to detect duplicates")
    df = clustering.add_clusters_to_df(df, embeddings)
    
    # 6. Extract highlights
    logger.info("Extracting important highlights")
    highlights_df = highlighter.extract_highlights(df)
    
    # 7. Index articles in FAISS vector store
    logger.info("Indexing articles in FAISS")
    if os.path.exists("faiss_store"):
        faiss_index = load_faiss_index("faiss_store")
    else:
        faiss_index = init_faiss_with_categories()
    
    faiss_index = upsert_articles_faiss(df, faiss_index)
    save_faiss_index(faiss_index, "faiss_store")
    
    # 8. Save results if requested
    if save_results:
        logger.info(f"Saving classified news to {Config.CLASSIFIED_ARTICLES_CSV_PATH}")
        df.to_csv(Config.CLASSIFIED_ARTICLES_CSV_PATH, index=False)
        
        logger.info(f"Saving highlights to {Config.HIGHLIGHTS_CSV_PATH}")
        highlights_df.to_csv(Config.HIGHLIGHTS_CSV_PATH, index=False)
    
    return df, highlights_df


# ───────────────────────────────
# Init FAISS Vector Store (Highlights)
# ───────────────────────────────
def init_vector_store(highlights_path=None):
    """Initialize FAISS vector store with highlights for RAG"""
    
    if highlights_path is None:
        highlights_path = Config.HIGHLIGHTS_CSV_PATH
    
    # Start with categories in FAISS
    if os.path.exists("faiss_store"):
        faiss_index = load_faiss_index("faiss_store")
    else:
        faiss_index = init_faiss_with_categories()
    
    # Load highlights data
    logger.info(f"Loading highlights data from {highlights_path}")
    try:
        highlights_df = pd.read_csv(highlights_path)
    except FileNotFoundError:
        logger.warning(f"Highlights file {highlights_path} not found. Vector store will be empty.")
        return faiss_index
    
    # Upsert highlights as articles
    faiss_index = upsert_articles_faiss(highlights_df, faiss_index)
    save_faiss_index(faiss_index, "faiss_store")
    
    return faiss_index


# ───────────────────────────────
# Answer Questions via FAISS RAG
# ───────────────────────────────
def answer_question(question, faiss_index=None, k=5):
    """Answer a question using FAISS RAG"""
    
    # Get or initialize FAISS index
    if faiss_index is None:
        faiss_index = init_vector_store()
    
    # Query for similar documents
    results = query_faiss(faiss_index, question, k=k)
    
    # Format context for prompt
    context = ""
    sources = []
    
    for i, r in enumerate(results):
        context += f"{r.page_content}\n\n"
        metadata = r.metadata or {}
        source = {
            "id": metadata.get("id", f"source-{i}"),
            "title": metadata.get("Title", "Unknown"),
            "category": metadata.get("predicted_category", "Unknown")
        }
        if source not in sources:
            sources.append(source)
    
    # Try to determine the question category to filter relevant sources
    categories = ["sports", "finance", "politics", "lifestyle", "music"]
    question_lower = question.lower()
    question_category = next((c for c in categories if c in question_lower), None)
    
    if question_category:
        filtered_sources = [s for s in sources if s["category"].lower() == question_category]
        if len(filtered_sources) >= 2:
            sources = filtered_sources
    
    # Prompt template
    prompt_template = """
    You are a helpful assistant that answers questions about today's news headlines.
    Use the following context to answer the question. If you don't know the answer, just say you don't know.
    Don't refer to "Document 1" or other document numbers in your answer.
    Provide a natural, conversational response based on the information provided.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # LLM chain
    llm = ChatOpenAI(temperature=0.1)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.invoke({"context": context, "question": question})
    answer = result.get("text", "") if isinstance(result, dict) else str(result)
    
    return {
        "answer": answer,
        "sources": sources
    }
