import os
import pandas as pd
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config import Config


# ───────────────────────────────
# Embeddings
# ───────────────────────────────
def get_langchain_embeddings():
    """Get OpenAI embeddings (text-embedding-ada-002) for FAISS"""
    return OpenAIEmbeddings(
        api_key=Config.OPENAI_API_KEY,
        model="text-embedding-ada-002"   # explicitly set here
    )


# ───────────────────────────────
# Initialize FAISS with Categories
# ───────────────────────────────
def init_faiss_with_categories():
    """Create FAISS index with categories pre-loaded"""
    embeddings = get_langchain_embeddings()

    docs = []
    for cat_name, cat_prompt in Config.NEWS_CATEGORIES.items():
        docs.append(Document(
            page_content=cat_prompt,
            metadata={"type": "category", "category": cat_name}
        ))

    faiss_index = FAISS.from_documents(docs, embeddings)
    logger.info(f"Initialized FAISS index with {len(docs)} categories")
    return faiss_index


# ───────────────────────────────
# Add Articles
# ───────────────────────────────
def upsert_articles_faiss(articles_df, faiss_index):
    """Add news articles into FAISS index"""
    embeddings = get_langchain_embeddings()

    docs = [
        Document(
            page_content=row["text"],
            metadata={
                "type": "article",
                "id": str(row["id"]),
                "Title": row["Title"],
                "predicted_category": row["predicted_category"],
                "cluster": row["cluster"]
            }
        )
        for _, row in articles_df.iterrows()
    ]

    faiss_index.add_documents(docs, embeddings=embeddings)
    logger.info(f"Upserted {len(docs)} articles to FAISS")
    return faiss_index


# ───────────────────────────────
# Save / Load
# ───────────────────────────────
def save_faiss_index(faiss_index, path):
    os.makedirs(path, exist_ok=True)
    faiss_index.save_local(path)
    logger.info(f"Saved FAISS index to {path}")


def load_faiss_index(path):
    embeddings = get_langchain_embeddings()
    index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Loaded FAISS index from {path}")
    return index


# ───────────────────────────────
# Querying
# ───────────────────────────────
def query_faiss(faiss_index, query, k=5):
    """Search FAISS for similar docs"""
    results = faiss_index.similarity_search(query, k=k)
    for r in results:
        print(f"\nContent: {r.page_content}\nMetadata: {r.metadata}\n")
    return results

