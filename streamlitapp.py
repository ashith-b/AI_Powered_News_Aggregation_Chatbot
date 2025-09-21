import streamlit as st
import pandas as pd
from loguru import logger
from rag.utils import init_vector_store, answer_question, process_news_pipeline
from config import Config
import time
import os


def init_app():
    """Initialize Streamlit page and FAISS vector store"""
    # Set Streamlit page config
    st.set_page_config(page_title="News Aggregator RAG App", layout="wide")

    # Title
    st.title("📰 News Aggregator with RAG Chatbot")
    st.write("This app lets you process news, ask questions using a RAG model, and explore articles with filters.")

    # Initialize FAISS vector store (only once)
    if "vector_store_initialized" not in st.session_state:
        with st.spinner("Initializing FAISS vector store..."):
            try:
                init_vector_store()
                st.session_state.vector_store_initialized = True
                logger.info("FAISS vector store initialized successfully.")
            except Exception as e:
                logger.error(f"FAISS vector store initialization failed: {e}")
                st.error("Failed to initialize FAISS vector store.")


def chatbot_section():
    """Chatbot interaction UI"""
    st.header("💬 Chat with the News Bot")
    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Generating answer..."):
                try:
                    response = answer_question(question)
                    st.success("Answer:")
                    st.write(response.get("answer", "No answer found."))

                    # Show sources if available
                    sources = response.get("sources", [])
                    if sources:
                        st.markdown("### 🔗 Sources")
                        for s in sources:
                            st.markdown(f"- **{s['title']}** ({s['category']})")
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    st.error("Failed to get answer.")
        else:
            st.warning("Please type a question.")


def process_news_section():
    """Process news pipeline and update session state"""
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.df = None
        st.session_state.highlights_df = None

    if not st.session_state.processed:
        # Only process if classified articles do not exist
        if not os.path.exists(Config.CLASSIFIED_ARTICLES_CSV_PATH):
            with st.spinner("⏳ Processing news articles..."):
                try:
                    start_time = time.time()
                    df, highlights_df = process_news_pipeline(
                        news_csv_path=Config.NEWS_CSV_PATH,
                        highlights_csv_path=Config.HIGHLIGHTS_CSV_PATH
                    )
                    st.session_state.df = df
                    st.session_state.highlights_df = highlights_df
                    st.session_state.processed = True

                    elapsed = time.time() - start_time
                    st.success(f"News processed in {elapsed:.2f} seconds")
                except Exception as e:
                    logger.error(f"Auto-processing error: {str(e)}")
                    st.error(f"Failed to process news: {str(e)}")
        else:
            st.info("ℹ️ Classified articles file already exists. Skipping auto-processing.")


def highlights_section():
    """Display highlights with category filters"""
    st.header("🌟 News Highlights")
    try:
        highlights_df = pd.read_csv(Config.HIGHLIGHTS_CSV_PATH)

        categories = highlights_df["predicted_category"].dropna().unique().tolist()
        selected_category = st.selectbox("Filter by category", options=["All"] + sorted(categories))

        if selected_category != "All":
            highlights_df = highlights_df[highlights_df["predicted_category"] == selected_category]

        st.write(f"📝 Showing {len(highlights_df)} highlights")

        expected_cols = ["Title", "summary", "predicted_category"]
        available_cols = [col for col in expected_cols if col in highlights_df.columns]

        if available_cols:
            st.dataframe(highlights_df[available_cols])
        else:
            st.dataframe(highlights_df)

    except FileNotFoundError:
        st.warning("Highlights file not found. Try clicking 'Process News' above to generate it.")
    except Exception as e:
        logger.error(f"Error loading highlights: {e}")
        st.error(f"Something went wrong: {e}")


def articles_viewer_section():
    """Display full articles with pagination and filters"""
    st.header("🗂️ Full News Articles Viewer")
    try:
        articles_df = pd.read_csv(Config.CLASSIFIED_ARTICLES_CSV_PATH)
        news_df = pd.read_csv(Config.NEWS_CSV_PATH)

        articles_df["id"] = articles_df["id"].astype(str)
        news_df["id"] = news_df.index.astype(str)

        merged_df = pd.merge(
            articles_df,
            news_df,
            left_on="id",
            right_on="id",
            how="left",
            suffixes=("", "_orig")
        )

        required_cols = [
            "Title", "news_summary", "text", "Author", "Publication",
            "news_card_image", "Link", "predicted_category"
        ]
        for col in required_cols:
            if col not in merged_df.columns:
                merged_df[col] = ""

        merged_df["author"] = merged_df["Author"].fillna("").replace("nil", "")
        merged_df["url"] = merged_df["Link"].fillna("")
        merged_df["urlToImage"] = merged_df["news_card_image"].fillna("")
        merged_df["description"] = merged_df["news_summary"].fillna("").replace("nil", "")
        merged_df["publishedAt"] = "2025-05-10T12:00:00Z"
        merged_df["content"] = merged_df["text"].fillna("")
        merged_df["category"] = merged_df["predicted_category"].fillna("general")

        # Sidebar filters
        st.header("📊 Filter Articles")
        categories = ["All"] + sorted(merged_df["category"].dropna().unique().tolist())
        selected_category = st.selectbox("Select Category", categories)

        # Pagination
        total_articles = len(merged_df)
        page_size = st.sidebar.slider("Articles per page", 5, 25, 10)
        total_pages = (total_articles + page_size - 1) // page_size
        current_page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, step=1)

        filtered_df = merged_df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_category]

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_df.iloc[start_idx:end_idx]

        st.markdown(f"### Showing {len(page_df)} of {len(filtered_df)} filtered articles")

        for _, row in page_df.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    if row["urlToImage"]:
                        st.image(row["urlToImage"], use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/120x80.png?text=No+Image", use_container_width=True)
                with cols[1]:
                    st.markdown(f"#### [{row['Title']}]({row['url']})")
                    st.markdown(f"**Source:** {row['Publication']} | **Author:** {row['author']}")
                    st.markdown(f"**Published:** {row['publishedAt']}")
                    st.markdown(f"{row['description']}")
                    st.markdown("---")

    except FileNotFoundError:
        st.error("📂 News files not found. Try running 'Process News' above.")
    except Exception as e:
        st.error(f"Error loading or displaying articles: {e}")


def main():
    init_app()
    chatbot_section()
    process_news_section()
    highlights_section()
    articles_viewer_section()


if __name__ == "__main__":
    main()
