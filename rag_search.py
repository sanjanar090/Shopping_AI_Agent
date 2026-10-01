"""
RAG Search Module — Vector Embeddings & Semantic Product Retrieval using ChromaDB.
Builds a persistent ChromaDB collection from store.db products and customer reviews.
"""

import os
import json
import sqlite3
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain.tools import tool

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

_embedding_fn = None

def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedding_fn


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def build_vector_store():
    """Extract products and reviews from store.db and populate ChromaDB collection."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run setup_db.py first.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch products
    cursor.execute("SELECT id, name, category, price, description, is_organic FROM products")
    products = cursor.fetchall()

    # Fetch reviews grouped by product_id
    cursor.execute("SELECT product_id, review_text, rating FROM reviews")
    reviews_rows = cursor.fetchall()
    conn.close()

    reviews_map = {}
    for pid, text, r in reviews_rows:
        if pid not in reviews_map:
            reviews_map[pid] = []
        reviews_map[pid].append(f"{text} ({r} stars)")

    documents = []
    metadatas = []
    ids = []

    for p in products:
        pid, name, category, price, description, is_organic = p
        prod_reviews = " ".join(reviews_map.get(pid, []))

        # Rich text representation for embedding
        doc_text = (
            f"Product Name: {name}. "
            f"Category: {category}. "
            f"Price: ${price:.2f}. "
            f"Organic: {'Yes' if is_organic else 'No'}. "
            f"Description: {description}. "
            f"Customer Feedback: {prod_reviews}"
        )

        documents.append(doc_text)
        metadatas.append({
            "id": pid,
            "name": name,
            "category": category,
            "price": float(price),
            "is_organic": bool(is_organic),
            "description": description,
        })
        ids.append(str(pid))

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="shopping_products",
        embedding_function=get_embedding_fn(),
    )

    # Upsert documents into vector store
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"RAG Vector Store initialized with {len(documents)} products at: {CHROMA_PATH}")
    return collection


def semantic_search_products(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    top_k: int = 5,
) -> list[dict]:
    """Perform RAG vector similarity search on product catalog with optional metadata filtering."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(
            name="shopping_products",
            embedding_function=get_embedding_fn(),
        )
    except Exception:
        # Build collection if not existing yet
        collection = build_vector_store()

    # Query ChromaDB vector index
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 2,  # fetch extra for post-filtering
    )

    matched_products = []
    if results and "metadatas" in results and results["metadatas"]:
        metas = results["metadatas"][0]
        for meta in metas:
            # Metadata filtering
            if max_price is not None and meta["price"] > max_price:
                continue
            if is_organic is not None and meta["is_organic"] != is_organic:
                continue
            matched_products.append(meta)
            if len(matched_products) >= top_k:
                break

    return matched_products


@tool
def rag_search_products(
    query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None
) -> str:
    """
    Perform semantic RAG vector search on the product catalog.
    Use this tool when the user describes what they want in natural language, health benefits,
    or conceptual queries (e.g. 'something sweet for a sore throat', 'low sugar breakfast', 'caffeine free tea').
    Returns a JSON list of matching products with: id, name, category, price, description, is_organic.
    """
    results = semantic_search_products(query, max_price=max_price, is_organic=is_organic)
    return json.dumps(results)


if __name__ == "__main__":
    build_vector_store()
    print("\n--- Test RAG Search 1: 'something sweet for a sore throat' ---")
    print(rag_search_products.invoke({"query": "something sweet for a sore throat"}))

    print("\n--- Test RAG Search 2: 'caffeine free soothing warm drink' ---")
    print(rag_search_products.invoke({"query": "caffeine free soothing warm drink"}))
