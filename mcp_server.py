"""
Model Context Protocol (MCP) Server — Shopping Store Tools.
Exposes database search, RAG vector retrieval, rating aggregation, and checkout tools
over the Model Context Protocol (MCP 2.x) standard via MCPServer.
"""

import os
import json
import sqlite3
import base64
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcp.server.mcpserver import MCPServer
from reviews_api import get_product_rating
from rag_search import semantic_search_products

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# Initialize MCP Server instance
mcp = MCPServer("ShoppingStoreMCPServer")


@mcp.tool()
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    MCP Tool: Search product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns JSON array of matching products.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    if query and query.lower() not in ("none", "null", ""):
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "description": row[4],
            "is_organic": bool(row[5]),
        }
        for row in rows
    ]
    return json.dumps(products)


@mcp.tool()
def rag_search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    MCP Tool: Perform RAG vector similarity search on product catalog using ChromaDB embeddings.
    Use for conceptual or health benefit queries (e.g. 'something sweet for a sore throat').
    """
    results = semantic_search_products(query, max_price=max_price, is_organic=is_organic)
    return json.dumps(results)


@mcp.tool()
def get_rating(product_id: int) -> str:
    """
    MCP Tool: Get average customer rating and total review count for a product by its ID.
    Returns JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(int(product_id))
    return json.dumps(result)


@mcp.tool()
def checkout(product_id: int) -> str:
    """
    MCP Tool: Place an order for the given product ID. Saves order to SQLite database and returns
    confirmation message with order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (int(product_id),))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (int(product_id), name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


if __name__ == "__main__":
    sys.stderr.write("Starting Shopping Store MCP Server (MCP 2.x)...\n")
    mcp.run()
