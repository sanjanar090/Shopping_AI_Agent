"""
LangGraph Shopping Agent Engine — Explicit StateGraph with Tool Nodes & Memory Checkpointing.
Integrates ReAct reasoning, RAG semantic search, SQLite tools, and Multimodal Vision LLM into a graph state machine.
"""

import base64
import json
import os
import sqlite3
import sys
from typing import Annotated, Literal, Optional, Sequence, TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rag_search import rag_search_products
from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# Initialize Models
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
vision_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search product database by keyword (matched against name, description, and category).
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


def get_rating(product_id: int) -> str:
    """
    Get average customer rating and total review count for a product by its ID.
    Returns JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(int(product_id))
    return json.dumps(result)


def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves order to SQLite database and returns
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


def describe_product_image(image_path: str) -> str:
    """
    Analyze product image and return key attributes as JSON object.
    Use this when user uploads a photo of a product they are interested in.
    """
    image_path = image_path.strip("'\"")
    if not os.path.isabs(image_path):
        base_dir = os.path.dirname(__file__)
        possible_path = os.path.join(base_dir, image_path)
        if os.path.exists(possible_path):
            image_path = possible_path

    if not os.path.exists(image_path):
        return json.dumps({"error": f"Image file not found at path: {image_path}"})

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is\n"
                "- search_query: a short keyword to search for it\n"
                "- is_organic: true if label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content


tools = [search_products, rag_search_products, get_rating, checkout, describe_product_image]
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# LangGraph State & Graph Definition
# ---------------------------------------------------------------------------

class ShoppingAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "IMAGE SEARCH — when user provides an image path:\n"
        "1. Call describe_product_image with path to identify product.\n"
        "2. Use returned search_query and is_organic to call search_products or rag_search_products.\n"
        "3. Continue with BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when user describes what they want to buy:\n"
        "1. Call search_products (for direct keyword/category matches) or rag_search_products (for conceptual, health benefit, or natural language queries).\n"
        "2. For each candidate, call get_rating to retrieve average rating.\n"
        "3. Filter by minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use exact format:\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between product entries. Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in list and ask: 'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when user confirms they want to buy (e.g. 'yes', 'sure', 'order number 2'):\n"
        "1. Look at previous message to find (ID:X) for chosen product.\n"
        "2. Call checkout with product_id.\n"
        "3. Confirm order to user in plain text.\n\n"
        "Never place an order unless user explicitly confirms. Never guess product_id — take it from (ID:X) in previous message."
    )
)


def agent_node(state: ShoppingAgentState):
    """Executes LLM reasoning with full message context and system prompt."""
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: ShoppingAgentState) -> Literal["tools", "__end__"]:
    """Routes execution to tools node if tool_calls present, else ends graph step."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools)

# Construct LangGraph StateGraph
workflow = StateGraph(ShoppingAgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Memory Saver Checkpointer
memory = MemorySaver()
langgraph_agent = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "cli_session_1"}}
    print("--- Testing LangGraph Shopping Agent ---")
    inputs = {
        "messages": [
            HumanMessage(content="I want organic honey under $20 with a 4.5+ rating.")
        ]
    }
    result = langgraph_agent.invoke(inputs, config=config)
    print(result["messages"][-1].content)
