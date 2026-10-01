"""
MCP Client Test Script — Validates MCP Server (mcp_server.py) over stdio transport.
Queries available tools and executes tool calls using Model Context Protocol (MCP 2.x).
"""

import asyncio
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    print(f"Connecting to MCP Server via stdio ({server_script})...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize MCP Session
            await session.initialize()
            print("[OK] MCP Session Initialized successfully!")

            # 2. Query List of Available MCP Tools
            tools_response = await session.list_tools()
            print("\n=== Registered MCP Tools ===")
            for t in tools_response.tools:
                print(f"  🔧 Tool Name : {t.name}")
                print(f"     Description: {t.description.strip().splitlines()[0]}")
                print()

            # 3. Test MCP Tool Execution: search_products
            print("--- Testing Tool Call: search_products(query='honey', max_price=20) ---")
            search_res = await session.call_tool(
                "search_products",
                arguments={"query": "honey", "max_price": 20},
            )
            print("Response:")
            print(search_res.content[0].text[:300])

            # 4. Test MCP Tool Execution: rag_search_products
            print("\n--- Testing Tool Call: rag_search_products(query='something sweet for sore throat') ---")
            rag_res = await session.call_tool(
                "rag_search_products",
                arguments={"query": "something sweet for sore throat"},
            )
            print("Response:")
            print(rag_res.content[0].text[:300])


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
