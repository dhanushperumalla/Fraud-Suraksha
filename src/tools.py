"""
Fraud Suraksha - Custom Tools
This module defines custom tools for the fraud detection and analysis system.
"""

from typing import List, Dict, Any
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.tools import StructuredTool
from langchain.schema import Document

def create_rag_query_tool(retriever: VectorStoreRetriever) -> StructuredTool:
    """
    Creates a structured tool for querying a RAG retriever.

    Args:
        retriever (VectorStoreRetriever): The retriever to query.

    Returns:
        StructuredTool: A tool that can be used to query the retriever.
    """
    def rag_query(query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Query the RAG retriever for relevant documents.

        Args:
            query (str): The user query.
            k (int): Number of top documents to retrieve.

        Returns:
            List[Dict[str, Any]]: List of relevant documents with metadata.
        """
        docs: List[Document] = retriever.get_relevant_documents(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    return StructuredTool.from_function(
        func=rag_query,
        name="rag_query",
        description="Query the RAG retriever for relevant documents. Input: query (str), k (int, optional, default=4). Returns: List of documents with content and metadata."
    )