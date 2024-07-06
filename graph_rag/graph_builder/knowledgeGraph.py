"""
This file contains the functions to build and save the KnowledgeGraph Index and save it as a pickle-file
"""

from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from pyvis.network import Network
import os
import pickle


def build_graph(
    documents: str,
    llm: str = None,
    max_triplets_per_chunk: int = 10,
    embeddings: str = "microsoft/codebert-base",
    include_embeddings: bool = False,
):
    """
    This function builds KnowledgeGraph Index that can be queried
    Args:
        documents: llama-index Document type object
        llm:
        max_triplets_per_chunk: Max triplets that can be extracted from each document chunk defaults:3
        embeddings: Hugging-Face Embeddings model name default: microsoft/codebert-base

    Returns:
        Knowledge Graph-index, also saves html visualization file
    """
    try:
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=max_triplets_per_chunk,
            llm=llm,
            embed_model=HuggingFaceEmbedding(model_name=embeddings),
            storage_context=storage_context,
            include_embeddings=include_embeddings,
        )
        print("KG built successfully!")

        os.makedirs("results", exist_ok=True)
        g = index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show("Graph_visualization.html")
        return index
    except Exception as e:
        print(f"Error building graph: {e}")
        return None


def save_index(index, output_dir_path: str = "Results/"):
    """
    Serializes the index object, so that it can be loaded and used later
    Args:
        index: Graph-Index object

    Returns:
        Saves pickle file of the Graph-Index
    """
    try:
        os.makedirs(output_dir_path[:-1], exist_ok=True)
        with open(output_dir_path + "graphIndex", "wb") as f:
            pickle.dump(index, f)
        print("Index saved successfully!")
    except Exception as e:
        print(f"Error saving index: {e}")
