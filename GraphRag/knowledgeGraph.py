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
):
    """
    This function builds KnowledgeGraph Index that can be queried
    Args:
        documents: llama-index Document type object
        llm:
        max_triplets_per_chunk: Max triplets that can be extracted from each document chunk defaults:3
        embeddings: Hugging-Face Embeddings model name default: microsoft/codebert-base

    Returns:
        Knowledge Graph-index,also saves html visualization file

    """
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=max_triplets_per_chunk,
        llm=llm,
        embed_model=HuggingFaceEmbedding(model_name=embeddings),
        storage_context=storage_context,
    )
    print("KG built succesfully!")
    os.makedirs("results", exist_ok=True)
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.show("Graph_visualization.html")
    return index


def save_index(index):
    """
    serializes the index object,so that it can be loaded and used later
    Args:
        index: Grpah-Index object

    Returns:
        saves pickle file of the Grpah-Index
    """
    os.makedirs("results", exist_ok=True)
    with open("results/graphIndex", "wb") as f:
        pickle.dump(index, f)
    print("Index saved succesfully!")
