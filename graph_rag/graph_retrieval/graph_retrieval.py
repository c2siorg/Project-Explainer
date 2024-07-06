"""
This file contains methods for loading graph_index from pkl file and retrieval of graph_index
"""

from graph_rag.graph_builder.tools import initialize_llm
import pickle


def get_index_from_pickle(
    file_path: str = "results/graphIndex.pkl",
):
    """
    Deserializes a .pkl file to get the graph_index.
    Args:
        file_path (str): The path to the .pkl file.

    Returns:
        object: The deserialized llama_index graph_index object.

    """
    try:
        with open(file_path, "rb") as file:
            index = pickle.load(file)
        return index
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except IOError as e:
        print(f"Error reading file: {e}")
        raise
    except pickle.UnpicklingError as e:
        print(f"Error deserializing file: {e}")
        raise


def get_query_engine(index, with_embedding: bool = False, similarity_top_k: int = 5):
    """
    create query-engine with preferred settings that is used to query graph_index
    Args:
        index (object): llama_index graph_index object
        with_embedding (bool): switch to True to query graph_index with embeddings Default:False
        similarity_top_k (int): Top number of chunks that is to be provided as context to llm for response to given query

    Returns:
        object: llama_index query_engine object

    """
    if index is None:
        raise ValueError("The index must not be None.")
    try:
        initialize_llm()
        if with_embedding:
            query_engine = index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=similarity_top_k,
            )
        else:
            query_engine = index.as_query_engine(
                include_text=True, response_mode="tree_summarize"
            )
        return query_engine
    except Exception as e:
        print(f"An error occurred while creating the query engine: {e}")
        raise


def graph_query(query: str, query_engine):
    """
    method to query graph_index
    Args:
        query (str): query that is to be answered using graph_rag
        query_engine (object): llama_index query_engine object

    Returns:
        str: response to the query in string

    """
    if not query:
        raise ValueError("The query must not be empty or None.")

    try:
        response = query_engine.query(query)
        return response.response
    except Exception as e:
        print(f"An error occurred while querying: {e}")
        raise
