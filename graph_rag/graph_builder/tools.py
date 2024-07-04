"""
This file contains functions for initializing llm for
1. building KnowledgeGraph
2. loading documents from directory (also function for splitting code files)
3. converting llama-index Node to llama-index Documents
"""

from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from llama_index.core import Document
from llama_index.core import Settings


def initialize_llm(
    base_url: str = "http://localhost:11434",
    model: str = "llama3",
    chunk_size: int = 512,
):
    """
    Initializes the llm for building the KnowledgeGraph
    Args:
        base_url: The ollama server URL where the model is listening
        model: The model string that ollama is hosting and will be used to build the KnowledgeGraph
        chunk_size: The Documents uploaded will be chunked, it represents size of each chunk

    Returns:
        None
    """
    try:
        llm = Ollama(base_url=base_url, model=model)
        Settings.llm = llm
        Settings.chunk_size = chunk_size
        print(f"{model} initialized successfully!")
    except Exception as e:
        print(f"Error initializing LLM: {e}")


def code_splitting(documents, language: str = "python"):
    """
    If the KnowledgeGraph is to be built for code-files then files are split using this function
    Args:
        documents: llama-index Document type object, then coding-files Document
        language: The language of coding-file

    Returns:
        nodes: Split code chunks, llama-index Nodes type object
    """
    try:
        splitter = CodeSplitter(
            language=language,
            chunk_lines=30,  # lines per chunk
            chunk_lines_overlap=6,  # lines overlap between chunks
            max_chars=1500,  # max chars per chunk
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"{len(nodes)} nodes created successfully!")
        return nodes
    except Exception as e:
        print(f"Error splitting code: {e}")
        return []


def convert_nodes_to_docs(nodes):
    """
    Converts llama-index Nodes Type object to llama-index Document Type objects
    Args:
        nodes: llama-index Nodes type object
    Returns:
        llama-index Document Type objects
    """
    try:
        documents_from_nodes = [
            Document(text=node.text, metadata=node.metadata) for node in nodes
        ]
        print(
            f"{len(documents_from_nodes)} number of documents converted successfully!"
        )
        return documents_from_nodes
    except Exception as e:
        print(f"Error converting nodes to documents: {e}")
        return []


def load_directory(
    directory_path: str, code_file: bool = False, language: str = "python"
):
    """
    Loads the documentation-directory, does preprocessing and chunking depending on code_file parameter
    Args:
        directory_path: Path to the Files Directory from which Knowledge graph is to be made
        code_file: Bool that specifies that given directory contains code files or not
        language: language of the code-files
    Returns:
        llama-index Document Type objects
    """
    try:
        documents = SimpleDirectoryReader(directory_path).load_data()
    except Exception as e:
        print(f"Error loading directory: {e}")
        return []

    try:
        if code_file:
            nodes = code_splitting(documents, language)
            docs = convert_nodes_to_docs(nodes)
            print(f"{len(docs)} documents loaded successfully!")
            return docs

        print(f"{len(documents)} documents loaded successfully!")
        return documents
    except Exception as e:
        print(f"Error processing documents: {e}")
        return []
