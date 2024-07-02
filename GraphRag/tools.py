from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from llama_index.core import Document
from llama_index.core import Settings




def initialize_llm(base_url: str ="http://localhost:11434",model: str ="llama3",chunk_size: int = 512):
    """
    Initializes the llm for building the KnowledgeGraph
    Args:
        base_url: The ollama server URL where the model is listening
        model: The model string that ollama is hosting and will be used to build the KnowledgeGraph
        chunk_size: The Documents uploaded will be chunked,it represents size of each chunk

    Returns:
        None
    """
    llm = Ollama(base_url=base_url,model=model)
    Settings.llm = llm
    Settings.chunk_size = chunk_size
    print(f"{model} initialized succesfully!")

def code_spiltting(documents,language: str = "python"):
    """
    If the KnowledgeGraph is to be built for code-files then files are splitted using this function
    Args:
        documents: llama-index Document type object,Then coding-files Document
        language: The language of coding-file

    Returns:
        nodes: Split code chunks,llama-index Nodes type object
    """
    splitter = CodeSplitter(
        language=language,
        chunk_lines=30,  # lines per chunk
        chunk_lines_overlap=6,  # lines overlap between chunks
        max_chars=1500,  # max chars per chunk
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"{len(nodes)} nodes created succesfully!")
    return nodes

def convert_nodes_to_docs(nodes):
    """
    converts llama-index Nodes Type object to llama-index Document Type objects
    Args:
        nodes: llama-index Nodes type object
    Returns:
        lama-index Document Type objects
    """
    documents_from_nodes = [Document(text=node.text, metadata=node.metadata) for node in nodes]
    print(f"{len(documents_from_nodes)} number of documents are being converted successfully!")
    return documents_from_nodes

def load_directory(directory_path: str,code_file: bool = False,language: str = "python"):
    """
    Loads the documentation-directory, does preprocessing and chunking depending on code_file parameter
    Args:
        directory_path: Path to the Files Directory from which Knowledge graph is to be made
        code_file: Bool that specifies that given directory contains code files or not
        language: language of the code-files
    Returns:
        lama-index Document Type objects
    """

    documents = SimpleDirectoryReader(directory_path).load_data()

    if code_file:
        nodes=code_spiltting(documents,language)
        docs=convert_nodes_to_docs(nodes)
        print(f"{len(documents)}Documents loaded succesfully!")
        return docs

    print(f"{len(documents)}Documents loaded succesfully!")
    return documents


