"""
This script evaluates a RagDataset using a RagEvaluatorPack, which assesses query engines by benchmarking against
labeled data using LLMs and embeddings.

Functions:
- evaluate: Evaluates the query engine using a labeled RAG dataset and specified models for both the LLM and embeddings.
"""

from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding





def evaluate(
    RAG_DATASET: str,
    query_engine: object,
    ollama_model: str = "llama3",
    embedd_model: str = "microsoft/codebert-base",
):
    """
    Evaluates a RAG dataset by using a query engine and benchmarks it using LLM and embedding models.

    Args:
        RAG_DATASET: Path to the JSON file containing the labeled RAG dataset.
        query_engine: The query engine to evaluate.
        ollama_model: The LLM model to use for evaluation (default: "llama3").
        embedd_model: The Hugging Face embedding model to use for evaluation (default: "microsoft/codebert-base").

    Returns:
        A DataFrame containing the benchmarking results, including LLM calls and evaluations.
    """

    RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./rag_evaluator_pack")
    rag_dataset = LabelledRagDataset.from_json(RAG_DATASET)
    rag_evaluator_pack = RagEvaluatorPack(
        rag_dataset=rag_dataset,
        query_engine=query_engine,
        judge_llm=Ollama(base_url="http://localhost:11434", model=ollama_model),
        embed_model=HuggingFaceEmbedding(model_name=embedd_model),
    )
    benchmark_df = await rag_evaluator_pack.arun(
        batch_size=5,  # batches the number of llm calls to make
        sleep_time_in_seconds=1,  # seconds to sleep before making an api call
    )
    return benchmark_df
