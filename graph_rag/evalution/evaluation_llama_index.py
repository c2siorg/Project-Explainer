from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

RAG_DATASET="./data/rag_dataset.json"

RagEvaluatorPack = download_llama_pack(
    "RagEvaluatorPack", "./rag_evaluator_pack"
)
def evaluate(RAG_DATASET: str,query_engine: object,ollama_model: str="llama3",embedd_model: str="microsoft/codebert-base"):
    rag_dataset = LabelledRagDataset.from_json(RAG_DATASET)
    rag_evaluator_pack = RagEvaluatorPack(
        rag_dataset=rag_dataset,
        query_engine=query_engine,
        judge_llm=Ollama(base_url="http://localhost:11434", model=ollama_model),
        embed_model=HuggingFaceEmbedding(model_name=embedd_model),
    )
    benchmark_df = await rag_evaluator_pack.arun(
        batch_size=5,  # batches the number of openai api calls to make
        sleep_time_in_seconds=1,  # seconds to sleep before making an api call
    )
    return benchmark_df