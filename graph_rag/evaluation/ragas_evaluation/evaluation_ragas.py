"""
This script loads a pre-processed dataset, slices it for batch evaluation, and runs a series of metrics to evaluate the
performance of a query engine using a language model and embeddings.

Functions:
- load_test_dataset: Loads a test dataset from a pickle file.
- slice_data: Slices the dataset into batches for evaluation.
- evaluate: Runs evaluation on the sliced dataset using specified metrics, LLMs, and embeddings.

"""

import pickle
import pandas as pd
from datasets import Dataset
from ragas.integrations.llama_index import evaluate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.metrics.critique import harmfulness
from llama_index.llms.ollama import Ollama
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def load_test_dataset(
    data: str,
):
    """
       Loads a test dataset from a pickle file.

       Args:
           data: The path to the dataset file in pickle format.

       Returns:
           A dictionary representing the loaded dataset or an empty dictionary if loading fails due to EOFError.
       """
    try:
        with open(data, "rb") as f:
            dataset = pickle.load(f)
    except EOFError:
        print("EOFError: The file may be corrupted or incomplete loading empty dictionary.")
        dataset = {}
    return dataset


def slice_data(i: int, k: int, dataset: dict):
    """
        Slices the dataset into smaller chunks for batch processing.

        Args:
            i: The starting index for the slice.
            k: The size of the slice (number of records to include in each batch).
            dataset: The dictionary representing the dataset to be sliced.

        Returns:
            A dictionary containing the sliced dataset with renamed columns for consistency with the evaluation process.
        """

    hf_dataset = Dataset.from_list(dataset[i : i + k])
    hf_dataset = hf_dataset.rename_column("context", "contexts")
    hf_dataset = hf_dataset.rename_column("answer", "ground_truth")
    ds_dict = hf_dataset.to_dict()
    return ds_dict


def evaluate(
    query_engine: object,
    dataset: object,
    batch: int = 4,
    metrics: list = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm: object = Ollama(base_url="http://localhost:11434", model="codellama"),
    embeddings=HuggingFaceEmbedding(model_name="microsoft/codebert-base"),
):
    """
       Evaluates the performance of a query engine on a dataset using various metrics and a language model.

       Args:
           query_engine: The query engine to be evaluated.
           dataset: The dataset to be evaluated against.
           batch: The number of records to process in each batch (default: 4).
           metrics: A list of metrics to be used for evaluation (default: faithfulness, answer relevancy, context precision, and context recall).
           llm: The language model to be used for evaluation (default: Ollama with model 'codellama').
           embeddings: The embedding model to be used (default: HuggingFaceEmbedding with 'microsoft/codebert-base').

       Returns:
           A pandas DataFrame containing the evaluation results for each batch.
       """

    rows_count = len(next(iter(dataset.values())))

    results_df = pd.DataFrame()

    for i in range(0, rows_count, batch):

        batch_data = slice_data(i, batch, dataset=dataset)

        result = evaluate(
            query_engine=query_engine,
            metrics=metrics,
            dataset=batch_data,
            llm=llm,
            embeddings=embeddings,
        )

        rdf = result.to_pandas()
        results_df = pd.concat([results_df, rdf], ignore_index=True)
        print(f"Processed batch {i // batch + 1}:")
        print(rdf)
    print(results_df)
    results_df.to_csv("results.csv", index=False)
    return results_df
