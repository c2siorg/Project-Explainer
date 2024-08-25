# Graph Index Retriever

This module provides methods for loading a graph index from a pickle file and querying it using a `llama_index` query engine.

## Usage

### Loading Graph Index from Pickle File

1. Load the graph index from a pickle file using `get_index_from_pickle(file_path)`.
2. By default, the file path is set to `"results/graphIndex.pkl"`.

```python
from graph_rag.graph_retrieval.graph_retrieval import get_index_from_pickle

index = get_index_from_pickle("path/to/your/graphIndex.pkl")
```

### Setting Up the Query Engine

1. Initialize the LLM with `initialize_llm()`.
2. Create a query engine using `get_query_engine(index, with_embedding=False, similarity_top_k=5)`.
   - `index`: The loaded `llama_index` graph index object.
   - `with_embedding` (bool): Set to `True` to query the graph index with embeddings. Default is `False`.
   - `similarity_top_k` (int): Number of top similar chunks to provide as context to LLM for responding to the query. Default is `5`.

```python
from graph_rag.graph_retrieval.graph_retrieval import get_query_engine

query_engine = get_query_engine(index, with_embedding=False, similarity_top_k=5)
```

### Querying the Graph Index

1. Query the graph index using `graph_query(query, query_engine)`.
   - `query` (str): The query to be answered using `graph_rag`.
   - `query_engine`: The `llama_index` query engine object.

```python
from graph_rag.graph_retrieval.graph_retrieval import graph_query

response = graph_query("Your query here", query_engine)
print(response)
```
## Advanced Training with QLoRA and P-Tuning

>fine-tuning LLMs on data(masked language or Next toke Prediction) for few epochs, may result in better retrieval and response

### 1. Setup

To use QLoRA and P-Tuning, ensure your environment is set up with the required libraries and that your model and dataset configurations are defined in a `config.yaml` file.

### 2. Finetuning with QLoRA

Use the QLoRA method for efficient fine-tuning by passing the appropriate configurations in your `config.yaml`. This method is ideal when working with large models on limited hardware.

```bash
python qlora_adapter.py --config path/to/config.yaml
```
Execute the training script with the `--config` argument to specify your configuration file:

### 3. Fine-Tuning with P-Tuning 

P-Tuning allows for parameter-efficient prompt-based fine-tuning. Adjust the number of virtual tokens and other related parameters in the `config.yaml` to customize the training process.

```bash
python p_tuning.py--config path/to/config.yaml
```
Execute the training script with the `--config` argument to specify your configuration file:






This will start the training process using the specified method (QLoRA or P-Tuning) and configurations.


