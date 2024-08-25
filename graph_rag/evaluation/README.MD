
# Knowledge Graph Evaluation

This module provides methods to evaluate the performance of GraphRag. The following integrations are available for evaluation:

- **Llama-Index Evaluation Pack**
- **Ragas Evaluation Pack**

Additionally, this module includes scripts for creating custom test datasets to benchmark and evaluate GraphRag.

## Getting Started
This section demonstrates how to use the functions provided in the module:

---

 ### 1. QA Generation and Critique

This module offers tools to generate question-answer (QA) pairs from input documents using a language model and critique them based on various criteria like groundedness, relevance, and standalone quality.

> #### Generate and Critique QA Pairs

To use this module, follow these steps:

#### 1. Generate QA Pairs

First, we prepare our dataset for generating QA pairs. In this example, we'll use Keras-IO documentation and Llama-Index's `SimpleDirectoryReader` to obtain `Document` objects.

```python
!git clone https://github.com/keras-team/keras-io.git

def get_data(input_dir="path/to/keras-io/templates"):
    reader = SimpleDirectoryReader(
        input_dir, 
        recursive=True, 
        exclude=["path/to/keras-io/templates/examples"]
    )
    docs = reader.load_data()

    splitter = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=20,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    documents = [Document(text=node.text, metadata=node.metadata) for node in nodes]
    
    return docs
    
# load the documents
documents=get_data()
```

Use the `qa_generator` function to generate QA pairs from your input documents.

```python
from evaluation.ragas_evaluation.QA_graphrag_testdataset import qa_generator

N_GENERATIONS = 20

# Generate the QA pairs
qa_pairs = qa_generator(documents, N_GENERATIONS)
```

#### 2. Critique the Generated QA Pairs

Once you have generated the QA pairs, critique them using the `critique_qa` function.

```python
from evaluation.ragas_evaluation.QA_graphrag_testdatasete import critique_qa

# Critique the generated QA pairs
critiqued_qa_pairs = critique_qa(qa_pairs)

# The critiqued pairs will include scores and evaluations for groundedness, relevance, and standalone quality
```

---
### 2. Evaluating Your Knowledge Graph with Llama-Index Evaluator Pack

This section demonstrates how to evaluate the performance of your query engine using the Llama-Index RAG evaluator pack.

> #### Evaluate Your Knowledge Graph with llama-index

To evaluate your query engine, follow these steps:
```shell
llamaindex-cli download-llamadataset PaulGrahamEssayDataset --download-dir ./data
```

```python
from evaluation.evaluation_llama_index import evaluate


# Path to your labeled RAG dataset
RAG_DATASET = "./data/rag_dataset.json"

# Define the language model and embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

llm = Ollama(base_url="http://localhost:11434", model="llama2")
embedding = HuggingFaceEmbedding(model_name="microsoft/codebert-base")

# Your query engine instance
from graph_rag.graph_retrieval.graph_retrieval import get_index_from_pickle, get_query_engine

index = get_index_from_pickle("path/to/graphIndex.pkl")
query_engine = get_query_engine(index)

# Evaluate the dataset
evaluation_results = evaluate(RAG_DATASET, query_engine)

# Review the results
print(evaluation_results)
```
| Metrics                      | RAG        | Base RAG  |
|------------------------------|------------|-----------|
| **Mean Correctness Score**    | 3.340909   |         0.934  |
| **Mean Relevancy Score**      | 0.750000   |    4.239       |
| **Mean Faithfulness Score**   | 0.386364   |   0.977        |
| **Mean Context Similarity Score** | 0.948765 |     0.977      |



This example shows how to quickly evaluate your query engine's performance using the Llama-Index RAG evaluator pack.


---
### 3. Evaluating Your Knowledge Graph with Ragas backend

You can easily evaluate the performance of your query engine using this module.

> #### Load and Evaluate Your Dataset with ragas

Use the `load_test_dataset` function to load your dataset and directly evaluate it using the `evaluate` function. This method handles all necessary steps, including batching the data.

```python
from evaluation.ragas_evaluation.evaluation_ragas load_test_dataset, evaluate

# Step 1: Load the dataset from a pickle file
dataset_path = "/content/keras_docs_embedded.pkl"
test_dataset = load_test_dataset(dataset_path)
```

> **Note:** `test_dataset` is a list of Llama-Index `Document` objects.

```python
# Step 2: Define the language model and embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

llm = Ollama(base_url="http://localhost:11434", model="codellama")
embedding = HuggingFaceEmbedding(model_name="microsoft/codebert-base")

# Step 3: Specify the metrics for evaluation
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# Step 4: Load the query engine (Llama-Index)
from graph_rag.graph_retrieval.graph_retrieval import get_index_from_pickle, get_query_engine

index = get_index_from_pickle("path/to/graphIndex.pkl")
query_engine = get_query_engine(index)

# Step 5: Evaluate the dataset
evaluation_results = evaluate(
    query_engine=query_engine,
    dataset=test_dataset,
    llm=llm,
    embeddings=embedding,
    metrics=metrics,
    # Default batch size is 4
)
```

**Output:**
```python
{'faithfulness': 0.0333, 'answer_relevancy': 0.9834, 'context_precision': 0.2000, 'context_recall': 0.8048}
```

```python
rdf = evaluation_results.to_pandas()
rdf.to_csv("results.csv", index=False)
```
---
**Detailed Result:**

| question                                      | contexts                                                                                                            | answer                                                                                                   | ground_truth                                                                                             | faithfulness | answer_relevancy | context_precision | context_recall |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|--------------|------------------|-------------------|----------------|
| What is mixed precision in computing?         | [Examples GPT-2 text generation Parameter…]                                                                        | Mixed precision is a technique used to improve…                                                          | A combination of different numerical precision…                                                             | 0.166667     | 0.981859         | 0.0               | 0.666667       |
| What is the title of the guide discussed in th... | [Available guides… Hyperparameter T…]                                                                              | The title of the guide discussed in the given…                                                           | How to distribute training                                                                                  | 0.000000     | 1.000000         | 0.0               | 1.000000       |
| What is Keras 3?                              | [No relationships found.]                                                                                          | Keras 3 is a new version of the popular deep l…                                                          | A deep learning framework that works with Tensor…                                                            | 0.000000     | 0.974711         | 0.0               | 0.500000       |
| What was the percentage boost in StableDiffusion... | [A first example: A MNIST convnet…]                                                                                | The percentage boost in StableDiffusion traini…                                                          | Over 150%                                                                                                    | 0.000000     | 0.970565         | 1.0               | 1.000000       |
| What are some examples of pretrained models av... | [No relationships found.]                                                                                          | Some examples of pre-trained models available…                                                           | BERT, OPT, Whisper, T5, StableDiffusion, YOLOv8…                                                             | 0.000000     | 0.989769         | 0.0               | 0.857143       |





