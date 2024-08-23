
# Knowledge Graph Evaluation

This module provides methods to evaluate the performance of GraphRag. The following integrations are available for evaluation:

- **Llama-Index Evaluation Pack**
- **Ragas Evaluation Pack**

Additionally, this module includes scripts for creating custom test datasets to benchmark and evaluate GraphRag.

## Getting Started

### Evaluating Your Knowledge Graph

You can easily evaluate the performance of your query engine using this module.

#### 1. Load and Evaluate Your Dataset

Use the `load_test_dataset` function to load your dataset and directly evaluate it using the `evaluate` function. This method handles all necessary steps, including batching the data.

```python
from your_module import load_test_dataset, evaluate

# Step 1: Load the dataset from a pickle file
dataset_path = "random/dataset_200_llama3.pkl"
test_dataset = load_test_dataset(dataset_path)
```

> **Note:** `test_dataset` is a list of Llama-Index `Document` objects.

```python
# Step 2: Define the language model and embedding
llm = Ollama(base_url="http://localhost:11434", model="codellama")
embedding = HuggingFaceEmbedding(model_name="microsoft/codebert-base")

# Step 3: Specify the metrics for evaluation
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# Step 4: Load the query engine (Llama-Index)
from graph_rag.graph_retrieval.graph_retrieval import get_index_from_pickle, get_query_engine

index = get_index_from_pickle("/content/Results/graphIndex.pkl")
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





