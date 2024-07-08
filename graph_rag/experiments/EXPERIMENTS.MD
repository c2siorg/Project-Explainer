# Experiments

The major portion of my time in the first phase of the GSoC project has been spent experimenting with different models, embeddings, and libraries.

## Knowledge Graph from Documentation

The majority of the documentation for libraries is stored in the form of HTML and markdown files in their GitHub repositories.

We first used llama-index document loaders to load all documents with the .md extension. We then performed chunking and created a Document instance of them.

## Knowledge Graph Using Code Embeddings

Implementation of the idea can be found here: [Colab](https://colab.research.google.com/drive/1uguR76SeMAukN4uAhKuXU_ja8Ik0s8Wj#scrollTo=CUgtX5D1Tl_x).

The idea is to separate code blocks or take code and split it using a code splitter, then pass it to a model for building a Knowledge Graph using code embeddings. I used:
- Salesforce/codegen2-7B_P quantized (4-bit)
- Salesforce/codet5p-110m-embedding
- Python files in Keras-io

### Model Selection

We need a model that is open source and can work on the free Colab version to begin with. For a better knowledge graph, we quantized models above 20GB to 4 bits using bitsandbytes configuration. We tried the following LLMs:
- gemini pro
- [QuantiPhy/zephyr-7b-beta(4bit-quantized)**](https://huggingface.co/QuantiPhy/zephyr-7b-beta-4bit-quantized)
- llama3 (Ollama version)
- codellama (Ollama version)
- [QuantiPhy/aya-23-8B (4bit quantized)**](https://huggingface.co/QuantiPhy/aya-23-8B-4bq)
- gpt-neo-2.7B(4bit-quantized)
- [Salesforce/codegen2-7B_P(4bit-quantized)**](https://huggingface.co/QuantiPhy/Salesforce_codegen2-7B_P)
- phi3 (Ollama)
- phi3:medium (Ollama)
- neural-chat (Ollama)
- gemma2 (Ollama)
- mistral (Ollama)   
** all these models,I have 4bit-quantized them using bitsandbytes
### Embeddings

For embeddings, we tried:
- microsoft/codebert-base
- Salesforce/codet5p-110m-embedding

### Libraries

In the initial phase, we are looking for libraries in the community that solve the problem of building Knowledge Graphs:
- [llama-index knowledge-graph builder](https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/indices/knowledge_graph)
- [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder)
- [graph_builder](https://github.com/sarthakrastogi/graph-rag)

### Table

| Model                       | Embeddings           | Libraries                  | Remarks     | Documents                | Artifacts                                                                                                                                                                                                                                                                            |
|:----------------------------|:---------------------|:---------------------------|:------------|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gemma2 (Ollama)             | microsoft/codebert-base | llama-index graph builder | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/gemma2/Graph_visualization_gemma2_mscb.html)<br/>[index](artifacts/gemma2/gemma2graphIndex.pkl)<br/>[collab](https://colab.research.google.com/drive/1q7FED2Lapk3D7ibqkO3NkqZ6iPNZ_x6H?usp=sharing)                                                                  |
| mistral (Ollama)            | microsoft/codebert-base | llama-index graph builder | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/mistral/Graph_visualization_mistral_mscb.html)<br/>[index](artifacts/mistral/mistralgraphIndex.pkl)<br/>[collab](https://colab.research.google.com/drive/1q7FED2Lapk3D7ibqkO3NkqZ6iPNZ_x6H?usp=sharing)                                                              |
| neural-chat (Ollama)        | microsoft/codebert-base | llama-index graph builder | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/neural_chat/Graph_visualization_neuralchat_mscb.html)<br/>[index](artifacts/neural_chat/graphIndex_neuralchat_mscb.pkl)<br/>[collab](https://colab.research.google.com/drive/1cM6ujhiKM1v0bRYVN9F9UEgjYlwkBTt9?usp=sharing)                                          |
| phi3:medium (Ollama)        | microsoft/codebert-base | llama-index graph builder | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/phi3-med/Graph_visualization_phi3-med_mscb.html)<br/>[index](artifacts/phi3-med/graphIndex_phi3_medium_mscb.pkl)<br/>[collab](https://colab.research.google.com/drive/1cM6ujhiKM1v0bRYVN9F9UEgjYlwkBTt9?usp=sharing)                                                 |
| phi3 (Ollama)               | microsoft/codebert-base | llama-index graph builder | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/phi3/Graph_visualization_phi3_mscb.html)<br/>[index](artifacts/phi3/graphIndex_phi3_mscb.pkl)<br/>[collab](https://colab.research.google.com/drive/1cM6ujhiKM1v0bRYVN9F9UEgjYlwkBTt9?usp=sharing)                                                                    |
| gpt-4o                      | open-ai              | Neo4jGraphBuilder          | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/vizualization/visualisation.png)                                                                                                                                                                                                                                     |
| Gemini                      | gemini               | llama-index graph builder | nil         | [keras-nlp](https://github.com/keras-team/keras-io/blob/master/templates/keras_nlp/index.md) | [viz](artifacts/vizualization/ex1.html)                                                                                                                                                                                                                                              |
| Gemini                      | gemini               | llama-index graph builder | Rate-error  | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) |                                                                                                                                                                                                                                                                                      |
| Gemini                      | microsoft/codebert-base | llama-index graph builder | nil         | [keras-nlp](https://github.com/keras-team/keras-io/blob/master/templates/keras_nlp/index.md) | [viz](artifacts/vizualization/gem_mcode_k_nlp.html)                                                                                                                                                                                                                                  |
| Zypher (4-bit)              | microsoft/codebert-base | llama-index graph builder | nil         | [keras-nlp](https://github.com/keras-team/keras-io/blob/master/templates/keras_nlp/index.md) | [viz](artifacts/vizualization/zy_knlp.html)                                                                                                                                                                                                                                          |
| Zypher (4-bit)              | microsoft/codebert-base | llama-index graph builder | nil         | [keras-io](https://github.com/keras-team/keras-io/tree/master/templates) | [viz](artifacts/vizualization/examp.html)                                                                                                                                                                                                                                            |
| llama3 (Ollama version)     | microsoft/codebert-base | llama-index graph builder | nil         | [keras-nlp](https://github.com/keras-team/keras-io/blob/master/templates/keras_nlp/index.md) | [viz](artifacts/vizualization/Graph_visualization.html)                                                                                                                                                                                                                              |
| codellama (Ollama version)  | microsoft/codebert-base | llama-index graph builder | nil         | [keras-nlp](https://github.com/keras-team/keras-io/blob/master/templates/keras_nlp/index.md) | [viz](artifacts/vizualization/code_1.html)                                                                                                                                                                                                                                           |
| gpt-neo-2.7B-4bit-quantized | microsoft/codebert-base | llama-index graph builder | nil         | [keras-nlp](https://github.com/keras-team/keras-io/blob/master/templates/keras_nlp/index.md) | [viz](artifacts/vizualization/graph_gpt3-neo.html)                                                                                                                                                                                                                                   |

### Notes
- ### [graph_builder](https://github.com/sarthakrastogi/graph-rag)   

  -  I explored graph_rag by Sarthak. It is fundamentally based on function calling (JSON output), and it works very well for powerful models. However, small-sized LLMs tend to make mistakes regardless of how well the prompt is crafted.
  - I tried and debugged the library, and this was my experience with it. I modified the system prompts, which led to fewer mistakes, and added a method to download .html files for visualization. Additionally, I added methods to use Ollama OS models.
  - [rough_codes](https://colab.research.google.com/drive/1q6T8mK-O2XKqY-iGFz6xdrzvqLzu73lm#scrollTo=H0QG6QUVub8T) contains codes/modification/implementation for the rep0