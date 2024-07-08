# Experiments
Spent major portion of my time in this Gsoc projects is been spent in experimenting with different models,embedding,libraries

## Knowledge Graph from docs
Majority of the docuumentation of libraries are stored in form of html,md files in their github repo

we first used llama index document loaders to load all documents with .md extentiona and performed chunking and created an Document instance of them. 

### Model Selection
 We need an model that is OpenSource,can work on free colab version to begin with
also for better knowledge graph we Quantized models above 20GB size to 4bits using bitsandbites configureation
we tried following llms
- Gemini
- Zypher(4-bit)
- llama3(ollama-version)
- codellama(ollama--version)
- cohere/aya(4-bit quantized)
- phi3(ollama)
- phi3:medium(ollama)
- neural-chat(ollama)
- gemma2(ollama)
- mistral(ollama)

### Embeddings 
for embeddings we tried
- microsoft/codebert-base
- Salesforce/embedding

### Libraries
At initial phase we are looking for libraries out there in community that solves the problem of building Knowledge Graphs
- llama-index graph builder
- Neo4jGraphBuilder
- networkX graph builder

### Table
| Model                       | Embeddings         | Libraries                 | Remarks    | Documents      | artifacts                                                                                                                                
|:----------------------------|:-------------------|:--------------------------|:-----------|:---------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
| gemma2(ollama)  | microsoft/codebert-base | llama-index graph builder | nil          | [data_keras](artifacts/data_keras) | -[viz](artifacts/gemma2/Graph_visualization_gemma2_mscb.html)<br/>[index](artifacts/gemma2/gemma2graphIndex.pkl)                         |
| mistral(ollama)  | microsoft/codebert-base | llama-index graph builder | nil          | [data_keras](artifacts/data_keras)    | -[viz](artifacts/mistral/Graph_visualization_mistral_mscb.html)<br/>[index](artifacts/mistral/mistralgraphIndex.pkl)                     |
| neural-chat(ollama)  | microsoft/codebert-base | llama-index graph builder | nil            | [data_keras](artifacts/data_keras)    | -[viz](artifacts/neural_chat/Graph_visualization_neuralchat_mscb.html)<br/>[index](artifacts/neural_chat/graphIndex_neuralchat_mscb.pkl) |
| phi3:medium(ollama)  | microsoft/codebert-base | llama-index graph builder | nil            | [data_keras](artifacts/data_keras)     | -[viz](artifacts/phi3-med/Graph_visualization_phi3-med_mscb.html)<br/>[index](artifacts/phi3-med/graphIndex_phi3_medium_mscb.pkl)        |
| phi3(ollama)  | microsoft/codebert-base | llama-index graph builder | nil            | [data_keras](artifacts/data_keras)      | -[viz](artifacts/phi3/Graph_visualization_phi3_mscb.html)<br/>[index](artifacts/phi3/graphIndex_phi3_mscb.pkl)                                                                  |
| gpt-4o| open-ai| Neo4jGraphBuilder        | k          | keras-io       |                                                                                                                                          |
| Gemini                      | gemini             | llama-index graph builder | k          | keras-nlp      | - [viz](artifacts/vizualization/ex1.html)                                                                                                |
| Gemini                      | gemini             | llama-index graph builder | Rate-error | keras-io       |                                                                                                                                          |
| Gemini                      | microsoft/codebert-base | llama-index graph builder | k          | keras-nlp      | - [viz](artifacts/vizualization/gem_mcode_k_nlp.html)                                                                                    |
| Zypher(4-bit)               | microsoft/codebert-base | llama-index graph builder | k          | keras-nlp      | -[viz](artifacts/vizualization/zy_knlp.html)                                                                                             |
| Zypher(4-bit)               | microsoft/codebert-base | llama-index graph builder | k          | keras-io       | -[viz](artifacts/vizualization/examp.html)                                                                                               |
| llama3(ollama-version)      | microsoft/codebert-base | llama-index graph builder | k          | keras-nlp      | -[viz](artifacts/vizualization/Graph_visualization.html)                                                                                 
| codellama(ollama--version)  | microsoft/codebert-base | llama-index graph builder | k          | keras-nlp      | -[viz](artifacts/vizualization/code_1.html)                                                                                              
| gpt-neo-2.7B-4bit-quantized | microsoft/codebert-base | llama-index graph builder | k          | keras-nlp      | [viz](artifacts/vizualization/graph_gpt3-neo.html)                                                                                       