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
- cohere/aya(4-bit quantized). 

### Embeddings 
for embeddings we tried
- microsoft/Codebert
- Salesforce/embedding

### Libraries
At initial phase we are looking for libraries out there in community that solves the problem of building Knowledge Graphs
- llama-index graph builder
- Neo4jGraphBuilder
- networkX graph builder

### Table
| Model                       | Embeddings         | Libraries                 | Remarks    | Documents | artifacts                                              
|:----------------------------|:-------------------|:--------------------------|:-----------|:----------|:-------------------------------------------------------|
| gpt-4o| open-ai| Neo4jGraphBuilder        | k          | keras-io  |                                                        |
| Gemini                      | gemini             | llama-index graph builder | k          | keras-nlp | - [viz](artifacts/vizualization/ex1.html)              |
| Gemini                      | gemini             | llama-index graph builder | Rate-error | keras-io  |                                                        |
| Gemini                      | microsoft/Codebert | llama-index graph builder | k          | keras-nlp | - [viz](artifacts/vizualization/gem_mcode_k_nlp.html)  |
| Zypher(4-bit)               | microsoft/Codebert | llama-index graph builder | k          | keras-nlp | -[viz](artifacts/vizualization/zy_knlp.html)           |
| Zypher(4-bit)               | microsoft/Codebert | llama-index graph builder | k          | keras-io  | -[viz](artifacts/vizualization/examp.html)             |
| llama3(ollama-version)      | microsoft/Codebert | llama-index graph builder | k          | keras-nlp | -[viz](artifacts/vizualization/Graph_visualization.html)
| codellama(ollama--version)  | microsoft/Codebert | llama-index graph builder | k          | keras-nlp | -[viz](artifacts/vizualization/code_1.html)            
| gpt-neo-2.7B-4bit-quantized | microsoft/Codebert | llama-index graph builder | k          | keras-nlp | [viz](artifacts/vizualization/graph_gpt3-neo.html)     