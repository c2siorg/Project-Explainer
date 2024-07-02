# Knowledge Graph Builder
This project let's you build knowledge graph from unstructured data(.md,.py files for now)

## Table of Contents

- [Installation from source](#installation)
- [Usage](#usage)
<!--- [Features](#features)-->

## Installation

Instructions on how to setup the project:

```bash
git clone https://github.com/debrupf2946/KnowlwdgeGraphBuilder.git
cd KnowlwdgeGraphBuilder
pip3 install -r requirements.txt
```
## Usage
### DataPreperation
1. first create/import a data directory  at root folder which contains Docs(.md files)
2. copy the path of the directory
3. load and chunk the documents with ***load_directory(PATH)***
```bash
documents=load_directory("/data")
```
### LLm Setup
***Users need to setup the llm(llama3) locally and build KnowledgeGraph***
1. Initialize the llm with ***initialize_llm()***
2. Defaults:  
            base_url="http://localhost:11434" (ollama server)   
            model="llama3",  
            chunk_size = 512
3.change the parameters as per choice/usage
```bash
initialize_llm()
```
### Build GrpahIndex
1.Now simply build the KG using the [documents](###DataPreperation)  
2. call ***build_graph(documents)*** this will creat an index  
3. It wil also save Graph_visualization.html,which can be opened in browser to visualise the KowledgeGraph built
```python
index=build_graph(documents)
```
4. Save the ***index*** as pickle file
```python
save_index(index)
```