# Knowledge Graph Builder

This project lets you build a Knowledge Graph from unstructured data (.md, .py files for now).

## Table of Contents

- [Installation from Source](#installation-from-source)
- [Usage](#usage)
  <!--- [Data Preparation](#data-preparation)-->
  <!--- [LLM Setup](#llm-setup)-->
  <!--- [Build Graph Index](#build-graph-index)-->

## Installation from Source

Follow these instructions to set up the project:

```bash
git clone https://github.com/debrupf2946/KnowledgeGraphBuilder.git
cd KnowledgeGraphBuilder
pip3 install -r requirements.txt
```

## Usage

### Data Preparation

1. First, create or import a data directory at the root folder containing documents (.md files).
2. Copy the path of the directory.
3. Load and chunk the documents using `load_directory(PATH)`.

```python
documents = load_directory("/data")
```

### LLM Setup

Users need to set up the LLM (llama3) locally to build the Knowledge Graph.

1. Initialize the LLM with `initialize_llm()`.
2. The default parameters are:  
    - `base_url="http://localhost:11434"` (Ollama server)
    - `model="llama3"`
    - `chunk_size = 512`
3. Change the parameters as needed.

```python
initialize_llm()
```

### Build Graph Index

1. Build the Knowledge Graph using the [documents](#data-preparation).
2. Call `build_graph(documents)` to create an index.
3. This will also save `Graph_visualization.html`, which can be opened in a browser to visualize the Knowledge Graph.

```python
index = build_graph(documents)
```

4. Save the `index` as a pickle file.

```python
save_index(index)
```
following example can referred for detailed implementation
[GraphRag Example Notebook](Example/GraphRagExample.ipynb)
