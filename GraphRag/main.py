from tools import initialize_llm,load_directory
from knowledgeGraph import build_graph,save_index


initialize_llm()
documents=load_directory("/data")
index=build_graph(documents)
save_index(index)
