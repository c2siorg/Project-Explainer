import gradio as gr
from gh_explainer import Explainer

def summarize(summarization_type, github_project_url, github_project_branch="main", huggingface_model_id="gpt2"):
    gptExplainer = Explainer(huggingface_model_id)
    if summarization_type == "brief":
        return gptExplainer.brief(github_url=github_project_url, branch=github_project_branch)["summary"]
    return gptExplainer.outline(github_url=github_project_url, branch=github_project_branch)

demo = gr.Interface(
    fn=summarize,
    inputs=[gr.Dropdown(["brief", "outline"], label="summary level"), "text", "text", "text"],
    outputs=["text"],
)
demo.launch()