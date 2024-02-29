from gh_explainer import Explainer

def summarize(summarization_type, github_project_url, github_project_branch="main", huggingface_model_id="gpt2"):
    gptExplainer = Explainer(huggingface_model_id)
    if summarization_type == "brief":
        return gptExplainer.brief(github_url=github_project_url, branch=github_project_branch)["summary"]
    return gptExplainer.outline(github_url=github_project_url, branch=github_project_branch)["summary"]


import streamlit as st
st.title("Github Project Explainer")

if "response" not in st.session_state:
    st.session_state.response=""

with st.sidebar:
    st.title("Enter Details of the Project")
    summary_level=st.selectbox("summary level",("brief","outLine"))
    github_project_url=st.text_input("github_project_url")
    github_project_branch=st.text_input("github_project_branch")
    huggingface_model_id=st.text_input("huggingface_model_id")
    if st.button("Run"):
        st.session_state.response=summarize(summary_level,str(github_project_url),str(github_project_branch),str(huggingface_model_id))



with st.container(border=True,height=100):

    st.markdown(st.session_state.response)