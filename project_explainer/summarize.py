
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from project_processor import (download_github_repo, 
                               extract_project_description_from_readme, 
                               extract_headings_with_paragraphs_from_markdown,
                               remove_tables_from_markdown,
                               remove_code_blocks_from_markdown,
                               remove_images_from_markdown,
                               remove_links_from_markdown)
import os
from jinja2 import Template


class Explainer():
    def __init__(self, base_model_id: str, device: str = "cpu") -> None:
        """
        Initializes the Explainer object.

        Args:
            base_model_id: The ID or path to the base model.
            device: The device to use for model inference (default is "cpu").

        Raises:
            ValueError: If the provided base model ID or path is invalid.
        """
        self.base_model_id = base_model_id
        self.device = device
        self.tokenizer=AutoTokenizer.from_pretrained(base_model_id)
        try:
            # support decoder only models
            if self.device == "cuda":
                self.model=AutoModelForCausalLM.from_pretrained(base_model_id, return_dict=True).to("cuda")
            else:
                self.model=AutoModelForCausalLM.from_pretrained(base_model_id, return_dict=True)
            self.brief_prompt_template = "{{ prompt }}\nExplain the above : "
        except Exception as e:
            # support encoder decoder models
            try:
                if self.device == "cuda":
                    self.model=AutoModelForSeq2SeqLM.from_pretrained(base_model_id, return_dict=True).to("cuda")
                else:
                    self.model=AutoModelForSeq2SeqLM.from_pretrained(base_model_id, return_dict=True)
                self.brief_prompt_template = "summarize: {{ prompt }}"
            except Exception as e2:
                raise ValueError(str(e), str(e2))

    def _fill_template(self, template_string: str, variables: dict) -> str:
        """
        Fills in variables in a template string using the provided dictionary and returns the filled template.

        Args:
            template_string: The template string with variables to be filled.
            variables: A dictionary containing the variable names and their corresponding values.

        Returns:
            The filled template string.

        Raises:
            TypeError: If the template_string is not a string or variables is not a dictionary.
        """
        template = Template(template_string)
        filled_template = template.render(variables)
        return filled_template
    
    def _model_gen(self, prompt: str) -> str:
        """
        Generates a response using a hugging face transformer model based on the provided prompt.

        Args:
            prompt: The input prompt for generating the response.

        Returns:
            The generated response as a string.

        Raises:
            TypeError: If the prompt is not a string.
        """
        inputs=self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
        output = self.model.generate(inputs, min_length=256, max_length=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def brief(self, github_url: str, branch: str = "main") -> dict:
        """
        Generates a brief summary of a project based on its README file.

        Args:
            github_url: The URL of the GitHub repository.
            branch: The branch name to download (default is "main").

        Returns:
            A dictionary containing the original prompt, prepared prompt, and the generated summary.

        Raises:
            ValueError: If the README.md file is not found.
        """
        repo_path = download_github_repo(github_url, branch)
        readme_path = os.path.join(repo_path, "README.md")
        if not os.path.exists(readme_path):
            raise ValueError("README.md not found")
        project_description = extract_project_description_from_readme(readme_path)
        prompt = {"prompt": project_description}
        prepared_prompt = self._fill_template(self.brief_prompt_template, prompt)
        summary=self._model_gen(prepared_prompt)
        return {"prompt": prompt, "prepared_prompt": prepared_prompt, "summary": str(summary)}
    
    def outline(self, github_url: str, branch: str = "main") -> dict:
        """
        Generates an outline of a project based on its README file.

        Args:
            github_url: The URL of the GitHub repository.
            branch: The branch name to download (default is "main").

        Returns:
            A dictionary containing the outline with headings as keys and generated summaries as values.

        Raises:
            ValueError: If the README.md file is not found.
        """
        repo_path = download_github_repo(github_url, branch)
        readme_path = os.path.join(repo_path, "README.md")
        if not os.path.exists(readme_path):
            raise ValueError("README.md not found")
        headings_and_paras = extract_headings_with_paragraphs_from_markdown(readme_path)
        outline_dict = {}
        for key,  value in headings_and_paras.items():
            content = remove_code_blocks_from_markdown(remove_images_from_markdown(remove_links_from_markdown(remove_tables_from_markdown(value))))
            prompt = {"prompt": content}
            prepared_prompt = self._fill_template(self.brief_prompt_template, prompt)
            outline_dict[key] = self._model_gen(prepared_prompt)
        return outline_dict
