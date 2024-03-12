import logging
from git import Repo
import os
import requests
import base64

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def download_github_repo(repo_url: str, branch: str = "main") -> str:
    """
    Download a GitHub repository from the provided URL.

    Args:
        repo_url (str): The URL of the GitHub repository.
        branch (str): The branch of the GitHub repository.

    Returns:
        repo_path (str): Absolute path to downloaded repo
    """
    repo_name = repo_url.split("/")[-1].split(".")[0]
    repo_path = os.path.abspath(repo_name)

    Repo.clone_from(repo_url, repo_name, branch=branch)

    logger.info(f"Repository '{repo_name}' downloaded successfully!")
    return repo_path

def download_github_readme_file(repo_url, branch: str = "main"):
    """
    Download a README.md file in the GitHub repository from the provided URL.

    Args:
        repo_url (str): The URL of the GitHub repository.
        branch (str): The branch of the GitHub repository.

    Returns:
        readme_path (str): Absolute path to the downloaded README.md file
    """
    
    username, repo_name = repo_url.split('/')[-2:]
    url = f"https://api.github.com/repos/{username}/{repo_name}/readme?ref={branch}"
    
    readme_files_directory = "repo_readme_files"
    if not os.path.exists(readme_files_directory):
        os.makedirs(readme_files_directory)
    
    readme_path = os.path.join(readme_files_directory, f'{username}_{repo_name}_{branch}.md')
    response = requests.get(url)
    if response.status_code == 200:
        readme_content = response.json()['content']
        readme_content = base64.b64decode(readme_content).decode('utf-8')
        with open(readme_path, 'w') as readme_file:
            readme_file.write(readme_content)
        logger.info(f"README.md from Repository '{repo_name}', branch '{branch}' downloaded successfully.")
    else:
        logger.info(f"Failed to download README.md from Repository '{repo_name}', branch '{branch}'")
        
    return readme_path
