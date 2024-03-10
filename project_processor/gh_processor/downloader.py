import logging
from git import Repo
import os

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
        # Check if the repository directory already exists
    if os.path.exists(repo_path):
        logger.info(f"Repository '{repo_name}' already exists at '{repo_path}'. Skipping cloning.")
        return repo_path
    
    try:
        repo_name = repo_url.split("/")[-1].split(".")[0]
        repo_path = os.path.abspath(repo_name)

        Repo.clone_from(repo_url, repo_name, branch=branch)

        logger.info(f"Repository '{repo_name}' downloaded successy!")
        return repo_path
    except Exception as e:
        logger.error(f"Error whule donwloading repository: {str(e)}")
        raise
