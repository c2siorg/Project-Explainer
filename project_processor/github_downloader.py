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
    repo_name = repo_url.split("/")[-1].split(".")[0]
    repo_path = os.path.abspath(repo_name)

    Repo.clone_from(repo_url, repo_name, branch=branch)

    logger.info(f"Repository '{repo_name}' downloaded successfully!")
    return repo_path
