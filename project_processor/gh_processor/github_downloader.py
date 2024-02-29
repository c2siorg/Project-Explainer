import logging
from git import Repo, GitCommandError
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
    try:
        Repo.clone_from(repo_url, repo_name, branch=branch)
    except GitCommandError as e:
        error_msg = str(e)
        if "fatal: destination path" in error_msg and "already exists" in error_msg:
            error_msg = f"The repository '{repo_name}' already exists on your system."
        elif "exit code(128)" in error_msg:
            error_msg = "Failed to clone the repository. Please check if the repository URL and branch are correct."
        else:
            error_msg = f"{error_msg}"
        raise Exception(error_msg)

    Repo.clone_from(repo_url, repo_name, branch=branch)

    logger.info(f"Repository '{repo_name}' downloaded successfully!")
    return repo_path
