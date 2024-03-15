from github import Github
import os

# Initialize a Github instance with an access token
g = Github(os.getenv('GITHUB_TOKEN'))

def get_repo_info(repo_name):
    repo = g.get_repo(repo_name)

    # Fetching repository metadata
    stars = repo.stargazers_count
    forks = repo.forks_count
    contributors = repo.get_contributors().totalCount

    print(f"Stars: {stars}, Forks: {forks}, Contributors: {contributors}")

def parse_issues(repo_name):
    repo = g.get_repo(repo_name)
    issues = repo.get_issues(state='open')

    print("\nOpen Issues:")
    for issue in issues:
        print(f"- {issue.title}: {issue.html_url}")

def parse_pull_requests(repo_name):
    repo = g.get_repo(repo_name)
    pull_requests = repo.get_pulls(state='open', sort='created')

    print("\nOpen Pull Requests:")
    for pr in pull_requests:
        print(f"- {pr.title}: {pr.html_url}")

if __name__ == "__main__":
    repo_name = "c2siorg/Project-Explainer"

    print(f"Fetching information for {repo_name}")
    get_repo_info(repo_name)
    parse_issues(repo_name)
    parse_pull_requests(repo_name)
