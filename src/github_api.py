import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}


def parse_issue_url(url: str):
    """
    Example:
    https://github.com/microsoft/vscode/issues/251234

    Returns:
    owner, repo, issue_number
    """

    pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"

    match = re.search(pattern, url)

    if not match:
        raise ValueError("Invalid GitHub issue URL")

    owner = match.group(1)
    repo = match.group(2)
    issue_number = match.group(3)

    return owner, repo, issue_number


def fetch_issue(url: str):

    owner, repo, issue = parse_issue_url(url)

    api = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue}"

    response = requests.get(api, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(
            f"GitHub API Error {response.status_code}: {response.text}"
        )
    issue = response.json()

    repository = fetch_repository(owner, repo)

    issue["repository"] = repository
    return issue
def fetch_repository(owner: str, repo: str):
    """
    Fetch repository metadata.
    """

    api = f"https://api.github.com/repos/{owner}/{repo}"

    response = requests.get(
        api,
        headers=HEADERS
    )

    if response.status_code != 200:
        raise Exception(
            f"Repository fetch failed ({response.status_code})"
        )

    return response.json()
