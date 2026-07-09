"""
Phase 2: Collect closed issues from public GitHub repos via the REST API.

Usage:
    python collect_data.py --repos pandas-dev/pandas,psf/requests --state closed --max_pages 20

Notes:
- Uses a personal access token for a 5,000 req/hour rate limit (vs 60/hour
  unauthenticated). See README.md Phase 0 to create one.
- The /issues endpoint returns BOTH issues and pull requests -- PRs have a
  "pull_request" key present, which we filter out.
- We only collect closed issues (see README for why: censoring).
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
API_BASE = "https://api.github.com"


def get_headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing GITHUB_TOKEN. Copy .env.example to .env and add your "
            "personal access token. See README.md Phase 0."
        )
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def check_rate_limit(headers):
    resp = requests.get(f"{API_BASE}/rate_limit", headers=headers)
    resp.raise_for_status()
    core = resp.json()["resources"]["core"]
    print(f"Rate limit: {core['remaining']}/{core['limit']} remaining")
    return core["remaining"]


def issue_to_dict(issue: dict, repo: str) -> dict:
    """Extract fields available at/near issue creation time, plus outcome fields
    for the label. Handles missing user (deleted account) and missing labels
    gracefully.
    """
    created_at = issue["created_at"]
    closed_at = issue.get("closed_at")

    user = issue.get("user") or {}
    author_login = user.get("login")
    author_type = user.get("type")  # "User" or "Bot"

    labels = issue.get("labels") or []
    label_names = [lbl["name"] if isinstance(lbl, dict) else lbl for lbl in labels]

    body = issue.get("body") or ""

    return {
        "repo": repo,
        "number": issue["number"],
        "title": issue["title"],
        "body_len": len(body),
        "body_has_code_block": "```" in body,
        "author_login": author_login,
        "author_type": author_type,
        "author_association": issue.get("author_association"),  # OWNER/MEMBER/CONTRIBUTOR/NONE/etc.
        "num_labels": len(label_names),
        "labels": label_names,
        "created_at": created_at,
        "closed_at": closed_at,
        "comments": issue.get("comments", 0),
        "state": issue["state"],
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def collect_repo(repo: str, state: str, max_pages: int, headers: dict) -> list:
    issues = []
    page = 1
    while page <= max_pages:
        url = f"{API_BASE}/repos/{repo}/issues"
        params = {
            "state": state,
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "desc",
        }
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 403:
            print("Rate limited mid-collection. Stopping early for this repo.")
            break
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        for item in batch:
            if "pull_request" in item:
                continue  # skip PRs, keep issues only
            issues.append(issue_to_dict(item, repo))

        print(f"  {repo}: page {page}, {len(batch)} items ({len(issues)} issues so far)")
        page += 1
        time.sleep(0.3)  # polite pacing, well under rate limit

    return issues


def main(repos, state, max_pages):
    headers = get_headers()
    check_rate_limit(headers)

    all_issues = []
    for repo in repos:
        print(f"Collecting r/{repo}...".replace("r/", ""))
        all_issues.extend(collect_repo(repo, state, max_pages, headers))

    os.makedirs(RAW_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RAW_DIR, f"issues_{state}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_issues, f, indent=2)

    print(f"\nSaved {len(all_issues)} issues to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos", required=True, help="Comma-separated owner/repo, e.g. pandas-dev/pandas,psf/requests"
    )
    parser.add_argument("--state", default="closed", choices=["closed", "open", "all"])
    parser.add_argument("--max_pages", type=int, default=20, help="100 issues per page")
    args = parser.parse_args()

    repo_list = [r.strip() for r in args.repos.split(",")]
    main(repo_list, args.state, args.max_pages)
