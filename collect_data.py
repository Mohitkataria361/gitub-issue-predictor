\
"""
Phase 3.1: Enhanced GitHub issue collection.

Collects closed issues together with repository metadata and engineered
features that are available at issue creation time.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.github.com"
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def get_headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN in .env")

    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def check_rate_limit(headers):
    r = requests.get(f"{API_BASE}/rate_limit", headers=headers)
    r.raise_for_status()
    core = r.json()["resources"]["core"]
    print(f"Rate Limit: {core['remaining']}/{core['limit']}")
    return core["remaining"]


def get_repo_metadata(repo, headers):
    url = f"{API_BASE}/repos/{repo}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    d = r.json()

    return {
        "repo_stars": d["stargazers_count"],
        "repo_forks": d["forks_count"],
        "repo_watchers": d["watchers_count"],
        "repo_open_issues": d["open_issues_count"],
        "repo_language": d["language"],
        "repo_created_at": d["created_at"],
        "repo_archived": d["archived"],
        "repo_size": d["size"],
        "default_branch": d["default_branch"],
    }


def issue_to_dict(issue, repo, repo_meta):
    body = issue.get("body") or ""

    labels = issue.get("labels") or []
    label_names = [
        l["name"] if isinstance(l, dict) else l
        for l in labels
    ]
    labels_lower = [x.lower() for x in label_names]

    def has(keyword):
        return any(keyword in l for l in labels_lower)

    user = issue.get("user") or {}

    num_code_blocks = body.count("```")

    return {
        # ---------------- Repository ----------------
        "repo": repo,
        **repo_meta,

        # ---------------- Issue ----------------
        "issue_number": issue["number"],
        "issue_url": issue["html_url"],
        "title": issue["title"],
        "state": issue["state"],
        "created_at": issue["created_at"],
        "closed_at": issue.get("closed_at"),
        "comments": issue.get("comments", 0),
        "reactions": issue.get("reactions", {}).get("total_count", 0),

        # ---------------- Author ----------------
        "author_login": user.get("login"),
        "author_type": user.get("type"),
        "author_association": issue.get("author_association"),

        # ---------------- Body ----------------
        "body_len": len(body),
        "body_word_count": len(body.split()),
        "body_has_code_block": num_code_blocks > 0,
        "num_code_blocks": num_code_blocks,
        "num_urls": len(re.findall(r"https?://", body)),
        "num_mentions": len(re.findall(r"@\w+", body)),

        # ---------------- Labels ----------------
        "labels": label_names,
        "num_labels": len(label_names),
        "has_bug_label": has("bug"),
        "has_enhancement_label": has("enhancement") or has("feature"),
        "has_documentation_label": has("documentation") or has("docs"),
        "has_help_wanted": has("help wanted"),
        "has_good_first_issue": has("good first issue"),

        # ---------------- Misc ----------------
        "has_milestone": issue.get("milestone") is not None,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def collect_repo(repo, state, max_pages, headers):
    print(f"\nCollecting {repo}")

    repo_meta = get_repo_metadata(repo, headers)

    issues = []

    for page in range(1, max_pages + 1):

        r = requests.get(
            f"{API_BASE}/repos/{repo}/issues",
            headers=headers,
            params={
                "state": state,
                "page": page,
                "per_page": 100,
                "sort": "created",
                "direction": "desc",
            },
        )

        if r.status_code == 403:
            print("Rate limited.")
            break

        r.raise_for_status()

        batch = r.json()

        if not batch:
            break

        for item in batch:
            if "pull_request" in item:
                continue

            issues.append(
                issue_to_dict(item, repo, repo_meta)
            )

        print(f"Page {page}: {len(issues)} issues")

        time.sleep(0.3)

    return issues


def main(repos, state, max_pages):
    headers = get_headers()
    check_rate_limit(headers)

    all_issues = []

    for repo in repos:
        all_issues.extend(
            collect_repo(repo, state, max_pages, headers)
        )

    os.makedirs(RAW_DIR, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    path = os.path.join(
        RAW_DIR,
        f"issues_{state}_{ts}.json"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=2)

    print(f"\nSaved {len(all_issues)} issues")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--repos", required=True)
    parser.add_argument("--state", default="closed",
                        choices=["closed", "open", "all"])
    parser.add_argument("--max_pages", type=int, default=20)

    args = parser.parse_args()

    repos = [r.strip() for r in args.repos.split(",")]

    main(repos, args.state, args.max_pages)
