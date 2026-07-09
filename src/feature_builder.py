
"""
feature_builder.py

Creates the same feature columns used during training from a live GitHub issue.
"""

from datetime import datetime, timezone
import re
import pandas as pd


def build_features(issue: dict) -> pd.DataFrame:
    repo = issue["repository"]
    body = issue.get("body") or ""
    title = issue.get("title") or ""

    created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
    repo_created = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)

    labels = [l["name"].lower() for l in issue.get("labels", [])]

    def has(keyword):
        return any(keyword in label for label in labels)

    row = {
        "title_len_chars": len(title),
        "title_len_words": len(title.split()),
        "body_len": len(body),
        "body_word_count": len(body.split()),
        "num_labels": len(labels),
        "comments": issue.get("comments", 0),
        "reactions": issue.get("reactions", {}).get("total_count", 0),
        "num_urls": len(re.findall(r"https?://", body)),
        "num_mentions": len(re.findall(r"@\w+", body)),
        "num_code_blocks": body.count("```"),
        "repo_stars": repo.get("stargazers_count", 0),
        "repo_forks": repo.get("forks_count", 0),
        "repo_watchers": repo.get("watchers_count", 0),
        "repo_open_issues": repo.get("open_issues_count", 0),
        "repo_size": repo.get("size", 0),
        "repo_age_days": (now - repo_created).days,
        "created_hour_utc": created.hour,
        "created_day_of_week": created.weekday(),
        "created_month": created.month,
        "title_has_number": any(c.isdigit() for c in title),
        "title_has_question": "?" in title,
        "body_has_code_block": "```" in body,
        "is_bot_author": issue.get("user", {}).get("type", "").lower() == "bot",
        "is_owner_or_member": issue.get("author_association", "") in ["OWNER", "MEMBER", "COLLABORATOR"],
        "is_weekend": created.weekday() >= 5,
        "is_business_hours": 9 <= created.hour <= 18,
        "repo_archived": repo.get("archived", False),
        "has_bug_label": has("bug"),
        "has_enhancement_label": has("enhancement") or has("feature"),
        "has_documentation_label": has("documentation") or has("docs"),
        "has_help_wanted": has("help wanted"),
        "has_good_first_issue": has("good first issue"),
        "has_milestone": issue.get("milestone") is not None,
        "repo_language": repo.get("language") or "Unknown",
        "author_association": issue.get("author_association") or "NONE",
        "default_branch": repo.get("default_branch", "main"),
    }

    return pd.DataFrame([row])
