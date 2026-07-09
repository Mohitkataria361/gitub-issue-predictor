# GitHub Issue Resolution Time Predictor

Predicts how long a GitHub issue will stay open, using only features known
at (or shortly after) the moment the issue is filed: title/body content,
labels, author's relationship to the repo, and repo-level baseline.

Like the Reddit version, this is a "data creator" project -- you collect
the data yourself via GitHub's REST API, deal with the mess (issues vs PRs
mixed together, missing labels, bots opening issues, censored/still-open
issues), and only then build a model.

**Why GitHub instead of Reddit:** Reddit's Responsible Builder Policy
explicitly restricts using scraped Reddit data to train ML models without
written approval. GitHub's API terms have no equivalent restriction for
this kind of public-data analysis of open source repos.

---
## Phase 0: Get a GitHub Personal Access Token (2 minutes)

Unauthenticated requests are capped at 60/hour -- not enough to collect a
real dataset. A token bumps you to 5,000/hour.

1. Go to https://github.com/settings/tokens
2. Click **"Generate new token"** -> **"Generate new token (classic)"**
3. Give it a name like `issue-predictor`, no scopes needed (public repo
   read access doesn't require any checkboxes checked)
4. Generate, copy the token (starts with `ghp_...`)
5. Copy `.env.example` to `.env` and paste it in:
   ```
   GITHUB_TOKEN=ghp_your_token_here
   ```

---

## Phase 1: Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Phase 2: Collect data

```bash
python collect_data.py --repos pandas-dev/pandas,psf/requests --state closed --max_pages 20
```

**Key design decision -- censoring:** an issue that's still open hasn't
"failed to resolve," it just hasn't yet. The simplest honest approach
(what this script does) is to **only collect closed issues** for training.
Worth explicitly naming this limitation in your write-up: your model
answers "given that this issue eventually closes, how long did it take?"
not "will this issue ever close?"

**Key design decision -- issues vs PRs:** GitHub's REST API returns pull
requests mixed into the issues endpoint (a PR *is* an issue, technically).
This script filters them out, since PRs have a totally different lifecycle.

---

## Phase 3: Feature engineering

```bash
python build_features.py
```

See `build_features.py` for the full feature list and cleaning decisions.

---

## Phase 4: Train a model

```bash
python train_model.py
```

Baseline (median) vs Ridge vs XGBoost, evaluated on log-scaled resolution
time in hours (like Reddit scores, resolution times are heavily
right-skewed -- most issues close fast, a few linger for months).

---

## Phase 5: Streamlit demo

```bash
streamlit run app.py
```

## Project structure

```
github-issue-predictor/
├── .env.example
├── requirements.txt
├── collect_data.py       # Phase 2: pulls closed issues via GitHub REST API
├── build_features.py     # Phase 3: raw JSON -> clean feature table
├── train_model.py        # Phase 4: trains + evaluates models
├── app.py                # Phase 5: Streamlit demo
└── data/
    ├── raw/
    └── processed/
```

## Things worth writing up in your README/resume bullet

- The censoring problem (closed-only training set) and what it does and
  doesn't let you claim
- Filtering PRs out of the issues endpoint
- Bot-opened issues (e.g. dependabot) -- these have very different
  resolution patterns and probably shouldn't be pooled with human-filed
  issues without a feature flagging them
- Repo-level baseline normalization -- a "3 day resolution" is fast for
  one repo and slow for another depending on maintainer activity level
