# DP text evaluation survey

Streamlit app backed by MongoDB for the human evaluation study. Respondents see a set of DP text rewrites and pick the most acceptable one. The app assigns questions to balance coverage across all variants and handles concurrent sessions.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your MongoDB URI, admin credentials, and Prolific redirect URLs.

Set the run mode in `.streamlit/config.toml` under `[app] environment`: `"development"` shows reference texts alongside rewrites, `"production"` hides them so respondents judge without anchoring to the original.

## Running

```bash
streamlit run survey.py    # respondent survey
streamlit run admin.py     # admin dashboard (password-protected)
```

## Layout

```
survey/
├── survey.py            Entry point; routes through the pages in order
├── admin.py             Admin dashboard: progress stats and response exports
├── db.py                MongoDB connection and collection setup
├── assignment.py        Question selection and soft-reservation logic
├── services/
│   └── persistence.py   Save/load for drafts, demographics, and final responses
├── state/session.py     Session state
├── models/session.py    Session data model
├── pages/               welcome → background → questions → feedback → thank-you
└── data/                Country reference lists for the demographics page
```
