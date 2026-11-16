# Local runtime pipeline contract

The canonical path is Streamlit → FastAPI → `src/` NLP core. Upload and confirmation persist server-owned runtime documents; preparation writes normalized/features/chunks/embeddings under `Data/Runtime`. Matching uses `POST /api/v1/matching/run` and explanation uses `POST /api/v1/explanations/generate`.

Runtime endpoints:

- `POST /api/v1/resume/parse`, `/resume/confirm`
- `POST /api/v1/job/parse`, `/job/confirm`
- `POST /api/v1/matching/run` (identity-only, real MDMS)
- `POST /api/v1/explanations/generate` (identity-only, selected candidate)

Runtime artifacts live under `Data/Runtime/resumes`, `Data/Runtime/jobs`, and `Data/Runtime/matching`. `Data/Results` remains research/evaluation output.

Scores are canonical on 0–1; UI may display `score_0_3 = score_0_1 * 3`. The only dimensions are Skill, Experience, Education, and Semantic. `None` means unknown/insufficient; valid `0.0` remains evaluated zero; NOT_APPLICABLE is not a weakness.

Freshness is enforced before matching. Reports are keyed by `(match_run_id, cv_id)`. Runtime XAI uses `pre_explanation_builder.py` as the canonical deterministic compact-facts owner; narrative generation is optional Groq or local offline deterministic output.

Run locally without a Groq key:

```text
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run app/frontend/Home.py
```

`GROQ_API_KEY` and `GROQ_MODEL` are optional. Embeddings use local `sentence-transformers/all-MiniLM-L6-v2`.
