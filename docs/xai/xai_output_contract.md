# XAI output contract (`xai_v1`)

Pipeline ownership is deliberately separated:

`MDMS → structured XAI evidence → deterministic report → future LLM renderer`.

The MDMS and matchers remain the scoring authority. Facts are copied from normalized CV/JD artifacts and matcher decisions are deterministic. A future LLM may only turn those decisions into wording; it may not add facts, alter scores, infer missing qualifications, or expose embeddings.

## Existing inventory

| Matcher | Existing reusable evidence | Missing for XAI |
|---|---|---|
| Skill | score, coverage, requirement list, required/preferred matches and misses, match method, matched CV skill, provenance, similarity | stable evidence IDs and explicit JD/CV source references |
| Experience | score, coverage, years result, responsibility-level best chunk, chunk ID/text/source, raw similarity, threshold/status | normalized `years_score`, explicit evidence status distinction, registry IDs |
| Education | score, coverage, degree/field result, candidate and requirement values, status | stable CV/JD evidence references and explicit unknown/no-evidence contract |
| Semantic | score, coverage, status, raw similarity | profile scope/source references; vectors must remain private |
| MDMS | final score, status, coverage, effective weights | explicit original weights and per-dimension weighted contributions |

Current artifacts use lowercase statuses (`available`, `evaluated`, `no_evidence`, etc.). The canonical contract normalizes them to uppercase `AVAILABLE`, `MATCHED`, `NO_EVIDENCE`, `UNKNOWN`, and `NOT_APPLICABLE`.

## Canonical object

See `src/Explainability/schemas.py` and the development example. Dimensions preserve matcher scores; they do not recompute them.

Semantic means global profile-to-role alignment only. It is not evidence of a specific skill or experience claim.

## HR sections

The contract supports candidate/job identity, overall MDMS score, four-dimension breakdown, deterministic strength/gap candidates, evidence traceability, and interview-focus triggers. It intentionally contains no generated prose.

## Missing-data wording

- `AVAILABLE` / `MATCHED`: evidence was available or a deterministic match was found.
- `NO_EVIDENCE`: the source was evaluable, but no supporting evidence was found. Safe wording: “No supporting evidence was found in the provided CV.”
- `UNKNOWN`: the source was unavailable or insufficient to evaluate. Safe wording: “The available CV data is insufficient to evaluate this requirement.”
- `NOT_APPLICABLE`: the JD does not impose this requirement; normally omit it from gaps.

`NO_EVIDENCE` must never be rewritten as proof that a candidate lacks a capability. Low similarity is still evaluated evidence and is not silently converted to missing data.

## Provenance and token policy

Evidence is registered once under an `evidence_id`; candidates and explanation payloads refer to that ID. Recommended later defaults are at most one selected evidence item per JD responsibility, short bounded excerpts (roughly 300 characters), and no vectors, unused chunks, full raw CV/JD text, or debug metadata in the LLM payload.

## Deterministic vs LLM ownership

| Information | Source | Deterministic | LLM may modify? |
|---|---|---:|---:|
| Final MDMS score/status/coverage | `mdms.py` | YES | NO |
| Component scores/weights/contributions | MDMS + matchers | YES | NO |
| Skill match/missing requirement | skill matcher | YES | NO |
| Years and responsibility similarity | experience matcher | YES | NO |
| Degree/field result | education matcher | YES | NO |
| Semantic score/scope | semantic matcher | YES | NO |
| Evidence quote/reference | CV/JD artifact registry | YES | NO |
| Strength/gap/interview trigger selection | deterministic report layer | YES | NO |
| Summary wording | future renderer | NO | YES |
| Interview question wording | future renderer, deterministic trigger | NO | YES |

